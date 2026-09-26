"""
gen3d: image -> textured GLB with TRELLIS.2-4B, patched (lowmem.py) to fit a 12 GB GPU.

    gen3d tree.png house.png -o out/                        # 1024_cascade, 500k faces, 2048 tex
    gen3d --faces 30000 --tex 1024 crate.png -o out/        # game budget
    gen3d --type 512 --no-preview *.png -o drafts/          # fast drafts
    gen3d --json hero.png -o out/                           # one JSON result line per image on stdout

Writes <out>/<name>.glb, <name>_preview.png (4 views: PBR shaded + normals) and <name>.json (settings, timings,
VRAM). stdout carries only the GLB path per image (or the JSON line with --json), progress goes to stderr; exit 0
when every image succeeded, 1 otherwise. The pipeline's low_vram mode keeps every sub-model on the CPU and moves
only the active one to the GPU.
"""
import os
import sys
import json
import time
import argparse

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from cli_args import ArgumentParser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'TRELLIS.2'))
sys.path.insert(0, ROOT)
MODEL_DIR = os.path.join(ROOT, 'models', 'TRELLIS.2-4B')

# The official DINOv3 and RMBG-2.0 repos are gated; use an ungated byte-identical DINOv3 mirror
# and the MIT BiRefNet (same architecture as RMBG-2.0) for background removal.
DINOV3_REPO = os.environ.get('TRELLIS_DINOV3', 'camenduru/dinov3-vitl16-pretrain-lvd1689m')
REMBG_REPO = os.environ.get('TRELLIS_REMBG', 'ZhengPeng7/BiRefNet')

NEEDED_MODELS = {
    '512': ['shape_slat_flow_model_512', 'tex_slat_flow_model_512'],
    '1024': ['shape_slat_flow_model_1024', 'tex_slat_flow_model_1024'],
    '1024_cascade': ['shape_slat_flow_model_512', 'shape_slat_flow_model_1024', 'tex_slat_flow_model_1024'],
    '1536_cascade': ['shape_slat_flow_model_512', 'shape_slat_flow_model_1024', 'tex_slat_flow_model_1024'],
}
COMMON_MODELS = ['sparse_structure_flow_model', 'sparse_structure_decoder', 'shape_slat_decoder', 'tex_slat_decoder']


def local_config() -> str:
    """Write pipeline_local.json next to pipeline.json with the ungated repos swapped in."""
    with open(os.path.join(MODEL_DIR, 'pipeline.json')) as f:
        cfg = json.load(f)
    cfg['args']['image_cond_model']['args']['model_name'] = DINOV3_REPO
    cfg['args']['rembg_model']['args']['model_name'] = REMBG_REPO
    with open(os.path.join(MODEL_DIR, 'pipeline_local.json'), 'w') as f:
        json.dump(cfg, f, indent=4)
    return 'pipeline_local.json'


def rss_gb() -> float:
    with open('/proc/self/status') as f:
        kb = next(int(l.split()[1]) for l in f if l.startswith('VmRSS'))
    return round(kb / 2**20, 2)


def clean_alpha(img, threshold: int = 24):
    """Drop faint alpha haze (soft shadows from RGBA generators) that would otherwise enlarge the crop."""
    import numpy as np
    from PIL import Image
    if img.mode != 'RGBA':
        return img
    a = np.array(img)
    a[..., 3] = np.where(a[..., 3] < threshold, 0, a[..., 3])
    return Image.fromarray(a, 'RGBA')


def save_preview(mesh, path: str, resolution: int = 512):
    """4 views (front, right, back, left) of the PBR-shaded mesh on top, normals below."""
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap
    hdri = cv2.imread(os.path.join(ROOT, 'TRELLIS.2', 'assets', 'hdri', 'studio.exr'), cv2.IMREAD_UNCHANGED)
    envmap = EnvMap(torch.tensor(cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB), dtype=torch.float32, device='cuda'))
    yaws = [np.pi / 2 + k * np.pi / 2 for k in range(4)]
    extr, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, [0.25] * 4, 2, 40)
    res = render_utils.render_frames(mesh, extr, intr, {'resolution': resolution, 'bg_color': (1, 1, 1)},
                                     envmap=envmap, verbose=False)
    top = np.concatenate(res['shaded'], axis=1)
    bottom = np.concatenate(res['normal'], axis=1)
    Image.fromarray(np.concatenate([top, bottom], axis=0)).save(path)


def main():
    p = ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('images', nargs='+')
    p.add_argument('-o', '--out', default=os.path.join(ROOT, 'outputs'))
    p.add_argument('--type', default='1024_cascade', choices=list(NEEDED_MODELS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--faces', type=int, default=500000, help='decimation target for the exported GLB')
    p.add_argument('--tex', type=int, default=2048, help='baked texture size')
    p.add_argument('--steps', type=int, default=None, help='override sampler steps (default 12 per stage)')
    p.add_argument('--max-tokens', type=int, default=49152, help='cascade token cap (lower = less VRAM)')
    p.add_argument('--suffix', default='', help='appended to output names')
    p.add_argument('--no-preview', action='store_true')
    p.add_argument('--save-latent', action='store_true', help='also save the sampled latents (<name>_latent.pt)')
    p.add_argument('--from-latent', action='store_true',
                   help='inputs are saved *_latent.pt files: skip sampling, only decode and export')
    p.add_argument('--json', action='store_true', help='print one JSON result per image on stdout')
    args = p.parse_args()

    # Keep stdout for results only: TRELLIS.2 and the CUDA extensions print progress to fd 1.
    result_out = os.fdopen(os.dup(1), 'w')
    os.dup2(2, 1)

    try:
        for path in args.images:
            if not os.path.isfile(path):
                raise ValueError(f'input not found: {path}')
        if min(args.faces, args.tex, args.max_tokens) <= 0 or (args.steps is not None and args.steps <= 0):
            raise ValueError('faces, tex, max-tokens and steps must be positive')
        import torch
        import o_voxel
        import lowmem
        from lowmem import LowMemTrellis2Pipeline as Pipeline

        args.out = os.path.abspath(args.out)
        os.makedirs(args.out, exist_ok=True)
        t0 = time.time()
        Pipeline.model_names_to_load = COMMON_MODELS + NEEDED_MODELS[args.type]
        pipeline = Pipeline.from_pretrained(MODEL_DIR, config_file=local_config())
        pipeline.low_vram = True
        pipeline.cuda()
        lowmem.trim_host_memory()
    except Exception as e:  # noqa: BLE001 - environment / model problems end the run
        import traceback
        traceback.print_exc()
        if args.json:
            print(json.dumps({'error': f'{type(e).__name__}: {e}'}), file=result_out, flush=True)
        sys.exit(1)
    print(f'[load] {time.time() - t0:.1f}s, RSS {rss_gb()} GB', file=sys.stderr)

    sampler_override = {'steps': args.steps} if args.steps else {}
    failed = []
    for path in args.images:
        try:
            stats = process(pipeline, path, args, sampler_override, o_voxel)
            print(json.dumps(stats) if args.json else stats['output'], file=result_out, flush=True)
        except Exception as e:  # noqa: BLE001 - report per image, keep going with the rest
            import traceback
            traceback.print_exc()
            msg = ('CUDA out of memory; try --type 512, a lower --max-tokens or fewer --faces' if lowmem.is_oom(e)
                   else f'{type(e).__name__}: {e}')
            latent = getattr(pipeline, 'info', {}).get('latent')  # saved by this image's run, before decoding
            if latent and os.path.exists(latent):
                msg += (f'; the sampled latent is kept: gen3d --from-latent {latent} retries only decode and export '
                        '(same -o, --faces, --tex)')
            print(f'[error] {path}: {msg}', file=sys.stderr)
            if args.json:
                print(json.dumps({'input': os.path.abspath(path), 'error': msg}), file=result_out, flush=True)
            failed.append(path)
        torch.cuda.empty_cache()
    if failed:
        print(f'failed: {failed}', file=sys.stderr)
        sys.exit(1)


def decode_saved_latent(pipeline, path):
    import torch
    from trellis2.modules.sparse import SparseTensor
    lat = torch.load(path)
    coords = lat['coords'].cuda()
    pipeline.info = {'pipeline_type': 'from_latent', 'resolution': lat['res'], 'hr_tokens': int(coords.shape[0])}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        mesh = pipeline.decode_latent(SparseTensor(lat['shape_feats'].cuda(), coords),
                                      SparseTensor(lat['tex_feats'].cuda(), coords), lat['res'])[0]
    pipeline.info['decode_peak_vram_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return mesh


def process(pipeline, path, args, sampler_override, o_voxel):
    import torch
    from PIL import Image
    name = os.path.splitext(os.path.basename(path))[0] + args.suffix
    stats = {'input': os.path.abspath(path), 'type': args.type, 'seed': args.seed, 'faces_target': args.faces,
             'tex': args.tex}
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    # always saved before decoding: a decode or export that fails keeps it, so a retry skips the sampling
    pipeline.save_latent_path = None if args.from_latent else os.path.join(args.out, f'{name}_latent.pt')
    if args.from_latent:
        name = os.path.basename(path).replace('_latent.pt', '')  # already carries the original suffix
        mesh = decode_saved_latent(pipeline, path)
    else:
        image = clean_alpha(Image.open(path))
        mesh = pipeline.run(image, seed=args.seed, pipeline_type=args.type, max_num_tokens=args.max_tokens,
                            sparse_structure_sampler_params=sampler_override,
                            shape_slat_sampler_params=sampler_override,
                            tex_slat_sampler_params=sampler_override)[0]
    stats['gen_s'] = round(time.time() - t, 1)
    stats['gen_peak_vram_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    stats.update(pipeline.info)
    stats['raw_faces'] = int(mesh.faces.shape[0])
    stats['host_rss_gb'] = rss_gb()
    torch.cuda.empty_cache()  # CuMesh (simplify, remesh, bake) allocates outside PyTorch's cache
    mesh.simplify(16777216)  # nvdiffrast limit

    if not args.no_preview:
        stats['preview'] = os.path.join(args.out, f'{name}_preview.png')
        save_preview(mesh, stats['preview'])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
        attr_layout=mesh.layout, voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.faces, texture_size=args.tex,
        remesh=True, remesh_band=1, remesh_project=0, verbose=False,
    )
    glb_path = os.path.join(args.out, f'{name}.glb')
    glb.export(glb_path, extension_webp=True)
    stats['export_s'] = round(time.time() - t, 1)
    stats['export_peak_vram_gb'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    stats['glb_faces'] = int(len(glb.faces)) if hasattr(glb, 'faces') else None
    stats['glb_mb'] = round(os.path.getsize(glb_path) / 2**20, 1)
    stats['output'] = glb_path
    latent = stats.pop('latent', None)
    if latent and not args.save_latent:
        os.remove(latent)
    with open(os.path.join(args.out, f'{name}.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats), file=sys.stderr)
    return stats


if __name__ == '__main__':
    main()
