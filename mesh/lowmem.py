"""
Memory patches that let TRELLIS.2's 1024 pipelines run on a 12 GB card.

Upstream stops shrinking the cascade resolution at 1024, so a dense shape such as a leafy tree can create
enough high-resolution voxels to overflow the 12 GB shape decoder even when sampling itself fits in about 5 GB.
Here:
  * the cascade can drop below 1024 (in steps of 128) until the token count fits `max_num_tokens`;
  * the sampling tensors are freed before decoding (upstream issue #188);
  * the per-voxel MLPs in the decoder's ConvNeXt blocks run in chunks, and the decoders' last upsampling stage runs
    on spatial slabs with a halo; both give bit-identical results (test_lowmem.py) with a fraction of
    the peak memory;
  * the mesh extraction after the shape decoder looks up voxel neighbours in slices (bit-identical too);
  * PyTorch's cached VRAM is released before CuMesh (hole filling, simplification, remeshing) allocates its own;
  * the GLB export's dual-contouring remesh retries on a coarser grid if it runs out of memory;
  * token counts, the chosen resolution and per-stage VRAM peaks are recorded in `pipeline.info`.
"""
import gc
import ctypes
import numpy as np
import torch
import torch.nn.functional as F
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.models.sc_vaes import sparse_unet_vae

MLP_CHUNK = 1 << 18  # voxels per MLP chunk


def _convnext_forward_chunked(self, x):
    h = self.conv(x)
    feats = h.feats
    out = torch.empty_like(x.feats)
    for i in range(0, feats.shape[0], MLP_CHUNK):
        f = self.norm(feats[i:i + MLP_CHUNK])
        out[i:i + MLP_CHUNK] = self.mlp(f).to(out.dtype) + x.feats[i:i + MLP_CHUNK]
    del h, feats
    return x.replace(out)


sparse_unet_vae.SparseConvNeXtBlock3d._forward = _convnext_forward_chunked


# ---- chunked final upsampling stage of the sparse VAE decoders ----
# The last channel-to-spatial block (512^3 -> 1024^3 for the 1024 pipelines) plus the output layers dominate decode
# memory (~6M output voxels for a leafy tree). That block is two 3x3x3 convolutions around a 2x upsample, so an
# output voxel only depends on input voxels within 2 cells: running it on x-axis slabs with a 2-cell halo and keeping
# the children of each slab's core voxels gives exactly the unchunked result.
FINAL_CHUNK = 300_000  # input voxels per slab (the last level has ~4x as many output voxels)
HALO = 2
_orig_decoder_forward = sparse_unet_vae.SparseUnetVaeDecoder.forward


def _sub_tensor(t, idx):
    from trellis2.modules.sparse import SparseTensor
    return SparseTensor(t.feats[idx], t.coords[idx], scale=t._scale)


def _chunked_final(dec, block, h, guide, out_dtype):
    from trellis2.modules.sparse import SparseTensor
    xs = h.coords[:, 1]
    n = xs.shape[0]
    k = -(-n // FINAL_CHUNK)
    lo, hi = int(xs.min()), int(xs.max()) + 1
    edges = sorted(set(round(lo + (hi - lo) * i / k) for i in range(k + 1)))
    feats, coords, sub_full, out_scale = [], [], None, None
    for a, b in zip(edges[:-1], edges[1:]):
        idx = ((xs >= a - HALO) & (xs < b + HALO)).nonzero()[:, 0]
        if idx.numel() == 0:
            continue
        hs = _sub_tensor(h, idx)
        if dec.pred_subdiv:
            out, sub = block(hs)
        else:
            out = block(hs, subdiv=_sub_tensor(guide, idx) if guide is not None else None)
        out_scale = out._scale
        keep = ((out.coords[:, 1] // 2) >= a) & ((out.coords[:, 1] // 2) < b)
        f = out.feats[keep].type(out_dtype)
        f = F.layer_norm(f, f.shape[-1:])
        feats.append(torch.nn.Linear.forward(dec.output_layer, f))
        coords.append(out.coords[keep])
        if dec.pred_subdiv:
            if sub_full is None:
                sub_full = torch.empty(n, sub.feats.shape[1], dtype=sub.feats.dtype, device=sub.feats.device)
            core = (xs[idx] >= a) & (xs[idx] < b)
            sub_full[idx[core]] = sub.feats[core]
        del out, hs, f, keep
    h_out = SparseTensor(torch.cat(feats), torch.cat(coords), scale=out_scale)
    sub = SparseTensor(sub_full, h.coords, scale=h._scale) if dec.pred_subdiv else None
    return h_out, sub


def _decoder_forward_lowmem(self, x, guide_subs=None, return_subs=False):
    if self.training or len(self.blocks[-1]) != 0:
        return _orig_decoder_forward(self, x, guide_subs=guide_subs, return_subs=return_subs)
    h = self.from_latent(x).type(self.dtype)
    subs = []
    last = len(self.blocks) - 2
    for i in range(len(self.blocks) - 1):
        res = self.blocks[i]
        for j, block in enumerate(res):
            if j < len(res) - 1:
                h = block(h)
                continue
            guide = guide_subs[i] if guide_subs is not None else None
            if i == last and h.feats.shape[0] > FINAL_CHUNK:
                h, sub = _chunked_final(self, block, h, guide, x.dtype)
                if self.pred_subdiv:
                    subs.append(sub)
                return (h, subs) if return_subs else h
            if self.pred_subdiv:
                h, sub = block(h)
                sub._spatial_cache = {}  # don't keep this level's conv neighbour maps alive via `subs`
                subs.append(sub)
            else:
                h = block(h, subdiv=guide)
    h = h.type(x.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = self.output_layer(h)
    return (h, subs) if return_subs else h


sparse_unet_vae.SparseUnetVaeDecoder.forward = _decoder_forward_lowmem


# ---- mesh extraction: edge lookups in chunks ----
# flexible_dual_grid_to_mesh builds the 12 edge neighbours of every voxel at once ((N, 3, 4, 3) int32) plus the hash
# keys of the intersected ones: ~300 bytes per voxel of temporaries, 4 GB for a 13M-voxel woodpile on top of the
# decoder's output. The same lookups run over slices of voxels here; the quads come out in the same order.
from o_voxel.convert import flexible_dual_grid as _fdg
from trellis2.models.sc_vaes import fdg_vae as _fdg_vae

_orig_fdg_to_mesh = _fdg.flexible_dual_grid_to_mesh
EDGE_CHUNK = 1 << 21  # voxels per edge-lookup chunk


def _fdg_to_mesh_chunked(coords, dual_vertices, intersected_flag, split_weight, aabb, voxel_size=None, grid_size=None,
                         train=False):
    if train or split_weight is None or grid_size is None or coords.shape[0] <= EDGE_CHUNK:
        return _orig_fdg_to_mesh(coords, dual_vertices, intersected_flag, split_weight, aabb, voxel_size=voxel_size,
                                 grid_size=grid_size, train=train)
    f = _orig_fdg_to_mesh
    if not hasattr(f, 'edge_neighbor_voxel_offset'):  # the static tables the original builds on its first call
        f.edge_neighbor_voxel_offset = torch.tensor([
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ], dtype=torch.int, device=coords.device).unsqueeze(0)
        f.quad_split_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=coords.device)
        f.quad_split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=coords.device)
    aabb = torch.as_tensor(np.asarray(aabb), dtype=torch.float32, device=coords.device) \
        if not isinstance(aabb, torch.Tensor) else aabb
    grid = torch.tensor([grid_size] * 3 if isinstance(grid_size, int) else list(grid_size), dtype=torch.int32,
                        device=coords.device)
    voxel = (aabb[1] - aabb[0]) / grid
    N = dual_vertices.shape[0]
    hashmap = _fdg._init_hashmap(grid, 2 * N, device=coords.device)
    _fdg._C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
                                              *grid.tolist())
    quads = []
    for i in range(0, N, EDGE_CHUNK):
        c = coords[i:i + EDGE_CHUNK]
        connected = (c.reshape(-1, 1, 1, 3) + f.edge_neighbor_voxel_offset)[intersected_flag[i:i + EDGE_CHUNK]]
        M = connected.shape[0]
        key = torch.cat([torch.zeros((M * 4, 1), dtype=torch.int, device=coords.device), connected.reshape(-1, 3)], 1)
        del connected
        idx = _fdg._C.hashmap_lookup_3d_cuda(*hashmap, key, *grid.tolist()).reshape(M, 4).int()
        del key
        quads.append(idx[(idx != 0xffffffff).all(dim=1)])
    del hashmap
    quad_indices = torch.cat(quads)
    del quads
    vertices = (coords.float() + dual_vertices) * voxel + aabb[0].reshape(1, 3)
    w = split_weight[quad_indices]
    triangles = torch.where(w[:, 0] * w[:, 2] > w[:, 1] * w[:, 3], quad_indices[:, f.quad_split_1],
                            quad_indices[:, f.quad_split_2]).reshape(-1, 3)
    return vertices, triangles


_fdg_vae.flexible_dual_grid_to_mesh = _fdg_to_mesh_chunked


# ---- GLB export: remesh fallback ----
# o_voxel.postprocess.to_glb rebuilds the surface with narrow-band dual contouring at the voxel resolution. For a
# 1024 leafy tree the band holds tens of millions of cells and the candidate expansion alone needs >4 GB. On OOM, retry
# at a coarser grid (1024 -> 768 -> 512): the result is decimated to the export face budget anyway, and colours are baked
# from the full-resolution voxel attributes afterwards.
import cumesh.remeshing as _remeshing
_orig_remesh = _remeshing.remesh_narrow_band_dc


def _remesh_with_fallback(vertices, faces, center, scale, resolution, band=1, project_back=0, verbose=False, bvh=None):
    while True:
        try:
            out = _orig_remesh(vertices, faces, center=center, scale=scale, resolution=resolution, band=band,
                               project_back=project_back, verbose=verbose, bvh=bvh)
            gc.collect()
            torch.cuda.empty_cache()  # the next step (CuMesh simplify) allocates outside PyTorch's cache
            return out
        except torch.OutOfMemoryError:
            if resolution <= 256:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            resolution -= 256  # 1024 -> 768 -> 512 (each still halves down to a <=32 base grid)
            print(f'[lowmem] remesh out of memory, retrying at resolution {resolution}')


_remeshing.remesh_narrow_band_dc = _remesh_with_fallback


def is_oom(e: BaseException) -> bool:
    """PyTorch OOMs and CuMesh's own CUDA 'out of memory' RuntimeErrors."""
    return isinstance(e, torch.OutOfMemoryError) or (isinstance(e, RuntimeError) and 'out of memory' in str(e))


import cumesh as _cumesh
_orig_simplify = _cumesh.CuMesh.simplify


def _simplify_retry(self, *args, **kwargs):
    try:
        return _orig_simplify(self, *args, **kwargs)
    except RuntimeError as e:
        if not is_oom(e):
            raise
        gc.collect()
        torch.cuda.empty_cache()
        print('[lowmem] CuMesh simplify out of memory, retrying after freeing the PyTorch cache')
        return _orig_simplify(self, *args, **kwargs)


_cumesh.CuMesh.simplify = _simplify_retry


def trim_host_memory():
    """Return freed heap pages to the OS (model construction leaves large fp32 holes behind)."""
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


def _gb(x):
    return round(x / 2**30, 2)


class LowMemTrellis2Pipeline(Trellis2ImageTo3DPipeline):
    min_resolution = 512 + 128  # floor for the adaptive cascade

    def sample_shape_slat_cascade(self, lr_cond, cond, flow_model_lr, flow_model, lr_resolution, resolution,
                                  coords, sampler_params={}, max_num_tokens=49152):
        from trellis2.modules.sparse import SparseTensor
        # low-resolution pass
        noise = SparseTensor(feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device), coords=coords)
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(flow_model_lr, noise, **lr_cond, **sampler_params,
                                              verbose=True, tqdm_desc='Sampling shape SLat (LR)').samples
        flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        self.info['lr_tokens'] = int(coords.shape[0])

        # upsample the LR latent to high-resolution voxel coordinates
        dec = self.models['shape_slat_decoder']
        dec.to(self.device)
        hr_coords = dec.upsample(slat, upsample_times=4)
        dec.cpu()
        del slat, noise
        torch.cuda.empty_cache()

        hr_resolution = resolution
        while True:
            quant = torch.cat([hr_coords[:, :1],
                               ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int()], dim=1)
            coords = quant.unique(dim=0)
            if coords.shape[0] < max_num_tokens or hr_resolution <= self.min_resolution:
                break
            hr_resolution -= 128
        self.info['hr_tokens'] = int(coords.shape[0])
        self.info['resolution'] = hr_resolution
        if hr_resolution != resolution:
            print(f'[lowmem] {coords.shape[0]} tokens: resolution reduced {resolution} -> {hr_resolution}')

        # high-resolution pass
        noise = SparseTensor(feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device), coords=coords)
        flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(flow_model, noise, **cond, **sampler_params,
                                              verbose=True, tqdm_desc='Sampling shape SLat (HR)').samples
        flow_model.cpu()
        slat = slat * std + mean
        return slat, hr_resolution

    @torch.no_grad()
    def run(self, image, seed=42, pipeline_type='1024_cascade', max_num_tokens=49152, preprocess_image=True,
            sparse_structure_sampler_params={}, shape_slat_sampler_params={}, tex_slat_sampler_params={}, **_):
        self.info = {'pipeline_type': pipeline_type}
        torch.cuda.reset_peak_memory_stats()
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure(cond_512, ss_res, 1, sparse_structure_sampler_params)

        if pipeline_type == '512':
            self.info['hr_tokens'] = int(coords.shape[0])
            shape_slat = self.sample_shape_slat(cond_512, self.models['shape_slat_flow_model_512'], coords,
                                                shape_slat_sampler_params)
            tex_slat = self.sample_tex_slat(cond_512, self.models['tex_slat_flow_model_512'], shape_slat,
                                            tex_slat_sampler_params)
            res = 512
        elif pipeline_type == '1024':
            self.info['hr_tokens'] = int(coords.shape[0])
            shape_slat = self.sample_shape_slat(cond_1024, self.models['shape_slat_flow_model_1024'], coords,
                                                shape_slat_sampler_params)
            tex_slat = self.sample_tex_slat(cond_1024, self.models['tex_slat_flow_model_1024'], shape_slat,
                                            tex_slat_sampler_params)
            res = 1024
        else:
            target = 1024 if pipeline_type == '1024_cascade' else 1536
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024, self.models['shape_slat_flow_model_512'],
                self.models['shape_slat_flow_model_1024'], 512, target, coords, shape_slat_sampler_params,
                max_num_tokens)
            tex_slat = self.sample_tex_slat(cond_1024, self.models['tex_slat_flow_model_1024'], shape_slat,
                                            tex_slat_sampler_params)
        self.info['sampling_peak_vram_gb'] = _gb(torch.cuda.max_memory_allocated())

        if getattr(self, 'save_latent_path', None):
            torch.save({'shape_feats': shape_slat.feats.cpu(), 'tex_feats': tex_slat.feats.cpu(),
                        'coords': shape_slat.coords.cpu(), 'res': res}, self.save_latent_path)
            self.info['latent'] = self.save_latent_path
        del cond_512, cond_1024, coords
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        out = self.decode_latent(shape_slat, tex_slat, res)
        self.info['decode_peak_vram_gb'] = _gb(torch.cuda.max_memory_allocated())
        return out

    @torch.no_grad()
    def decode_latent(self, shape_slat, tex_slat, resolution):
        # Same as upstream, but PyTorch's cached VRAM is released before CuMesh runs: CuMesh allocates with
        # cudaMalloc directly and cannot reuse blocks that PyTorch's caching allocator still holds.
        from trellis2.representations import MeshWithVoxel
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        torch.cuda.empty_cache()
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        del subs
        gc.collect()
        torch.cuda.empty_cache()
        out = []
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out.append(MeshWithVoxel(m.vertices, m.faces, origin=[-0.5, -0.5, -0.5], voxel_size=1 / resolution,
                                     coords=v.coords[:, 1:], attrs=v.feats,
                                     voxel_shape=torch.Size([*v.shape, *v.spatial_shape]), layout=self.pbr_attr_layout))
        self.info['voxels'] = int(tex_voxels.coords.shape[0])
        return out
