"""Check that lowmem's chunked final decoder stage and chunked mesh extraction reproduce the unchunked ones exactly (run
after changing lowmem.py or updating TRELLIS.2). Needs a latent saved with `gen3d --save-latent`:

    mesh/.conda/bin/python mesh/test_lowmem.py outputs/<name>_latent.pt
"""
import os, sys, json
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'TRELLIS.2')]
import torch
import lowmem
from trellis2 import models
from trellis2.modules.sparse import SparseTensor

lat = torch.load(sys.argv[1])
ck = os.path.join(ROOT, 'models', 'TRELLIS.2-4B', 'ckpts')
cfg = json.load(open(os.path.join(ROOT, 'models', 'TRELLIS.2-4B', 'pipeline.json')))['args']
shape_dec = models.from_pretrained(os.path.join(ck, 'shape_dec_next_dc_f16c32_fp16')).cuda().eval()
tex_dec = models.from_pretrained(os.path.join(ck, 'tex_dec_next_dc_f16c32_fp16')).cuda().eval()
shape_dec.set_resolution(lat['res'])
coords = lat['coords'].cuda()
def norm(f, key):
    return f * torch.tensor(cfg[key]['std']).cuda()[None] + torch.tensor(cfg[key]['mean']).cuda()[None]
shape = SparseTensor(lat['shape_feats'].cuda(), coords)
tex = SparseTensor(lat['tex_feats'].cuda(), coords)

def decode(chunk):
    lowmem.FINAL_CHUNK = chunk
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        # raw decoder outputs (before mesh extraction) of both decoders
        h, subs = lowmem._decoder_forward_lowmem(shape_dec, shape, return_subs=True)
        t = lowmem._decoder_forward_lowmem(tex_dec, tex, guide_subs=subs)
    peak = torch.cuda.max_memory_allocated() / 2**30
    def canon(st):
        key = st.coords[:, 1:].long()
        key = (key[:, 0] * 4096 + key[:, 1]) * 4096 + key[:, 2]
        o = key.argsort()
        return st.coords[o], st.feats[o].float()
    return canon(h), canon(t), subs[-1].feats.float(), peak

(hc, hf), (tc, tf), s_full, p_full = decode(10**12)
(hc2, hf2), (tc2, tf2), s_chunk, p_chunk = decode(20_000)
print('voxels', hc.shape[0], hc2.shape[0], '| tex voxels', tc.shape[0], tc2.shape[0])
print('shape coords equal', torch.equal(hc, hc2), 'max|dfeat|', (hf - hf2).abs().max().item())
print('tex coords equal', torch.equal(tc, tc2), 'max|dfeat|', (tf - tf2).abs().max().item())
print('last subdiv logits max|d|', (s_full - s_chunk).abs().max().item())
print(f'peak VRAM unchunked {p_full:.2f} GB, chunked {p_chunk:.2f} GB')

# mesh extraction: chunked edge lookups (small chunks, so any latent has several) against the original
lowmem.FINAL_CHUNK, lowmem.EDGE_CHUNK = 20_000, 1 << 16
with torch.no_grad():
    h, _ = lowmem._decoder_forward_lowmem(shape_dec, shape, return_subs=True)
    args = (h.coords[:, 1:], (1 + 2 * shape_dec.voxel_margin) * torch.sigmoid(h.feats[:, 0:3]) - shape_dec.voxel_margin,
            h.feats[:, 3:6] > 0, torch.nn.functional.softplus(h.feats[:, 6:7]))
    box = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    torch.cuda.reset_peak_memory_stats()
    v1, f1 = lowmem._orig_fdg_to_mesh(*args, aabb=box, grid_size=lat['res'])
    p1 = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    v2, f2 = lowmem._fdg_to_mesh_chunked(*args, aabb=box, grid_size=lat['res'])
    p2 = torch.cuda.max_memory_allocated() / 2**30
print('extraction vertices equal', torch.equal(v1, v2), 'faces equal', torch.equal(f1, f2),
      f'| peak VRAM {p1:.2f} GB, chunked {p2:.2f} GB')
