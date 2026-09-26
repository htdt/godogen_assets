"""
loops.py: seamless cycles for gen-moves' loop moves, in place of kimogen's loop trim. Used by gen_moves.py.

find() searches one generated sample (canonical joint positions and root) for its best cycle: a window [s, s + P]
whose ends match in pose and velocity (the pose distance of the frames s-2..s+2 to s+P-2..s+P+2, root-relative, feet
weighted double), measured in frames of the window's own motion, among the windows that keep the take's motion (mean
joint speed at least MOTION_FLOOR of the take's). Both guard against the still stretch of a hammering take, whose ends
always match. The period P is then refined below a frame: a cycle rarely lasts a whole number of frames, and cutting at
the nearest frame leaves a phase jump at the wrap (a planted foot slides once per cycle).

cut() turns the window into the baked clip: the local rotations and the root are resampled at M + 1 even steps over
[s, s + P] (M = round(P), so the tempo changes by less than half a frame per cycle), the residual mismatch between the
two ends is spread over the cycle (a rotation ramp per joint, a height ramp for the root), forward kinematics rebuilds
the positions, and the clip is re-canonicalized. The result is closed: frame M equals frame 0 and the cycle lasts M
frames, so an engine loops the clip as imported. The root keeps its travel per cycle.

support() is the fraction of frames with a foot joint at rest (under SUPPORT_SPEED): the feet carry the body, on the
floor or on anything above it (stairs, a ladder, a seat), where Kimodo's foot-contact labels only cover the floor.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

VERSION = 1                 # part of a loop move's generation key: a new cut regenerates the loops
BODY =['Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head'] + [
    f'{s}{b}' for s in ('Left', 'Right')
    for b in ('Shoulder', 'Arm', 'ForeArm', 'Hand', 'Leg', 'Shin', 'Foot', 'ToeBase')]
FEET = {'LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase'}
CONTACT = ['LeftFoot', 'LeftToeBase', 'LeftToeEnd', 'RightFoot', 'RightToeBase', 'RightToeEnd']
SEAM = np.arange(-2, 3)     # frames compared around each end: pose and velocity continuity
MIN_LOOP_S = 1.0            # shortest cycle, seconds
MOTION_FLOOR = 0.75         # the cycle's mean joint speed, at least this share of the take's (good loops: 0.78-2.8)
LOOP_ERR_MAX = 0.06         # m, the gate on the seam (walks and idles 0.001-0.03, one strike from rest to rest ~0.05)
SUPPORT_SPEED = 0.1         # m/s: a foot joint slower than this carries weight (planted feet move < 0.05)


def _features(j, r, idx):
    """Root-relative body joint positions [T, B, 3] and their weights (sum 1)."""
    body = [idx[n] for n in BODY]
    w = np.array([2.0 if n in FEET else 1.0 for n in BODY])
    return j[:, body] - r[:, None, :] * np.array([1.0, 0.0, 1.0]), w / w.sum()


def _at(p, t):
    """p sampled at fractional frames t (linear)."""
    t = np.clip(t, 0, len(p) - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(p) - 1)
    f = (t - i0).reshape(-1, *[1] * (p.ndim - 1))
    return p[i0] * (1 - f) + p[i1] * f


def find(j, r, idx, fps):
    """Best cycle of one sample: {start, period (frames, fractional), seam (m), seam_frames (the seam in frames of the
    cycle's own motion), motion (share of the take's)}, or None when the take is too short or too still for one.

    Windows are ranked by seam_frames: a seam jump smaller than a frame's worth of the cycle's motion does not show,
    and ranking by the plain distance would favour the stillest stretch (its ends barely move, so they always match)."""
    p, w = _features(np.asarray(j, float), np.asarray(r, float), idx)
    T = len(p)
    if T < 4:
        return None
    lo = min(int(round(MIN_LOOP_S * fps)), T - 2)
    speed = (np.linalg.norm(np.diff(p, axis=0), axis=-1) * w).sum(-1)          # [T-1], m/frame
    take = max(float(speed.mean()), 1e-6)
    csum = np.concatenate([[0.0], np.cumsum(speed)])
    best = (np.inf, None, None)
    for P in range(lo, T - 1):
        s = np.arange(T - P)
        motion = (csum[s + P] - csum[s]) / P
        dist = (np.linalg.norm(p[:T - P] - p[P:], axis=-1) * w).sum(-1)           # frame t vs t + P
        # seam of a start s: the mean over s + SEAM, clipped to frames that exist on both sides
        k = np.clip(s[:, None] + SEAM[None], 0, T - P - 1)
        cost = dist[k].mean(1) / np.maximum(motion, 1e-3 * take)
        cost[motion < MOTION_FLOOR * take] = np.inf
        i = int(np.argmin(cost))
        if cost[i] < best[0]:
            best = (float(cost[i]), i, P)
    if best[1] is None:
        return None
    _, s, P = best
    motion = (csum[s + P] - csum[s]) / P
    # sub-frame period: the seam at P + fractions of a frame, the far end interpolated
    cands = [Pf for Pf in P + np.linspace(-1, 1, 41) if lo <= Pf <= T - 1 - s]
    ends = s + SEAM
    ends = ends[(ends >= 0) & (ends + max(cands) <= T - 1)]

    def seam(Pf):
        return float((np.linalg.norm(p[ends] - _at(p, ends + Pf), axis=-1) * w).sum(-1).mean())
    Pf = min(cands, key=seam)
    return {'start': int(s), 'period': round(float(Pf), 3), 'seam': round(seam(Pf), 4),
            'seam_frames': round(seam(Pf) / max(motion, 1e-6), 2), 'motion': round(float(motion / take), 3)}


def support(j, idx, fps):
    """Share of frames where a foot joint is at rest, at any height."""
    c = [idx[n] for n in CONTACT]
    v = np.linalg.norm(np.diff(np.asarray(j, float)[:, c], axis=0), axis=-1) * fps
    return float((v < SUPPORT_SPEED).any(axis=1).mean())


def cut(z, start, period, skeleton, canonicalize, idx):
    """Resample, close and re-canonicalize the cycle [start, start + period] of an accepted NPZ (dict of arrays).
    Returns (new arrays, M, R2, p02): the rigid step x -> R2 (x - p02) re-expresses canonical-frame data."""
    import torch
    M = max(2, int(round(period)))
    t = start + np.arange(M + 1) * (period / M)
    lrm = np.asarray(z['local_rot_mats'], float)
    T, J = lrm.shape[:2]
    q = Rot.from_matrix(lrm.reshape(-1, 3, 3)).as_quat().reshape(T, J, 4)
    i0 = np.clip(np.floor(t).astype(int), 0, T - 1)
    i1 = np.minimum(i0 + 1, T - 1)
    f = (t - i0)[:, None, None]
    qa, qb = q[i0], q[i1]
    qb = np.where((qa * qb).sum(-1, keepdims=True) < 0, -qb, qb)
    qs = qa * (1 - f) + qb * f                              # nlerp: neighbouring frames differ by a few degrees
    qs /= np.linalg.norm(qs, axis=-1, keepdims=True)
    rots = Rot.from_quat(qs.reshape(-1, 4)).as_matrix().reshape(M + 1, J, 3, 3)
    root = _at(np.asarray(z['root_positions'], float), t)

    # closure: rotate each joint by a growing share of its end-to-start residual, so frame M lands on frame 0
    resid = Rot.from_matrix(np.einsum('jab,jcb->jac', rots[0], rots[M])).as_rotvec()          # R0 RM^T per joint
    ramp = np.arange(M + 1)[:, None, None] / M
    rots = np.einsum('tjab,tjbc->tjac', Rot.from_rotvec((ramp * resid).reshape(-1, 3)).as_matrix().reshape(
        M + 1, J, 3, 3), rots)
    root[:, 1] += (np.arange(M + 1) / M) * (root[0, 1] - root[M, 1])

    grm, posed, _ = skeleton.fk(torch.from_numpy(rots).float(), torch.from_numpy(root).float())
    grm, posed = grm.numpy().astype(float), posed.numpy().astype(float)
    posed_c, root_c, R2 = canonicalize(posed, root, idx)
    root_i = skeleton.root_idx
    grm = np.einsum('ij,tnjk->tnik', R2, grm)
    rots[:, root_i] = np.einsum('ij,tjk->tik', R2, rots[:, root_i])
    out = dict(z)
    out.update(posed_joints=posed_c.astype(np.float32), root_positions=root_c.astype(np.float32),
               global_rot_mats=grm.astype(np.float32), local_rot_mats=rots.astype(np.float32),
               canonical_rotation=(R2 @ np.asarray(z['canonical_rotation'], float)).astype(np.float32))
    if 'foot_contacts' in z:
        fc = np.asarray(z['foot_contacts'])[np.clip(np.round(t).astype(int), 0, T - 1)]
        fc[M] = fc[0]
        out['foot_contacts'] = fc.astype(np.float32)
    return out, M, R2, root[0] * np.array([1.0, 0.0, 1.0])
