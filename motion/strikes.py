"""Strike intent checks in source metres, before best-of-N selection. No model or engine dependencies."""
import numpy as np

VERSION = 1


def validate(mv):
    if mv.get('strike') not in (None, 'hand', 'foot'):
        raise ValueError(f'{mv["name"]}: strike must be hand or foot')
    gate = mv.get('strike_gate', {})
    if not isinstance(gate, dict) or set(gate) - {'side', 'min_speed', 'min_excursion'}:
        raise ValueError(f'{mv["name"]}: strike_gate accepts side, min_speed and min_excursion')
    if gate and not mv.get('strike'):
        raise ValueError(f'{mv["name"]}: strike_gate needs strike: hand or foot')
    if gate.get('side', 'either') not in ('left', 'right', 'either'):
        raise ValueError(f'{mv["name"]}: strike_gate.side must be left, right or either')
    for key in ('min_speed', 'min_excursion'):
        v = gate.get(key, 1.0 if key == 'min_speed' else 0.25)
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) or v <= 0:
            raise ValueError(f'{mv["name"]}: strike_gate.{key} must be finite and positive')


def measure(j, r, idx, mv, fps):
    """Pick one authored-side limb. Root-relative movement prevents a running idle from passing as a strike.

    Speed must persist for ~100 ms (at least three intervals), so a single bad frame cannot pass. Excursion is the diameter of the limb path;
    it includes wind-up and follow-through. These are amplitude gates, not proof of a weapon hitting a target.
    """
    gate = mv.get('strike_gate', {})
    sides = ['Left', 'Right'] if gate.get('side', 'either') == 'either' else [gate['side'].title()]
    tips = [s + ('Hand' if mv['strike'] == 'hand' else 'Foot') for s in sides]
    candidates = []
    for tip in tips:
        p = np.asarray(j[:, idx[tip]] - r, float)
        speed = np.linalg.norm(np.diff(p, axis=0), axis=-1) * fps
        width = min(len(speed), max(3, round(fps * 0.1)))
        peak = float(np.lib.stride_tricks.sliding_window_view(speed, width).min(axis=-1).max())
        speed = np.convolve(np.pad(speed, (width // 2, (width - 1) // 2), mode='edge'),
                            np.ones(width) / width, mode='valid')
        excursion = float(np.linalg.norm(p[:, None] - p[None, :], axis=-1).max())
        margin = min(peak / gate.get('min_speed', 1.0), excursion / gate.get('min_excursion', 0.25))
        candidates.append((margin, tip, speed, excursion, peak))
    margin, tip, speed, excursion, peak = max(candidates, key=lambda c: c[0])
    return {'strike_ok': bool(margin >= 1), 'strike_tip': tip, 'strike_speed': peak,
            'strike_excursion': excursion}, speed


def frame_data(j, r, idx, mv, fps):
    if not mv.get('strike'):
        return None
    metrics, speed = measure(j, r, idx, mv, fps)
    peak = int(speed.argmax())
    lo = hi = peak
    threshold = max(0.3 * speed[peak], 0.3)
    while lo > 0 and speed[lo - 1] > threshold:
        lo -= 1
    while hi < len(speed) - 1 and speed[hi + 1] > threshold:
        hi += 1
    end = hi + 1
    return {'startup': lo, 'active': [lo, end], 'contact': peak + 1,
            'contact_estimate': True, 'recovery': len(j) - 1 - end,
            'strike_tip': metrics['strike_tip'], 'peak_speed': round(float(speed.max()), 2),
            'height': mv.get('height')}
