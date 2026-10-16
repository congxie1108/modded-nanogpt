"""Recompute the submission's statistics from the unmodified logs (stdlib only)."""
import ast
import hashlib
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
ROW = re.compile(r"^step:(\d+)/\d+ val_loss:([\d.]+) val_ema_loss:([\d.]+)", re.M)


def load_runs(folder, expected_seeds):
    runs = {}
    source_hashes = set()
    for path in sorted(folder.glob('H100_*.txt')):
        text = path.read_text()
        source, metadata = text.split('=' * 100, 1)
        ast.parse(source, filename=str(path))
        source_hashes.add(hashlib.sha256(source.encode()).hexdigest())
        seed_match = re.search(r'^(?:Resolved config: |Using )seed=(\d+)\b', metadata, re.M)
        if seed_match is None:
            raise ValueError(f'Missing seed metadata in {path}')
        seed = int(seed_match[1])
        if seed in runs:
            raise ValueError(f'Duplicate seed {seed} in {folder}')
        rows = {}
        for step, raw, ema in ROW.findall(metadata):
            step = int(step)
            if step in rows:
                raise ValueError(f'Duplicate validation step {step} in {path}')
            rows[step] = (float(raw), float(ema))
        if not rows or not all(math.isfinite(v) for row in rows.values() for v in row):
            raise ValueError(f'Missing or non-finite validation losses in {path}')
        runs[seed] = rows
    if set(runs) != set(expected_seeds):
        raise ValueError(f'Unexpected seeds in {folder}: {sorted(runs)}')
    if len(source_hashes) != 1:
        raise ValueError(f'Embedded training source differs between runs in {folder}')
    return runs


def summary(label, runs):
    common = sorted(set.intersection(*(set(rows) for rows in runs.values())))
    for step in common:
        mean = statistics.mean(rows[step][1] for rows in runs.values())
        score = (3.28 - mean) * math.sqrt(len(runs))
        if score >= 0.004:
            print(f'{label}: n={len(runs)}, first passing step={step}, '
                  f'mean={mean:.9f}, score={score:.9f}')
            return step, mean
    raise ValueError(f'{label}: no common step passes the Track 3 criterion')


def main():
    claimed = load_runs(ROOT, range(49, 65))
    tuning = load_runs(ROOT / 'tuning_runs_same_config', range(37, 45))
    controls = load_runs(ROOT / 'controls_46', range(17, 25))
    step, mean = summary('Claimed seeds 49-64', claimed)
    assert step == 2615
    for s in (2610, 2615, 2620, 2630):
        loss = statistics.mean(rows[s][1] for rows in claimed.values())
        score = (3.28 - loss) * math.sqrt(len(claimed))
        print(f'  step={s}, mean={loss:.9f}, score={score:.9f}, pass={score >= 0.004}')
    tuning_step, _ = summary('Tuning seeds 37-44', tuning)
    all_step, _ = summary('All seeds (tuning included)', {**tuning, **claimed})
    control_step, control_mean = summary('#46 same-hardware controls', controls)
    assert (tuning_step, all_step, control_step) == (2635, 2620, 2690)
    raw = statistics.mean(rows[step][0] for rows in claimed.values())
    pairwise = ((control_mean - mean) + (control_step - step) / 100 * 0.0045) / math.sqrt(
        1 / len(controls) + 1 / len(claimed))
    print(f'Raw-weight mean at {step}: {raw:.9f}')
    print(f'Pairwise score vs #46 same-hardware controls: {pairwise:.9f} (required >= 0.004)')
    assert pairwise >= 0.004


if __name__ == '__main__':
    main()
