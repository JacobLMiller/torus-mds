"""
Attach separately-scored per-shard results (e.g. results/parallelogram/, from
``compute_metrics.py --methods TorusMDS_parallelogram``) to an already-merged
results/<family>_normalized_comparison.csv.

The extra CSVs keep shard-local exp_idx, while merge_results.py renumbered
each shard's sorted unique exp_idx into a contiguous global block. This
rebuilds that map per source_shard and moves the extra rows onto the same
global ids. Extra rows for graphs the merged CSV never scored are dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def attach_shard_results(df: pd.DataFrame, extra_dir: str | Path, results_path: str | Path,
                         ref_method: str = 's_gd2') -> pd.DataFrame:
    extra_dir, results_path = Path(extra_dir), Path(results_path)
    frames = []
    for shard, g in df.groupby('source_shard', sort=False):
        extra_csv = extra_dir / Path(shard).name
        if not extra_csv.exists():
            continue
        local_ids = sorted(pd.read_csv(results_path / Path(shard).name)['exp_idx'].unique())
        id_map = dict(zip(local_ids, sorted(g['exp_idx'].unique())))
        extra = pd.read_csv(extra_csv)
        extra = extra[extra['exp_idx'].isin(id_map)].assign(
            exp_idx=lambda d: d['exp_idx'].map(id_map), source_shard=shard)
        frames.append(extra)
    if not frames:
        return df
    added = pd.concat(frames, ignore_index=True)

    # Sanity check: every remapped row must land on the same graph as
    # ref_method's row with that exp_idx.
    keys = [c for c in ('n', 'k', 'graph_type', 'name') if c in added.columns and c in df.columns]
    ref = df[df['method'] == ref_method][['exp_idx', *keys]]
    chk = added.merge(ref, on='exp_idx', suffixes=('', '_ref'))
    assert len(chk) == len(added), 'extra rows without a matching reference row'
    for c in keys:
        assert (chk[c] == chk[f'{c}_ref']).all(), f'exp_idx remap mismatch on {c!r}'

    return pd.concat([df, added], ignore_index=True)
