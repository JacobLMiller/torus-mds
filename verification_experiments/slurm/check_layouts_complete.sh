#!/bin/bash
# Read-only completeness check for an embed run's output tree.
#
# For each shard's runs.csv, compares every method's status=="ok" row count
# against s_gd2's row count in that same shard (s_gd2 is never stripped by
# rerun_converged.sh, so it's the ground truth for "how many graphs belong in
# this shard"). Reports any shard where a method is short, plus any non-"ok"
# status rows (failures) anywhere in the tree.
#
# Usage:
#   bash verification_experiments/slurm/check_layouts_complete.sh [LAYOUTS_ROOT]
# Defaults to LAYOUTS_ROOT=layouts_converged. Run from the repo root, on a
# node with the torus-mds conda env (needs pandas).

set -euo pipefail

LAYOUTS_ROOT="${1:-layouts_converged}"

source "${MINIFORGE_ROOT:-$HOME/nobackup/miniforge3}/etc/profile.d/conda.sh"
conda activate torus-mds

if [ ! -d "$LAYOUTS_ROOT" ]; then
    echo "No such directory: $LAYOUTS_ROOT" >&2
    exit 1
fi

find "$LAYOUTS_ROOT" -mindepth 1 -name runs.csv | sort -V | python3 -c "
import sys
import pandas as pd

# wrap_python is deliberately capped at n <= wrap-python-max-n (see
# run_embed_array.sbatch), so its 'skipped_too_large' rows and lower ok-count
# vs s_gd2 are expected, not a sign of an incomplete run -- only the
# TorusMDS* methods (the ones rerun_converged.sh actually strips and
# resubmits) are compared 1:1 against s_gd2's graph count below.

any_incomplete = False
any_failed = False

for path in sys.stdin.read().splitlines():
    df = pd.read_csv(path)
    if len(df) == 0:
        print(f'{path}: EMPTY runs.csv')
        any_incomplete = True
        continue

    ok = df[df['status'] == 'ok']
    expected = ok[ok['method'] == 's_gd2']['exp_idx'].nunique() if (ok['method'] == 's_gd2').any() else None

    torus_methods = [m for m in df['method'].unique() if m.startswith('TorusMDS')]
    counts = ok[ok['method'].isin(torus_methods)].groupby('method')['exp_idx'].nunique()
    short = {}
    if expected is not None:
        for m in torus_methods:
            n = counts.get(m, 0)
            if n < expected:
                short[m] = (n, expected)
    if short:
        any_incomplete = True
        detail = ', '.join(f'{m}: {n}/{exp}' for m, (n, exp) in short.items())
        print(f'{path}: incomplete -- {detail}')

    # Failures other than the expected wrap_python size cap.
    bad = df[(df['status'] != 'ok') & ~((df['method'] == 'wrap_python') & (df['status'] == 'skipped_too_large'))]
    if len(bad) > 0:
        any_failed = True
        reasons = bad['status'].value_counts().to_dict()
        print(f'{path}: {len(bad)} unexpected non-ok rows -- {reasons}')

if not any_incomplete and not any_failed:
    print('All shards complete: every TorusMDS* method matches s_gd2\'s graph count, no unexpected failures.')
"
