"""Figures from real GENCO GRID MIRAGE CSVs written by genco_mirage.py (no other inputs)."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def read(path):
    with Path(path).open() as f:
        return [{k: (float(v) if k not in ('dtype',) else v) for k, v in r.items()} for r in csv.DictReader(f)]


def by_scale(rows, key, reduce=max, **match):
    acc = defaultdict(list)
    for r in rows:
        if all(r[k] == v for k, v in match.items()):
            acc[r['relative_load_change']].append(r[key])
    xs = sorted(acc)
    return np.array(xs) * 100, np.array([reduce(acc[x]) for x in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='?', default='results/genco_n50')
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = Path(a.results)
    mirror, butterfly, physics = read(d / 'mirror.csv'), read(d / 'butterfly.csv'), read(d / 'physics_lens.csv')

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for i, dt in enumerate(['float32', 'float64']):
        for ch, label in [('vm_pu', 'Vm [p.u.]'), ('va_rad', 'Va [rad]'), ('pg_mw', 'Pg [MW]'), ('qg_mvar', 'Qg [Mvar]')]:
            x, y = by_scale(mirror, f'{ch}_equivariance_max', dtype=dt)
            ax[i].semilogy(x, np.maximum(y, 1e-18), marker='o', label=label)
        ax[i].set(title=f'Mirror, GENCO {dt}', xlabel='Load change (%)')
        ax[i].grid(alpha=.2)
    ax[0].set_ylabel('max |canonical(permuted) - original|')
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / 'mirror.png', dpi=160)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    mean = lambda v: float(np.mean(v))  # noqa: E731
    for key, label in [('genco_mean_dvm_pu', 'GENCO'), ('reference_mean_dvm_pu', 'AC reference')]:
        x, y = by_scale(butterfly, key, reduce=mean)
        ax[0].plot(x, y * 1e3, marker='o', label=label)
    ax[0].set(title='Butterfly: mean ΔVm at PQ buses', xlabel='Load change (%)', ylabel='10⁻³ p.u.')
    for key, label in [('genco_slack_dpg_mw', 'GENCO'), ('reference_slack_dpg_mw', 'AC reference')]:
        x, y = by_scale(butterfly, key, reduce=mean)
        ax[1].plot(x, y, marker='o', label=label)
    ax[1].set(title='Butterfly: slack ΔPg', xlabel='Load change (%)', ylabel='MW')
    for key, label in [('genco_p_residual_mw_pq_pv_mean', 'GENCO |ΔP| PQ+PV [MW]'),
                       ('genco_q_residual_mvar_pq_mean', 'GENCO |ΔQ| PQ [Mvar]')]:
        x, y = by_scale(physics, key, reduce=mean)
        ax[2].plot(x, y, marker='o', label=label)
    ax[2].set(title='Physics Lens: mean nodal residual', xlabel='Load change (%)')
    for axis in ax:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / 'butterfly_physics.png', dpi=160)
    print('Wrote', d / 'mirror.png', d / 'butterfly_physics.png')


if __name__ == '__main__':
    main()
