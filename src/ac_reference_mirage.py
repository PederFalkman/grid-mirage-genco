"""GRID MIRAGE AC reference experiment on IEEE 118 (independent Newton-Raphson; NOT GENCO).

Mirror: original + K consistent relabelings (buses, generators and branch order).
Butterfly: seven relative load changes; the REF bus absorbs the balance.
Physics Lens: AC nodal balance residual of every solution.

Writes results/ac_reference/{ac_reference.csv, ac_reference_outputs.npz,
ac_sensitivity.csv, ac_reference.png}. The NPZ follows the compare_genco.py
contract, so the same metric code is exercised on real AC solutions.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import acgrid

SCALES = [-.01, -.005, -.001, 0., .001, .005, .01]


def run(grid, scales=SCALES, permutations=5, seed=118):
    rng = np.random.default_rng(seed)
    n, ng, ne = grid.n, len(grid.pg), len(grid.y_self)
    perms = [(np.arange(n), np.arange(ng), np.arange(ne))]
    perms += [(rng.permutation(n), rng.permutation(ng), rng.permutation(ne)) for _ in range(permutations)]
    original = np.zeros((len(scales), n, 4))
    permuted = np.zeros((len(scales), len(perms), n, 4))
    rows = []
    for si, delta in enumerate(scales):
        base = acgrid.scale_load(grid, delta)
        for k, (bp, gp, ep) in enumerate(perms):
            g = acgrid.permute(base, bp, gp, ep)
            sol = acgrid.newton_pf(g)
            out = np.stack([sol['vm'], sol['va'], sol['pg'], sol['qg']], axis=1)
            permuted[si, k] = out
            if k == 0:
                original[si] = out
            canonical = out[bp]
            dp, dq, inf = acgrid.power_balance_residual(g, sol['vm'], sol['va'], sol['pg'], sol['qg'])
            err = np.abs(canonical - original[si])
            rows.append({
                'relative_load_change': delta, 'permutation': k, 'nr_iterations': sol['iterations'],
                'vm_equivariance_error_pu': err[:, 0].max(), 'va_equivariance_error_rad': err[:, 1].max(),
                'pg_equivariance_error_mw': err[:, 2].max(), 'qg_equivariance_error_mvar': err[:, 3].max(),
                'max_abs_p_residual_mw': np.abs(dp[inf['P']]).max(),
                'max_abs_q_residual_mvar': np.abs(dq[inf['Q']]).max(),
                'slack_pg_mw': sol['pg'][g.bus_type == acgrid.REF][0],
            })
    return rows, original, permuted, np.array([p[0] for p in perms]), np.array(scales)


def sensitivity(original, scales, grid):
    base = int(np.argmin(np.abs(scales)))
    pq = grid.bus_type == acgrid.PQ
    out = []
    for si, delta in enumerate(scales):
        d = original[si] - original[base]
        out.append({'relative_load_change': delta,
                    'max_abs_dvm_pq_pu': np.abs(d[pq, 0]).max(),
                    'mean_dvm_pq_pu': d[pq, 0].mean(),
                    'max_abs_dva_rad': np.abs(d[:, 1]).max(),
                    'slack_dpg_mw': d[grid.bus_type == acgrid.REF, 2][0]})
    return out


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(rows, sens, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.array([r['relative_load_change'] for r in sens]) * 100
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, label in [('vm_equivariance_error_pu', 'Vm [p.u.]'), ('va_equivariance_error_rad', 'Va [rad]'),
                       ('pg_equivariance_error_mw', 'Pg [MW]'), ('qg_equivariance_error_mvar', 'Qg [Mvar]')]:
        y = [max(r[key] for r in rows if r['relative_load_change'] == s['relative_load_change'] and r['permutation'] > 0)
             for s in sens]
        ax[0].semilogy(x, np.maximum(y, 1e-18), marker='o', label=label)
    ax[0].set(xlabel='Load change (%)', ylabel='Max |canonical(permuted) - original|',
              title='Mirror: AC reference equivariance')
    ax[0].legend(fontsize=8)
    ax[1].plot(x, [s['mean_dvm_pq_pu'] * 1e3 for s in sens], marker='o', label='mean ΔVm at PQ buses [10⁻³ p.u.]')
    ax[1].plot(x, [s['max_abs_dva_rad'] * 1e3 for s in sens], marker='s', label='max |ΔVa| [mrad]')
    ax[1].set(xlabel='Load change (%)', title='Butterfly: AC reference response')
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.2)
    fig.suptitle('GRID MIRAGE, IEEE 118 independent AC reference (not GENCO)', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/ac_reference')
    ap.add_argument('--permutations', type=int, default=5)
    ap.add_argument('--seed', type=int, default=118)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grid, pp_solution = acgrid.ieee118_from_pandapower()
    base = acgrid.newton_pf(grid)
    validation = {'source': 'pandapower.networks.case118 (MATPOWER IEEE 118)',
                  'max_abs_vm_diff_vs_pandapower_pu': float(np.abs(base['vm'] - pp_solution['vm']).max()),
                  'max_abs_va_diff_vs_pandapower_rad': float(np.abs(base['va'] - pp_solution['va']).max())}
    rows, original, permuted, perms, scales = run(grid, permutations=args.permutations, seed=args.seed)
    sens = sensitivity(original, scales, grid)
    write_csv(out / 'ac_reference.csv', rows)
    write_csv(out / 'ac_sensitivity.csv', sens)
    np.savez(out / 'ac_reference_outputs.npz', original=original, permuted=permuted,
             permutations=perms, scales=scales, reference=original)
    keys = ['vm_equivariance_error_pu', 'va_equivariance_error_rad', 'pg_equivariance_error_mw',
            'qg_equivariance_error_mvar', 'max_abs_p_residual_mw', 'max_abs_q_residual_mvar']
    summary = {**validation, 'cases': len(rows), **{f'max_{k}': float(max(r[k] for r in rows)) for k in keys}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plot(rows, sens, out / 'ac_reference.png')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
