"""Hybrid GENCO + Newton-Raphson: does one physics step fix the Butterfly attenuation?

Uses the saved float32 GENCO predictions of a genco_mirage.py run (scenario*_float32.npz,
canonical order, original representation) and rebuilds the same grids through
GraphKit's own preprocessing. For every scenario and load level:
  * GENCO alone,
  * GENCO + k Newton-Raphson steps (k = 1, 2) warm-started from GENCO's Vm/Va,
  * iterations to converge (mismatch < 1e-8 p.u.) from GENCO warm start vs flat start,
compared with the exact AC solution. Writes hybrid.csv, hybrid_summary.json/.md and
hybrid.png (headline figure) into the run directory.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acgrid  # noqa: E402
import genco_mirage as gm  # noqa: E402

TOL = 1e-8


def gain(d, r):
    den = float(r @ r)
    return float(d @ r / den) if den > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='?', default='results/genco_n50')
    ap.add_argument('--raw-dir', required=True)
    a = ap.parse_args()
    d = Path(a.results)
    meta = json.loads((d / 'metadata.json').read_text())
    scales = meta['scales']
    base = scales.index(0.0)
    samples = gm.load_raw_samples(a.raw_dir, meta['scenarios'])
    rows, scatter = [], {'ref': [], 'genco': [], 'h1': []}
    for s in meta['scenarios']:
        z = np.load(d / f'scenario{s}_float32.npz')
        pred, ref = z['original'], z['reference']
        states = {}
        for si, delta in enumerate(scales):
            g = gm.grid_from_heterodata(gm.scale_heterodata(samples[s], delta), 100.0)
            v0 = (pred[si, :, 0].copy(), pred[si, :, 1].copy())
            flat = acgrid.newton_pf(g, tol=TOL)
            warm = acgrid.newton_pf(g, tol=TOL, flat_start=False, v0=v0)
            st = {'genco': (pred[si, :, 0], pred[si, :, 1], pred[si, :, 2], pred[si, :, 3])}
            for k in (1, 2):
                h = acgrid.newton_pf(g, tol=0, max_iter=k, flat_start=False, v0=v0, strict=False)
                st[f'genco+{k}nr'] = (h['vm'], h['va'], h['pg'], h['qg'])
            states[si] = (g, st)
            spec_pg = acgrid.bus_generation_setpoint(g)
            nonref_, pq_ = g.bus_type != acgrid.REF, g.bus_type == acgrid.PQ
            for name, (vm, va, pg, qg) in st.items():
                # specified injections (PV set-points, zero generation at PQ), not values derived from V
                pg_eval = np.where(nonref_, spec_pg, pg)
                qg_eval = np.where(pq_, 0.0, qg)
                dp, dq, inf = acgrid.power_balance_residual(g, vm, va, pg_eval, qg_eval)
                rows.append({'scenario': s, 'relative_load_change': delta, 'method': name,
                             'vm_err_max_pu': float(np.abs(vm - ref[si, :, 0]).max()),
                             'va_err_max_rad': float(np.abs(va - ref[si, :, 1]).max()),
                             'p_residual_mean_mw': float(np.abs(dp[inf['P']]).mean()),
                             'p_residual_max_mw': float(np.abs(dp[inf['P']]).max()),
                             'q_residual_mean_mvar': float(np.abs(dq[inf['Q']]).mean()),
                             'nr_iters_flat': flat['iterations'], 'nr_iters_warm_genco': warm['iterations']})
        g0 = states[base][0]
        pq, nonref = g0.bus_type == acgrid.PQ, g0.bus_type != acgrid.REF
        for si, delta in enumerate(scales):
            if si == base:
                continue
            dr = ref[si] - ref[base]
            gs, gb = states[si][0], states[base][0]
            df_ref = acgrid.branch_flows_mw(gs, ref[si, :, 0], ref[si, :, 1]) - acgrid.branch_flows_mw(gb, ref[base, :, 0], ref[base, :, 1])
            moving = np.abs(df_ref) > 0.5  # edges whose true flow moves by more than 0.5 MW
            for name in ('genco', 'genco+1nr', 'genco+2nr'):
                cur, b0 = states[si][1][name], states[base][1][name]
                dvm, dva = cur[0] - b0[0], cur[1] - b0[1]
                df = acgrid.branch_flows_mw(gs, cur[0], cur[1]) - acgrid.branch_flows_mw(gb, b0[0], b0[1])
                ratio = df[moving] / df_ref[moving]
                for r in rows:
                    if r['scenario'] == s and r['relative_load_change'] == delta and r['method'] == name:
                        r['gain_vm'] = gain(dvm[pq], dr[pq, 0])
                        r['gain_va'] = gain(dva[nonref], dr[nonref, 1])
                        r['gain_flow'] = gain(df, df_ref)
                        r['flow_ratio_median'] = float(np.median(ratio)) if ratio.size else float('nan')
                        r['flow_ratio_p10'] = float(np.percentile(ratio, 10)) if ratio.size else float('nan')
                        r['flow_ratio_p90'] = float(np.percentile(ratio, 90)) if ratio.size else float('nan')
                        r['flow_moving_edges'] = int(moving.sum())
            if abs(delta) == 0.01:
                scatter['ref'].append(dr[nonref, 1])
                scatter['genco'].append((states[si][1]['genco'][1] - states[base][1]['genco'][1])[nonref])
                scatter['h1'].append((states[si][1]['genco+1nr'][1] - states[base][1]['genco+1nr'][1])[nonref])
        print(f'scenario {s}: done', flush=True)

    keys = sorted({k for r in rows for k in r})
    with (d / 'hybrid.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for name in ('genco', 'genco+1nr', 'genco+2nr'):
        r = [x for x in rows if x['method'] == name]
        nz = [x for x in r if x['relative_load_change'] != 0]
        summary[name] = {
            'gain_vm_mean': float(np.mean([x['gain_vm'] for x in nz])),
            'gain_va_mean': float(np.mean([x['gain_va'] for x in nz])),
            'gain_va_min': float(np.min([x['gain_va'] for x in nz])),
            'gain_va_max': float(np.max([x['gain_va'] for x in nz])),
            'gain_flow_mean': float(np.mean([x['gain_flow'] for x in nz])),
            'gain_flow_min': float(np.min([x['gain_flow'] for x in nz])),
            'gain_flow_max': float(np.max([x['gain_flow'] for x in nz])),
            'flow_ratio_median_at_1pct': float(np.median([x['flow_ratio_median'] for x in nz if abs(x['relative_load_change']) == 0.01])),
            'flow_ratio_p10_at_1pct': float(np.median([x['flow_ratio_p10'] for x in nz if abs(x['relative_load_change']) == 0.01])),
            'flow_ratio_p90_at_1pct': float(np.median([x['flow_ratio_p90'] for x in nz if abs(x['relative_load_change']) == 0.01])),
            'vm_err_max_pu': float(np.max([x['vm_err_max_pu'] for x in r])),
            'va_err_max_rad': float(np.max([x['va_err_max_rad'] for x in r])),
            'p_residual_mean_mw': float(np.mean([x['p_residual_mean_mw'] for x in r])),
            'p_residual_max_mw': float(np.max([x['p_residual_max_mw'] for x in r])),
        }
    g = [x for x in rows if x['method'] == 'genco']
    summary['nr_iterations_to_1e-8'] = {
        'flat_start_mean': float(np.mean([x['nr_iters_flat'] for x in g])),
        'genco_warm_start_mean': float(np.mean([x['nr_iters_warm_genco'] for x in g])),
        'flat_start_max': int(np.max([x['nr_iters_flat'] for x in g])),
        'genco_warm_start_max': int(np.max([x['nr_iters_warm_genco'] for x in g]))}
    (d / 'hybrid_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['| Method | Line-flow gain [min–max] | median per-line ΔP ratio at ±1 % [p10–p90] |', '|---|---|---|']
    for name, label in (('genco', 'GENCO'), ('genco+1nr', 'GENCO + 1 NR step'), ('genco+2nr', 'GENCO + 2 NR steps')):
        q = summary[name]
        lines.append(f"| {label} | {q['gain_flow_mean']:.4f} [{q['gain_flow_min']:.4f}–{q['gain_flow_max']:.4f}] | "
                     f"{q['flow_ratio_median_at_1pct']:.3f} [{q['flow_ratio_p10_at_1pct']:.3f}–{q['flow_ratio_p90_at_1pct']:.3f}] |")
    lines += ['', '| Method | Vm gain | Va gain [min–max] | max |Vm err| [p.u.] | max |Va err| [rad] | mean |ΔP| [MW] | max |ΔP| [MW] |',
             '|---|---|---|---|---|---|---|']
    for name, label in (('genco', 'GENCO'), ('genco+1nr', 'GENCO + 1 NR step'), ('genco+2nr', 'GENCO + 2 NR steps')):
        q = summary[name]
        lines.append(f"| {label} | {q['gain_vm_mean']:.4f} | {q['gain_va_mean']:.4f} [{q['gain_va_min']:.4f}–{q['gain_va_max']:.4f}] | "
                     f"{q['vm_err_max_pu']:.1e} | {q['va_err_max_rad']:.1e} | {q['p_residual_mean_mw']:.1e} | {q['p_residual_max_mw']:.1e} |")
    it = summary['nr_iterations_to_1e-8']
    lines += ['', f"Newton-Raphson iterations to 1e-8 p.u.: flat start {it['flat_start_mean']:.2f} (max {it['flat_start_max']}), "
                  f"GENCO warm start {it['genco_warm_start_mean']:.2f} (max {it['genco_warm_start_max']})"]
    (d / 'hybrid_summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    import figstyle as fs
    import matplotlib.pyplot as plt
    ref = np.concatenate(scatter['ref']) * 1e3
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={'width_ratios': [1.25, 1]})
    a, b = ax
    lim = np.ceil(np.abs(ref).max() / 10) * 10
    a.plot([-lim, lim], [-lim, lim], color=fs.REFERENCE, lw=1.2, zorder=2, label='physics: predicted = true')
    gv, hv = summary['genco']['gain_va_mean'], summary['genco+1nr']['gain_va_mean']
    a.scatter(ref, np.concatenate(scatter['genco']) * 1e3, s=5, alpha=.35, color=fs.ORANGE, linewidth=0,
              zorder=3, label=f'GENCO (gain {gv:.2f})')
    a.scatter(ref, np.concatenate(scatter['h1']) * 1e3, s=5, alpha=.35, color=fs.BLUE, linewidth=0,
              zorder=4, label=f'GENCO + 1 Newton step (gain {hv:.3f})')
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_aspect('equal')
    a.axhline(0, color=fs.AXIS, lw=0.8, zorder=1)
    a.axvline(0, color=fs.AXIS, lw=0.8, zorder=1)
    a.set_xlabel('true change in voltage angle for ±1 % load [mrad]')
    a.set_ylabel('predicted change [mrad]')
    a.set_title(f'Butterfly: every non-slack bus, {len(meta["scenarios"])} test scenarios')
    leg = a.legend(loc='upper left', markerscale=3)
    for hnd in leg.legend_handles:
        hnd.set_alpha(1)

    names = ['GENCO', 'GENCO + 1 Newton step', 'GENCO + 2 Newton steps']
    keys = ('genco', 'genco+1nr', 'genco+2nr')
    vals = [summary[k]['p_residual_mean_mw'] for k in keys]
    y = np.arange(3)[::-1]
    b.barh(y, vals, height=0.28, color=[fs.ORANGE, fs.BLUE, '#86b6ef'], zorder=3)
    for yi, v, k in zip(y, vals, keys):
        b.annotate(f'{v:.1e} MW', (v, yi), xytext=(6, 0), textcoords='offset points', va='center',
                   fontsize=9, color=fs.INK2)
    b.set_xscale('log')
    b.set_xlim(1e-8, 10)
    b.set_yticks(y)
    b.set_yticklabels(names)
    b.grid(axis='y', visible=False)
    b.set_xlabel('mean |ΔP| at PQ + PV buses [MW], log scale')
    b.set_title('Physics Lens: power-balance residual')
    fs.header(fig, 'GENCO under-reacts to load changes; one Newton step from its prediction restores the physics',
              'Left: predicted vs true voltage-angle change at ±1 % load. Right: mean power-balance residual '
              'over all 350 cases (50 scenarios × 7 load levels).')
    fig.savefig(d / 'hybrid.png')


if __name__ == '__main__':
    main()
