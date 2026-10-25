"""Aggregate real GENCO GRID MIRAGE CSVs into summary.json and summary.md (no other inputs)."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

CH = [('vm_pu', 'Vm [p.u.]'), ('va_rad', 'Va [rad]'), ('pg_mw', 'Pg [MW]'), ('qg_mvar', 'Qg [Mvar]')]


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def col(rows, key, **match):
    return np.array([float(r[key]) for r in rows if all(r[k] == str(v) for k, v in match.items())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='?', default='results/genco_n50')
    d = Path(ap.parse_args().results)
    mirror, butterfly = read(d / 'mirror.csv'), read(d / 'butterfly.csv')
    physics, accuracy = read(d / 'physics_lens.csv'), read(d / 'accuracy.csv')
    meta = json.loads((d / 'metadata.json').read_text())
    s = {'scenarios': meta['scenarios'], 'reference_validated_all': all(r['reference_validated'] == 'True' for r in accuracy)}
    s['mirror'] = {dt: {ch: {'max': float(col(mirror, f'{ch}_equivariance_max', dtype=dt).max()),
                             'mean': float(col(mirror, f'{ch}_equivariance_mean', dtype=dt).mean())}
                        for ch, _ in CH} for dt in ('float32', 'float64')}
    s['accuracy_delta0'] = {ch: {'genco_mean_abs': float(col(accuracy, f'genco_vs_dataset_{ch}_mean').mean()),
                                 'genco_max_abs': float(col(accuracy, f'genco_vs_dataset_{ch}_max').max()),
                                 'reference_max_abs': float(col(accuracy, f'reference_vs_dataset_{ch}_max').max())}
                            for ch, _ in CH}
    nz = [r for r in butterfly if float(r['relative_load_change']) != 0]
    s['butterfly'] = {
        'response_gain_vm': {'mean': float(col(nz, 'response_gain_vm_pu').mean()), 'min': float(col(nz, 'response_gain_vm_pu').min()),
                             'max': float(col(nz, 'response_gain_vm_pu').max())},
        'response_gain_va': {'mean': float(col(nz, 'response_gain_va_rad').mean()), 'min': float(col(nz, 'response_gain_va_rad').min()),
                             'max': float(col(nz, 'response_gain_va_rad').max())},
        'slack_dpg_ratio_genco_over_reference': {
            str(x): float(np.mean(col(butterfly, 'genco_slack_dpg_mw', relative_load_change=x)
                                  / col(butterfly, 'reference_slack_dpg_mw', relative_load_change=x)))
            for x in (-0.01, -0.005, -0.001, 0.001, 0.005, 0.01)},
    }
    s['physics_lens'] = {
        str(x): {'p_residual_mw_mean': float(col(physics, 'genco_p_residual_mw_pq_pv_mean', relative_load_change=x).mean()),
                 'p_residual_mw_max': float(col(physics, 'genco_p_residual_mw_pq_pv_max', relative_load_change=x).max()),
                 'q_residual_mvar_mean': float(col(physics, 'genco_q_residual_mvar_pq_mean', relative_load_change=x).mean()),
                 'q_residual_mvar_max': float(col(physics, 'genco_q_residual_mvar_pq_max', relative_load_change=x).max()),
                 'reference_p_residual_mw_max': float(col(physics, 'reference_p_residual_mw_max', relative_load_change=x).max())}
        for x in (-0.01, -0.005, -0.001, 0.0, 0.001, 0.005, 0.01)}
    rng = np.random.default_rng(0)
    for key, ch in (('vm', 'response_gain_vm_pu'), ('va', 'response_gain_va_rad'), ('slack', None)):
        per = []
        for sc in s['scenarios']:
            rows = [r for r in nz if int(r['scenario']) == sc]
            if ch:
                per.append(np.mean([float(r[ch]) for r in rows]))
            else:
                per.append(np.mean([float(r['genco_slack_dpg_mw']) / float(r['reference_slack_dpg_mw']) for r in rows]))
        per = np.array(per)
        boot = rng.choice(per, size=(10000, len(per))).mean(axis=1)
        s['butterfly'][f'per_scenario_{key}'] = {
            'n': len(per), 'mean': float(per.mean()), 'std': float(per.std(ddof=1)) if len(per) > 1 else 0.0,
            'p5': float(np.percentile(per, 5)), 'p95': float(np.percentile(per, 95)),
            'min': float(per.min()), 'max': float(per.max()),
            'ci95_mean': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'fraction_below_1': float(np.mean(per < 1))}
    (d / 'summary.json').write_text(json.dumps(s, indent=2) + '\n')

    m = s['mirror']
    lines = ['| Channel | float32 max | float32 mean | float64 max | GENCO vs dataset mean abs (δ=0) |', '|---|---|---|---|---|']
    for ch, label in CH:
        lines.append(f"| {label} | {m['float32'][ch]['max']:.2e} | {m['float32'][ch]['mean']:.2e} | "
                     f"{m['float64'][ch]['max']:.2e} | {s['accuracy_delta0'][ch]['genco_mean_abs']:.2e} |")
    lines += ['', '| Load change | GENCO/ref slack ΔPg | P residual mean [MW] | P residual max [MW] | Q residual mean [Mvar] |',
              '|---|---|---|---|---|']
    for x in (-0.01, -0.005, -0.001, 0.0, 0.001, 0.005, 0.01):
        ratio = s['butterfly']['slack_dpg_ratio_genco_over_reference'].get(str(x))
        p = s['physics_lens'][str(x)]
        lines.append(f"| {x * 100:+.1f} % | {'—' if ratio is None else f'{ratio:.3f}'} | {p['p_residual_mw_mean']:.3f} | "
                     f"{p['p_residual_mw_max']:.3f} | {p['q_residual_mvar_mean']:.4f} |")
    b = s['butterfly']
    lines += ['', f"Response gain (projection of GENCO Δ onto reference Δ): Vm {b['response_gain_vm']['mean']:.3f} "
                  f"[{b['response_gain_vm']['min']:.3f}, {b['response_gain_vm']['max']:.3f}], "
                  f"Va {b['response_gain_va']['mean']:.3f} [{b['response_gain_va']['min']:.3f}, {b['response_gain_va']['max']:.3f}]"]
    lines += ['', '| Per-scenario gain | n | mean | 95% CI of mean | std | p5–p95 | min–max | share < 1 |', '|---|---|---|---|---|---|---|---|']
    for key, label in (('vm', 'Vm response'), ('va', 'Va response'), ('slack', 'slack ΔPg ratio')):
        q = b[f'per_scenario_{key}']
        lines.append(f"| {label} | {q['n']} | {q['mean']:.3f} | [{q['ci95_mean'][0]:.3f}, {q['ci95_mean'][1]:.3f}] | {q['std']:.3f} | "
                     f"{q['p5']:.3f}–{q['p95']:.3f} | {q['min']:.3f}–{q['max']:.3f} | {q['fraction_below_1']:.0%} |")
    (d / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
