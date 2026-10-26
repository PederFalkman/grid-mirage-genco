"""Cross-checkpoint comparison of real GENCO GRID MIRAGE runs (same protocol, official test splits).

Reads results/<run>/{summary.json, hybrid_summary.json, metadata.json} for every run directory
given (default: all results/genco_* with a summary) and writes results/checkpoints.{md,json,png}.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt

SIZES = {'tiny': 12, 'small': 24, 'base': 48}


def describe(d):
    meta = json.loads((d / 'metadata.json').read_text())
    case, _, size, seed = Path(meta['config']).stem.split('_')
    s = json.loads((d / 'summary.json').read_text())
    h = json.loads((d / 'hybrid_summary.json').read_text()) if (d / 'hybrid_summary.json').exists() else {}
    pub = None
    mfile = Path(meta['config']).parent / 'metrics.csv'
    if mfile.exists():
        for line in mfile.read_text().splitlines():
            if line.startswith('Avg. active res'):
                pub = float(line.split(',')[1])
    b = s['butterfly']
    return {
        'run': d.name, 'grid': case.replace('case', 'IEEE '), 'buses': int(case[4:]), 'size': size,
        'hidden': SIZES.get(size), 'seed': seed, 'scenarios': len(meta['scenarios']),
        'reference_validated_all': s['reference_validated_all'],
        'mirror_float64_max_va_rad': s['mirror']['float64']['va_rad']['max'],
        'mirror_float32_max_va_rad': s['mirror']['float32']['va_rad']['max'],
        'gain_vm': b['per_scenario_vm'], 'gain_va': b['per_scenario_va'], 'gain_slack': b['per_scenario_slack'],
        'gain_flow_mean': h.get('genco', {}).get('gain_flow_mean'),
        'gain_flow_1nr': h.get('genco+1nr', {}).get('gain_flow_mean'),
        'p_residual_mw_delta0': s['physics_lens']['0.0']['p_residual_mw_mean'],
        'published_p_residual_mw': pub,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--out', default='results/checkpoints')
    a = ap.parse_args()
    dirs = [Path(r) for r in a.runs] or sorted(p.parent for p in Path('results').glob('genco_*/summary.json'))
    rows = [describe(d) for d in dirs if (d / 'metadata.json').exists()]
    rows = [r for r in rows if r['scenarios'] >= 50]
    rows.sort(key=lambda r: (r['buses'], r['hidden'], r['seed']))
    out = Path(a.out)
    out.with_suffix('.json').write_text(json.dumps(rows, indent=2) + '\n')

    lines = ['| Checkpoint | hidden | n | Mirror float64 max [rad] | Vm gain [95 % CI] | Va gain [95 % CI] | line-flow gain | '
             'slack ΔP gain | P residual δ=0 [MW] (published) |', '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        r['label'] = f"{r['grid']} {r['size']} {r['seed'].replace('seed', 'seed ')}"
        ci = lambda g: f"{g['mean']:.3f} [{g['ci95_mean'][0]:.3f}, {g['ci95_mean'][1]:.3f}]"  # noqa: E731
        fl = '—' if r['gain_flow_mean'] is None else f"{r['gain_flow_mean']:.3f}"
        pub = '—' if r['published_p_residual_mw'] is None else f"{r['published_p_residual_mw']:.3g}"
        lines.append(f"| {r['label']} | {r['hidden']} | {r['scenarios']} | {r['mirror_float64_max_va_rad']:.0e} | "
                     f"{ci(r['gain_vm'])} | {ci(r['gain_va'])} | {fl} | {r['gain_slack']['mean']:.3f} | "
                     f"{r['p_residual_mw_delta0']:.3g} ({pub}) |")
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.9 + 0.42 * len(rows) + 0.8), sharey=True)
    grids = sorted({r['buses'] for r in rows})
    for ax, (key, title) in zip(axes, [('gain_vm', 'Voltage magnitude response'), ('gain_va', 'Voltage angle response')]):
        ax.axvline(1, color=fs.REFERENCE, lw=1.2, zorder=2)
        ticks, labels = [], []
        yi = 0
        for gi, nb in enumerate(grids):
            for r in [x for x in rows if x['buses'] == nb][::-1]:
                g = r[key]
                color = fs.BLUE if nb == grids[0] else fs.ORANGE
                ax.plot(g['ci95_mean'], [yi, yi], color=color, lw=2.4, solid_capstyle='round', zorder=3)
                ax.plot([g['p5'], g['p95']], [yi, yi], color=color, lw=0.9, alpha=0.6, zorder=2)
                ax.scatter([g['mean']], [yi], s=60, color=color, edgecolor=fs.SURFACE, linewidth=1.5, zorder=4)
                ax.annotate(f"{g['mean']:.2f}", (g['mean'], yi), xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=8.5, color=fs.INK2)
                ticks.append(yi)
                labels.append(f"{r['grid']} {r['size']} (h={r['hidden']}), {r['seed'].replace('seed', 'seed ')}")
                yi += 1
            yi += 0.6
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.grid(axis='y', visible=False)
        ax.set_xlim(0.45, 1.12)
        ax.set_xlabel('response gain (1 = physics)')
        ax.set_title(title)
    axes[0].annotate('physics', (1, ticks[-1] + 0.5), xytext=(4, 0), textcoords='offset points',
                     fontsize=8.5, color=fs.INK2, va='center')
    fs.header(fig, 'GENCO under-reacts on IEEE 118 but not on IEEE 14, and larger IEEE 118 models under-react more',
              f'{len(rows)} released checkpoints × {rows[0]["scenarios"]} official test scenarios, ±0.1 … ±1 % demand change. '
              'Dot = mean, thick = 95 % CI, thin = 5th–95th percentile.')
    fig.savefig(out.with_suffix('.png'))
    print('Wrote', out.with_suffix('.png'))


if __name__ == '__main__':
    main()
