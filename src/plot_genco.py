"""Figures from real GENCO GRID MIRAGE results written by genco_mirage.py / summarize_genco.py.

mirror.png            relabeling deviation (float64, float32) vs GENCO's own error, per output
butterfly_physics.png response to load changes, per-scenario gains, Physics Lens residual
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import figstyle as fs
import matplotlib.pyplot as plt

CH = [('vm_pu', 'Voltage magnitude', 'p.u.'), ('va_rad', 'Voltage angle', 'rad'),
      ('pg_mw', 'Active generation', 'MW'), ('qg_mvar', 'Reactive generation', 'Mvar')]


def read(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def by_scale(rows, key, **match):
    acc = defaultdict(list)
    for r in rows:
        if all(r[k] == str(v) for k, v in match.items()):
            acc[float(r['relative_load_change'])].append(float(r[key]))
    xs = sorted(acc)
    return np.array(xs) * 100, np.array([np.mean(acc[x]) for x in xs])


def header(fig, title, subtitle, top=None):
    """Left-aligned title + subtitle at fixed distances (inches) from the top edge."""
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.78 / h))
    fig.text(0.012, 1 - 0.18 / h, title, ha='left', va='top', fontsize=12.5, weight='bold', color=fs.INK)
    fig.text(0.012, 1 - 0.48 / h, subtitle, ha='left', va='top', fontsize=9.5, color=fs.INK2)


def mirror_figure(d, summary):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, (ch, name, unit) in zip(axes, CH):
        vals = [summary['mirror']['float64'][ch]['max'], summary['mirror']['float32'][ch]['max'],
                summary['accuracy_delta0'][ch]['genco_mean_abs']]
        labels = ['Relabeled, float64 (max)', 'Relabeled, float32 (max)', 'GENCO error vs truth (mean)']
        colors = [fs.BLUE, fs.AQUA, fs.ORANGE]
        y = np.arange(3)[::-1]
        ax.hlines(y, 1e-16, vals, color=fs.GRID, lw=2, zorder=1)
        ax.scatter(vals, y, s=70, color=colors, edgecolor=fs.SURFACE, linewidth=1.5, zorder=3)
        for yi, v in zip(y, vals):
            ax.annotate(f'{v:.0e}'.replace('e-0', 'e-'), (v, yi), xytext=(7, 0), textcoords='offset points',
                        va='center', fontsize=8.5, color=fs.INK2)
        ax.set_xscale('log')
        ax.set_xlim(1e-16, 1e3)
        ax.set_ylim(-0.6, 2.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is axes[0] else [''] * 3)
        ax.grid(axis='y', visible=False)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(f'|deviation| [{unit}], log scale', fontsize=8.5)
        ax.set_xticks([1e-15, 1e-10, 1e-5, 1])
    header(fig, 'Mirror: relabeling the grid does not change GENCO\'s answer',
           '1,750 relabeled cases (50 test scenarios × 7 load levels × 5 relabelings). Deviation after mapping back '
           'to the original numbering, compared with GENCO\'s own error against the AC solution.', 0.80)
    fig.savefig(d / 'mirror.png')
    plt.close(fig)


def butterfly_figure(d, summary, butterfly, physics, hybrid):
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.4))
    a, b, c, e = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    for axis, gkey, rkey, scale, ylabel, title in (
            (a, 'genco_mean_dvm_pu', 'reference_mean_dvm_pu', 1e3, 'mean ΔVm at PQ buses [10⁻³ p.u.]',
             'Voltage response to a load change'),
            (b, 'genco_slack_dpg_mw', 'reference_slack_dpg_mw', 1, 'slack-generator ΔP [MW]',
             'Slack generation response')):
        x, yr = by_scale(butterfly, rkey)
        _, yg = by_scale(butterfly, gkey)
        axis.axhline(0, color=fs.AXIS, lw=0.8, zorder=1)
        axis.plot(x, yr * scale, color=fs.REFERENCE, marker='o', label='AC physics (exact)', zorder=3)
        axis.plot(x, yg * scale, color=fs.ORANGE, marker='o', label='GENCO', zorder=4)
        fs.end_label(axis, x[-1], yr[-1] * scale, 'physics', color=fs.INK2, dx=6)
        fs.end_label(axis, x[-1], yg[-1] * scale, 'GENCO', color=fs.INK2, dx=6)
        axis.set_xlim(-1.15, 1.45)
        axis.set_xlabel('load change [%]')
        axis.set_ylabel(ylabel)
        axis.set_title(title)

    # per-scenario response gains
    def per_scenario(rows, key, method=None):
        acc = defaultdict(list)
        for r in rows:
            if float(r['relative_load_change']) == 0 or (method and r['method'] != method):
                continue
            acc[r['scenario']].append(float(r[key]))
        return np.array([np.mean(v) for v in acc.values()])
    slack = defaultdict(list)
    for r in butterfly:
        if float(r['relative_load_change']) != 0:
            slack[r['scenario']].append(float(r['genco_slack_dpg_mw']) / float(r['reference_slack_dpg_mw']))
    groups = [('Voltage magnitude', per_scenario(butterfly, 'response_gain_vm_pu')),
              ('Voltage angle', per_scenario(butterfly, 'response_gain_va_rad')),
              ('Line flows', per_scenario(hybrid, 'gain_flow', 'genco')),
              ('Slack generation', np.array([np.mean(v) for v in slack.values()]))]
    rng = np.random.default_rng(0)
    c.axvline(1, color=fs.REFERENCE, lw=1.2, zorder=2)
    c.text(1.005, 3.62, 'physics = 1', color=fs.INK2, fontsize=8.5, va='center')
    for i, (name, g) in enumerate(groups):
        yi = len(groups) - 1 - i
        c.scatter(g, yi + rng.uniform(-0.18, 0.18, g.size), s=22, color=fs.ORANGE, alpha=0.75,
                  edgecolor=fs.SURFACE, linewidth=0.6, zorder=3)
        c.plot([g.mean()] * 2, [yi - 0.3, yi + 0.3], color=fs.INK, lw=2, zorder=4)
        c.text(1.005, yi, f'mean {g.mean():.2f}', va='center', fontsize=8.5, color=fs.INK2)
    c.set_yticks(range(len(groups)))
    c.set_yticklabels([g[0] for g in groups][::-1])
    c.set_xlim(0.45, 1.13)
    c.set_ylim(-0.6, 3.9)
    c.grid(axis='y', visible=False)
    c.set_xlabel('response gain = GENCO change / true change  (one dot per scenario, n = 50)')
    c.set_title('GENCO under-reacts in every scenario')

    x, p = by_scale(physics, 'genco_p_residual_mw_pq_pv_mean')
    e.plot(x, p, color=fs.ORANGE, marker='o', zorder=3)
    e.axhline(0.106, color=fs.MUTED, lw=1, zorder=2)
    e.text(-1.1, 0.1075, 'published checkpoint metric 0.106 MW', color=fs.INK2, fontsize=8.5, va='bottom')
    for xi, pi in zip(x, p):
        if abs(xi) in (0, 1):
            e.annotate(f'{pi:.3f}', (xi, pi), xytext=(0, 8), textcoords='offset points', ha='center',
                       fontsize=8.5, color=fs.INK2)
    e.set_ylim(0, max(p) * 1.25)
    e.set_xlim(-1.15, 1.15)
    e.set_xlabel('load change [%]')
    e.set_ylabel('mean |ΔP| at PQ + PV buses [MW]')
    e.set_title('Physics Lens: power-balance residual of GENCO')
    header(fig, 'Butterfly: GENCO moves the grid only ~3/4 as far as the physics does',
           '50 official test scenarios, uniform demand change of ±0.1 … ±1 %; reference = exact AC Newton–Raphson '
           'solution of the same input.', 0.90)
    fig.savefig(d / 'butterfly_physics.png')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='?', default='results/genco_n50')
    d = Path(ap.parse_args().results)
    summary = json.loads((d / 'summary.json').read_text())
    mirror_figure(d, summary)
    hybrid = read(d / 'hybrid.csv') if (d / 'hybrid.csv').exists() else []
    if hybrid:
        butterfly_figure(d, summary, read(d / 'butterfly.csv'), read(d / 'physics_lens.csv'), hybrid)
    print('Wrote', d / 'mirror.png', d / 'butterfly_physics.png' if hybrid else '(butterfly needs hybrid.csv)')


if __name__ == '__main__':
    main()
