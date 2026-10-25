"""Workflow schematic for GRID MIRAGE (figure 1 of the write-up). Pure drawing, no data."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, RED, GREY, GREEN = '#2166ac', '#d6604d', '#4d4d4d', '#1b7837'


def box(ax, x, y, w, h, text, color, fs=8.8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                                fc=color + '18', ec=color, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=GREY,
            weight='bold' if bold else 'normal', wrap=True)


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3))


def main(out):
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(0, 1.012)
    ax.set_ylim(0, 1.1)
    ax.axis('off')
    box(ax, 0.01, 0.40, 0.15, 0.22, 'Official test scenario\nIEEE 118, held-out split\n(GraphKit pipeline)', GREY, bold=True)
    rows = [
        (0.72, 'MIRROR', 'Relabel buses, generators\nand branches (same physics)', 'GENCO → map back\nto original order',
         'Same answer?\n✔ ≤1e-14 (float64)\nfloat32 spread = round-off', BLUE, GREEN),
        (0.40, 'BUTTERFLY', 'Scale all demand\n±0.1 … ±1 %', 'GENCO vs exact\nAC Newton–Raphson',
         'Same response?\n✘ voltages 0.72–0.79×\nline flows 0.76×', RED, RED),
        (0.08, 'PHYSICS LENS', 'Use predicted\nVm, Va, Pg, Qg', 'AC nodal balance on\nnon-enforced quantities',
         'Physically consistent?\n0.125 MW mean residual,\ngrows with |Δload|', BLUE, GREY),
    ]
    for y, name, step1, step2, verdict, c, vc in rows:
        ax.text(0.205, y + 0.205, name, fontsize=10, weight='bold', color=c)
        box(ax, 0.20, y, 0.17, 0.19, step1, c)
        box(ax, 0.41, y, 0.17, 0.19, step2, c)
        box(ax, 0.62, y, 0.17, 0.19, verdict, vc)
        arrow(ax, 0.163, 0.51, 0.197, y + 0.095)
        arrow(ax, 0.37, y + 0.095, 0.41, y + 0.095)
        arrow(ax, 0.58, y + 0.095, 0.62, y + 0.095)
    box(ax, 0.825, 0.30, 0.17, 0.40, 'HYBRID CHECK\n\nGENCO + 1 Newton step:\nresponse 0.9996–0.9999×\nresidual 0.0006 MW\n\nthe bias is smooth,\nsystematic and\ncorrectable', GREEN, fs=8.5)
    arrow(ax, 0.79, 0.495, 0.825, 0.495, GREEN)
    arrow(ax, 0.79, 0.175, 0.825, 0.32, GREY)
    ax.text(0.5, 1.09, 'GRID MIRAGE: a label-free acceptance test for grid foundation models',
            ha='center', va='top', fontsize=12.5, weight='bold', color=GREY)
    fig.savefig(out, dpi=170, bbox_inches='tight')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/genco_n50/schematic.png')
