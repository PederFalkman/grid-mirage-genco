"""Workflow schematic for GRID MIRAGE (figure 1 of the write-up). Pure drawing, no data."""
import sys
from pathlib import Path
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, RED, GREY, GREEN, AQUA, ORANGE = fs.BLUE, fs.CRITICAL, '#898781', fs.GOOD, fs.AQUA, fs.ORANGE


INK = fs.INK


def box(ax, x, y, w, h, text, color, fs=8.8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                                fc=color + '14', ec=color, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
            weight='bold' if bold else 'normal', wrap=True)


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=color, lw=1.1))


def main(out):
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(0, 1.1)
    ax.axis('off')
    box(ax, 0.01, 0.40, 0.15, 0.22, 'Official test scenario\nIEEE 118, held-out split\nGraphKit pipeline', GREY, fs=8.4, bold=True)
    rows = [
        (0.72, 'MIRROR', 'Relabel buses, generators\nand branches (same physics)', 'GENCO → map back\nto original order',
         '✔  Same answer\n≤1e-14 in float64;\nfloat32 spread = round-off', BLUE, GREEN),
        (0.40, 'BUTTERFLY', 'Scale all demand\n±0.1 … ±1 %', 'GENCO vs exact\nAC Newton–Raphson',
         '✘  Under-reacts\nvoltages 0.72–0.79×,\nline flows 0.76× of physics', ORANGE, RED),
        (0.08, 'PHYSICS LENS', 'Use predicted\nVm, Va, Pg, Qg', 'AC nodal balance on\nnon-enforced quantities',
         'Residual 0.125 MW mean,\ngrows with |Δload|', AQUA, GREY),
    ]
    for y, name, step1, step2, verdict, c, vc in rows:
        ax.text(0.205, y + 0.205, name, fontsize=10, weight='bold', color=INK)
        box(ax, 0.20, y, 0.17, 0.19, step1, c)
        box(ax, 0.41, y, 0.17, 0.19, step2, c)
        box(ax, 0.62, y, 0.17, 0.19, verdict, vc)
        arrow(ax, 0.174, 0.51, 0.186, y + 0.095)
        arrow(ax, 0.384, y + 0.095, 0.397, y + 0.095)
        arrow(ax, 0.594, y + 0.095, 0.607, y + 0.095)
    box(ax, 0.825, 0.30, 0.17, 0.40, 'Hybrid check\n\nGENCO + 1 Newton step\n\n✔  response 0.9996–0.9999×\n✔  residual 0.0006 MW\n\nthe bias is systematic\nand correctable', GREEN, fs=8.5)
    arrow(ax, 0.804, 0.495, 0.812, 0.495, GREEN)
    arrow(ax, 0.804, 0.19, 0.818, 0.30, GREY)
    ax.text(0.0, 1.09, 'GRID MIRAGE: a label-free acceptance test for grid foundation models',
            ha='left', va='top', fontsize=12.5, weight='bold', color=INK)
    fig.savefig(out, dpi=170, bbox_inches='tight')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/genco_n50/schematic.png')
