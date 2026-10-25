"""Shared figure style for GRID MIRAGE (validated categorical palette, recessive chrome)."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
# categorical slots in fixed order (validated: CVD/normal-vision separation on light surface)
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
GOOD, CRITICAL = '#0ca30c', '#d03b3b'
REFERENCE = INK2  # the physics reference is drawn in ink, GENCO in a series colour

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'DejaVu Sans', 'font.size': 9.5,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-',
    'axes.axisbelow': True,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'xtick.minor.size': 0, 'ytick.minor.size': 0,
    'lines.linewidth': 2, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    'lines.markersize': 7, 'lines.markeredgewidth': 1.5, 'lines.markeredgecolor': SURFACE,
    'legend.frameon': False, 'legend.fontsize': 8.5, 'legend.labelcolor': INK2,
    'figure.dpi': 110, 'savefig.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
})


def suptitle(fig, title, subtitle=None):
    fig.text(0.01, 1.0, title, ha='left', va='bottom', fontsize=12.5, weight='bold', color=INK,
             transform=fig.transFigure)
    if subtitle:
        fig.text(0.01, 0.985, subtitle, ha='left', va='top', fontsize=9.5, color=INK2, transform=fig.transFigure)


def end_label(ax, x, y, text, color=INK2, dx=4, dy=0, **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points', va='center', ha='left',
                fontsize=8.5, color=color, **kw)


def header(fig, title, subtitle):
    """Left-aligned title + subtitle at fixed distances (inches) from the top edge."""
    h = fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.78 / h))
    fig.text(0.012, 1 - 0.18 / h, title, ha='left', va='top', fontsize=12.5, weight='bold', color=INK)
    fig.text(0.012, 1 - 0.48 / h, subtitle, ha='left', va='top', fontsize=9.5, color=INK2)
