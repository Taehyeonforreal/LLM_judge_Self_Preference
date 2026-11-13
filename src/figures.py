"""The three figures in the README, drawn from results/results.json.

Run `python -m src.analyze` first. figures.ipynb calls these with the settings
used in the README; every keyword below is safe to change.
"""
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator

from . import config

FONT = ["Pretendard", "Helvetica Neue", "Arial", "DejaVu Sans"]
INK, MUTED, HAIR = "#243746", "#7A8792", "#D3D8DC"
BLUE, ORANGE, GREY = "#2F6F9F", "#C4622D", "#8A949C"
# One-hue ramp for D. Monotone in lightness; every cell also prints its value.
RAMP = ["#E3EDF5", "#BBD4E7", "#8CB5D2", "#5892BD", "#2F6F9F"]


def load_results():
    return json.load(open(config.RESULTS / "results.json"))


def _save(fig, name, save):
    if save:
        config.FIGURES.mkdir(exist_ok=True)
        fig.savefig(config.FIGURES / f"{name}.svg", bbox_inches="tight", transparent=True)
        fig.savefig(config.FIGURES / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")


# ---- Figure 1 ---------------------------------------------------------------
def gap_by_level(results=None, annotate_levels=(3, 4, 5), show_pass_at_1=True,
                 caption="llama-3.1-8b · MATH500, 40 problems per level · k = 16, T = 0.7",
                 save=True, figsize=(9.6, 5.4)):
    """coverage@16 vs majority@16 by level; the band between them is the gap."""
    r = (results or load_results())["motivation"]["by_level"]
    lv = sorted(int(k) for k in r)
    get = lambda key: [r[str(l)][key] for l in lv]
    p1, maj, cov = get("pass_at_1"), get("majority_at_16"), get("coverage_at_16")
    plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(lv, maj, cov, color=ORANGE, alpha=0.14, lw=0, zorder=1)
    if show_pass_at_1:
        ax.fill_between(lv, p1, maj, color="#9AA5AE", alpha=0.20, lw=0, zorder=1)
    ax.plot(lv, cov, color=BLUE, lw=2.2, marker="o", ms=9, mec="white", mew=1.6,
            zorder=4, label="coverage@16")
    ax.plot(lv, maj, color=ORANGE, lw=2.2, marker="s", ms=8, mec="white", mew=1.6,
            zorder=4, label="majority vote@16")
    if show_pass_at_1:
        ax.plot(lv, p1, color=GREY, lw=1.7, ls=(0, (5, 3)), marker="^", ms=7, mfc="white",
                mec=GREY, mew=1.5, zorder=3, label="pass@1")

    for x, c, m in zip(lv, cov, maj):          # dimension line + value for the gap
        if x not in annotate_levels:
            continue
        ax.annotate("", xy=(x, c), xytext=(x, m), zorder=5,
                    arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.5,
                                    shrinkA=4, shrinkB=4, mutation_scale=11))
        ax.annotate(f"{c - m:.3f}", (x, (c + m) / 2), ha="center", va="center",
                    fontsize=12.5, color=INK, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="none", alpha=0.94))

    ax.set_xticks(lv)
    ax.set_xticklabels([f"Level {l}" for l in lv], fontsize=12, color=INK)
    ax.set_xlim(0.62, 5.5); ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.tick_params(axis="y", labelsize=11, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.grid(axis="y", color="#E3E7EA", lw=1, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(HAIR)
    ax.set_ylabel("fraction of problems", fontsize=10.5, color=MUTED, labelpad=8)
    if caption:
        ax.set_title(caption, fontsize=12, color=MUTED, loc="center", pad=14)

    h, l = ax.get_legend_handles_labels()
    h.append(Patch(fc=ORANGE, alpha=0.14, ec="none")); l.append("generation–verification gap")
    if show_pass_at_1:
        h.append(Patch(fc="#9AA5AE", alpha=0.20, ec="none")); l.append("recovered by majority vote")
    ax.legend(h, l, loc="lower left", frameon=False, fontsize=10.5, labelcolor=INK,
              handlelength=2.0, borderaxespad=0.6, labelspacing=0.55)
    plt.tight_layout()
    _save(fig, "gap_by_level", save)
    return fig


# ---- matrix figures ---------------------------------------------------------
def _matrix(values, rows, cols, *, vmax=0.50, row_mean=None, col_mean=None,
            caption="", self_note="judging its own pool", footer=None, unit=1.45):
    """values[row][col] -> D. Diagonal (row == col) cells get a ring.

    footer: optional list of (label, value, bold) lines drawn under the grid.
    """
    cw, ch, gap = 1.18, 0.74, 0.06
    pad_col, pad_axis, pad_rot, pad_leg = 0.12, 0.44, 1.28, 0.26
    nx, ny = len(cols), len(rows)
    top_y = ny * ch
    margin = row_mean is not None
    m_x0, m_w, m_y1, m_h = nx * cw + 0.34, 1.0, -0.18, 0.40
    right = (m_x0 + m_w) if margin else nx * cw
    bottom = (m_y1 - m_h) if margin else 0.0
    if footer:
        bottom -= 0.30 * len(footer) + 0.30

    sw, sh = (1.45 if nx > 2 else 0.85), 0.15
    sy = bottom - pad_leg - sh
    top = top_y + pad_axis + 0.26 + (0.34 if caption else 0)
    xlim, ylim = (-pad_rot - 0.30, right + 0.20), (sy - 0.32, top)

    cmap = LinearSegmentedColormap.from_list("d", RAMP)
    plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=((xlim[1] - xlim[0]) * unit, (ylim[1] - ylim[0]) * unit))
    ax.set_aspect("equal")

    def cell(x, y, w, h, v, *, ring=False, flat=None, fs=13.5):
        t = max(0.0, min(1.0, v / vmax))
        fc = flat or cmap(t)
        tc = "white" if (not flat and t > 0.72) else INK
        ax.add_patch(Rectangle((x + gap, y + gap), w - 2 * gap, h - 2 * gap, fc=fc, ec="none", zorder=2))
        if ring:
            ax.add_patch(Rectangle((x + gap, y + gap), w - 2 * gap, h - 2 * gap,
                                   fc="none", ec=INK, lw=1.9, zorder=4))
        ax.text(x + w / 2, y + h / 2, f"{v:.3f}".replace("-", "−"), ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if ring else "normal", zorder=5)

    for i, g in enumerate(rows):
        for j, v in enumerate(cols):
            cell(j * cw, top_y - (i + 1) * ch, cw, ch, values[g][v], ring=(g == v))

    for j, v in enumerate(cols):
        ax.text(j * cw + cw / 2, top_y + pad_col, config.LABEL[v], ha="center", va="bottom",
                fontsize=10.5, color=INK)
    ax.text(nx * cw / 2, top_y + pad_axis, "verifier", ha="center", va="bottom",
            fontsize=11, color=MUTED, style="italic")
    for i, g in enumerate(rows):
        ax.text(-0.14, top_y - (i + 0.5) * ch, config.LABEL[g], ha="right", va="center",
                fontsize=11.5, color=INK)
    ax.text(-pad_rot, top_y - ny * ch / 2, "pool (generator)", ha="center", va="center",
            fontsize=11, color=MUTED, style="italic", rotation=90)
    if caption:
        ax.text(nx * cw / 2, top_y + pad_axis + 0.36, caption, ha="center", va="bottom",
                fontsize=10.5, color=MUTED)

    if margin:
        flat = "#F0F2F4"
        for i, g in enumerate(rows):
            cell(m_x0, top_y - (i + 1) * ch, m_w, ch, row_mean[g], flat=flat, fs=11.5)
        ax.text(m_x0 + m_w / 2, top_y + pad_col, "pool\nmean", ha="center", va="bottom",
                fontsize=10, color=MUTED, linespacing=1.35)
        for j, v in enumerate(cols):
            cell(j * cw, m_y1 - m_h, cw, m_h, col_mean[v], flat=flat, fs=11.5)
        ax.text(-0.14, m_y1 - m_h / 2, "verifier mean", ha="right", va="center",
                fontsize=10, color=MUTED)

    if footer:
        y0 = (m_y1 - m_h if margin else 0.0) - 0.42
        rx = min(right, 1.66)
        for k, (lab, val, bold) in enumerate(footer):
            y = y0 - 0.30 * k
            if bold:
                ax.plot([0, rx], [y + 0.16, y + 0.16], color=HAIR, lw=0.9)
            ax.text(0, y, lab, ha="left", va="center", fontsize=12.5 if bold else 10.5,
                    color=INK if bold else MUTED, fontweight="bold" if bold else "normal")
            ax.text(rx, y, f"{val:+.3f}".replace("-", "−"), ha="right", va="center",
                    fontsize=13.5 if bold else 11.5, color=INK,
                    fontweight="bold" if bold else "normal")

    for t in range(100):                      # colour scale
        ax.add_patch(Rectangle((sw * t / 100, sy), sw / 100 + 0.004, sh, fc=cmap(t / 99), ec="none"))
    ax.text(0, sy - 0.10, "0", ha="left", va="top", fontsize=9.5, color=MUTED)
    ax.text(sw, sy - 0.10, f"{vmax:.2f}", ha="right", va="top", fontsize=9.5, color=MUTED)
    lx = right - 0.075 * len(self_note) - 0.42
    ax.add_patch(Rectangle((lx, sy - 0.02), 0.30, sh + 0.04, fc="none", ec=INK, lw=1.7))
    ax.text(lx + 0.42, sy + sh / 2, self_note, ha="left", va="center", fontsize=10, color=INK)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    plt.tight_layout()
    return fig


# ---- Figure 2 ---------------------------------------------------------------
def matrix_4x4(results=None, show_means=False, caption="", save=True):
    """D for every pool x verifier cell, all 500 problems."""
    m = (results or load_results())["matrix"]
    vals = {g: {v: m["cells"][g][v]["d"] for v in config.MODELS} for g in config.MODELS}
    fig = _matrix(vals, config.MODELS, config.MODELS, caption=caption,
                  row_mean=m["pool_mean"] if show_means else None,
                  col_mean=m["verifier_mean"] if show_means else None)
    _save(fig, "matrix_4x4", save)
    return fig


# ---- Figure 3 ---------------------------------------------------------------
def matrix_qwen_gemma(results=None, show_delta=True, save=True):
    """The one testable pair, on the problems both pools share."""
    r = results or load_results()
    qg = r["qwen_gemma"]
    keys = ["qwen", "gemma"]
    vals = {g: {v: qg["cells"][g][v]["d"] for v in keys} for g in keys}
    p = next(x for x in r["pairs"] if set(x["pair"]) == set(keys))
    footer = None
    if show_delta:
        footer = [("drop  qwen", p["drop"]["qwen"], False),
                  ("drop  gemma", p["drop"]["gemma"], False),
                  ("Δ", p["pooled"], True)]
    fig = _matrix(vals, keys, keys, caption=f"{qg['n_problems']} problems where both pools are split",
                  self_note="self", footer=footer)
    _save(fig, "matrix_qwen_gemma", save)
    return fig
