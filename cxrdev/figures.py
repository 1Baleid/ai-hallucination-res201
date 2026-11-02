"""Figure 1 (example pair and yes-rates by arm) and PLUS Figure 2 (risk-coverage).

Figure 1. Panel a shows an example A/B pair from two NIH ChestX-ray14 PNGs with arrows at
student-provided coordinates (never RANZCR polylines), or placeholder text when no pair is
given. Panel b shows the yes-rate per arm (A, B and each U stratum) for every model (small
multiples) and prompt (bars, Wilson 95% error bars); the BLIND-text and BLIND-grey rows are single
deterministic generations, drawn as dashed reference lines at their first-token P(yes) with no
interval. Every value in the figure is also in Table 2 (its table view).

Style: categorical colours in a fixed order per prompt (validated palette: P0 blue, P1 orange,
P2-hi aqua, P2-lo yellow), thin bars separated by a surface gap, hairline solid gridlines, text in
ink colours (never the series colour), a legend always present. Matplotlib is used through its
object API (no pyplot state, no GUI backend); PDF and PNG metadata carry no timestamp, so the
same inputs give the same files.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cxrdev import metrics as M
from cxrdev.tables import MODEL_LABELS, PROMPT_LABELS, STRATUM_LABELS

__all__ = ["INK", "PROMPT_COLORS", "MODEL_COLORS", "YES_RATE_COLUMNS", "figure1", "figure2", "group_order", "yes_rate_frame"]

#: Fixed categorical order (validated with the dataviz palette validator, light mode).
PROMPT_COLORS: dict[str, str] = {"P0": "#2a78d6", "P1": "#eb6834", "P2hi": "#1baf7a", "P2lo": "#eda100"}
MODEL_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK: dict[str, str] = {
    "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "surface": "#ffffff",
}
YES_RATE_COLUMNS: tuple[str, ...] = ("model_key", "prompt_id", "group", "label", "rate", "lo", "hi", "k", "n")
BLIND_STYLES: dict[str, Any] = {"BLIND-text": (0, (5, 2)), "BLIND-grey": (0, (1.5, 1.5))}
_FONT = {"family": "sans-serif", "size": 7.5}


# ----------------------------------------------------------------------------- data for panel b
def group_order(groups: Sequence[str]) -> list[str]:
    """Panel-b groups in order: A, B, then the U strata in ``metrics.U_STRATUM_ORDER``, then others."""
    known = ["A", "B"] + [f"U_{s}" for s in M.U_STRATUM_ORDER]
    uniq = list(dict.fromkeys(str(g) for g in groups))
    return [g for g in known if g in uniq] + sorted(g for g in uniq if g not in known)


def _group_label(group: str) -> str:
    if group == "A":
        return "A (ETT)"
    if group == "B":
        return "B (no ETT)"
    if group.startswith("U_"):
        return "U " + STRATUM_LABELS.get(group[2:], group[2:])
    return group


def yes_rate_frame(summaries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Yes-rates per arm group from the primary ``metrics.summarise`` results (columns ``YES_RATE_COLUMNS``).

    Groups: A (hit_A), B (FAR_B) and one ``U_<stratum>`` per ``FAR_U_<stratum>`` key.
    """
    rows = []
    for s in summaries:
        if s.get("unparsable_as_wrong") or s.get("condition", "image") != "image":
            continue
        pairs = [("A", "hit_A"), ("B", "FAR_B")] + [(f"U_{k[6:]}", k) for k in s if k.startswith("FAR_U_")]
        for group, key in pairs:
            r = s.get(key)
            if not r:
                continue
            rows.append({"model_key": s["model_key"], "prompt_id": s["prompt_id"], "group": group,
                         "label": _group_label(group), "rate": r["estimate"], "lo": r["lo"], "hi": r["hi"],
                         "k": r["k"], "n": r["n"]})
    return pd.DataFrame(rows, columns=list(YES_RATE_COLUMNS))


# ----------------------------------------------------------------------------- drawing helpers
def _new_figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height), facecolor=INK["surface"])
    FigureCanvasAgg(fig)
    return fig


def _style_axis(ax: Any) -> None:
    ax.set_facecolor(INK["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK["secondary"], labelsize=_FONT["size"], length=2.5, width=0.6)
    ax.yaxis.grid(True, color=INK["grid"], linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def _save(fig: Figure, out_stem: str | Path) -> dict[str, Path]:
    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = stem.with_name(stem.name + ".pdf"), stem.with_name(stem.name + ".png")
    fig.savefig(pdf, format="pdf", facecolor=INK["surface"], metadata={"CreationDate": None, "Producer": None})
    fig.savefig(png, format="png", dpi=200, facecolor=INK["surface"], metadata={"Software": None})
    return {"pdf": pdf, "png": png}


def _placeholder(ax: Any, text: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines:
        ax.spines[side].set_color(INK["grid"])
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=_FONT["size"], color=INK["secondary"],
            transform=ax.transAxes, wrap=True)


def _draw_image(ax: Any, path: str | Path, arrow: Sequence[float] | None, title: str) -> None:
    from PIL import Image  # local import: only needed when a pair is drawn

    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=_FONT["size"], color=INK["primary"], pad=3)
    if arrow is not None:
        x, y = float(arrow[0]), float(arrow[1])
        h, w = arr.shape[:2]
        dx, dy = 0.16 * w, 0.12 * h
        tx = x + dx if x + dx < w else x - dx
        ty = max(0.0, y - dy)
        ann = ax.annotate("", xy=(x, y), xytext=(tx, ty),
                          arrowprops={"arrowstyle": "-|>", "color": "#ffffff", "lw": 1.6, "mutation_scale": 9})
        ann.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=3.2, foreground=INK["primary"]), patheffects.Normal()])


# ----------------------------------------------------------------------------- Figure 1
def figure1(
    yes_rates: pd.DataFrame,
    blind: pd.DataFrame | None,
    out_stem: str | Path,
    example_pair: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Draw Figure 1 and write ``<out_stem>.pdf`` and ``<out_stem>.png``; returns both paths.

    ``yes_rates`` has ``YES_RATE_COLUMNS`` (``yes_rate_frame``); ``blind`` has the
    ``metrics.blind_rows`` columns (or None). ``example_pair`` (optional) holds ``a_path`` and
    ``b_path`` (NIH PNG files) and optionally ``a_arrow`` / ``b_arrow`` = [x, y] in image pixels,
    ``a_title`` / ``b_title`` and ``caption``; without it panel a shows placeholder text.
    """
    fig = _new_figure(7.2, 3.35)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.0], wspace=0.16, left=0.02, right=0.99, top=0.88, bottom=0.19)

    # ---- panel a
    gsa = gs[0].subgridspec(1, 2, wspace=0.04)
    if example_pair and example_pair.get("a_path") and example_pair.get("b_path"):
        ax_a, ax_b = fig.add_subplot(gsa[0]), fig.add_subplot(gsa[1])
        try:
            _draw_image(ax_a, example_pair["a_path"], example_pair.get("a_arrow"), example_pair.get("a_title", "A: ETT present"))
            _draw_image(ax_b, example_pair["b_path"], example_pair.get("b_arrow"),
                        example_pair.get("b_title", "B: ETT absent, same patient"))
            caption = example_pair.get("caption", "NIH ChestX-ray14 (NIH Clinical Center; Wang et al., CVPR 2017)")
            ax_a.text(0.0, -0.06, caption, transform=ax_a.transAxes, ha="left", va="top", fontsize=5.5, color=INK["muted"])
        except (OSError, ValueError) as exc:
            ax_a.remove()
            ax_b.remove()
            _placeholder(fig.add_subplot(gsa[:]), f"Example pair could not be drawn:\n{type(exc).__name__}")
    else:
        _placeholder(fig.add_subplot(gsa[:]), "Example A/B pair\n(two NIH PNGs with arrow\ncoordinates supplied by the student)")
    fig.text(0.01, 0.95, "a", fontsize=10, fontweight="bold", color=INK["primary"], va="top")

    # ---- panel b
    yr = yes_rates.copy() if yes_rates is not None else pd.DataFrame(columns=list(YES_RATE_COLUMNS))
    models = M.ordered_models(yr["model_key"]) if len(yr) else []
    fig.text(0.355, 0.95, "b", fontsize=10, fontweight="bold", color=INK["primary"], va="top")
    if not models:
        _placeholder(fig.add_subplot(gs[1]), "No yes-rates available yet")
        return _save(fig, out_stem)
    prompts = M.ordered_prompts(yr["prompt_id"])
    gsb = gs[1].subgridspec(1, len(models), wspace=0.08)
    axes = []
    for mi, mk in enumerate(models):
        ax = fig.add_subplot(gsb[mi], sharey=axes[0] if axes else None)
        axes.append(ax)
        _style_axis(ax)
        sub = yr.loc[yr["model_key"] == mk]
        groups = group_order(sub["group"])
        width = min(0.8 / max(len(prompts), 1), 0.24)
        for pj, pid in enumerate(prompts):
            ps = sub.loc[sub["prompt_id"] == pid].set_index("group")
            xs, hs, lo_err, hi_err = [], [], [], []
            for gi, g in enumerate(groups):
                if g not in ps.index or not math.isfinite(float(ps.loc[g, "rate"])):
                    continue
                r = ps.loc[g]
                x = gi + (pj - (len(prompts) - 1) / 2.0) * width
                xs.append(x)
                hs.append(float(r["rate"]))
                lo_err.append(max(0.0, float(r["rate"]) - float(r["lo"])) if math.isfinite(float(r["lo"])) else 0.0)
                hi_err.append(max(0.0, float(r["hi"]) - float(r["rate"])) if math.isfinite(float(r["hi"])) else 0.0)
            if xs:
                ax.bar(xs, hs, width=width, color=PROMPT_COLORS.get(pid, INK["muted"]), edgecolor=INK["surface"],
                       linewidth=1.0, zorder=2)
                ax.errorbar(xs, hs, yerr=[lo_err, hi_err], fmt="none", ecolor=INK["secondary"], elinewidth=0.7,
                            capsize=1.6, capthick=0.7, zorder=3)
        if blind is not None and len(blind):
            for r in blind.loc[blind["model_key"].astype(str) == mk].itertuples(index=False):
                if not math.isfinite(float(r.p_yes_norm)):
                    continue
                ax.axhline(float(r.p_yes_norm), color=PROMPT_COLORS.get(str(r.prompt_id), INK["muted"]),
                           linestyle=BLIND_STYLES.get(str(r.condition), (0, (5, 2))), linewidth=1.0, zorder=4)
        ax.set_xticks(range(len(groups)))
        # two-line labels: five groups fit under the substitute (A, B, U >=2 lines, U 1 line, U no line)
        ax.set_xticklabels([_group_label(g).replace(" ", "\n", 1).replace(">=", "\u2265") for g in groups], fontsize=6.5,
                           color=INK["secondary"], rotation=0)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(MODEL_LABELS.get(mk, mk), fontsize=_FONT["size"] + 0.5, color=INK["primary"], pad=3)
        if mi == 0:
            ax.set_ylabel("Yes-rate (Wilson 95% CI)", fontsize=_FONT["size"], color=INK["secondary"])
        else:
            ax.tick_params(labelleft=False)
    handles = [Patch(facecolor=PROMPT_COLORS.get(p, INK["muted"]), edgecolor="none", label=PROMPT_LABELS.get(p, p))
               for p in prompts]
    if blind is not None and len(blind):
        handles += [Line2D([0], [0], color=INK["secondary"], linestyle=BLIND_STYLES["BLIND-text"], linewidth=1.0,
                           label="BLIND-text P(yes)"),
                    Line2D([0], [0], color=INK["secondary"], linestyle=BLIND_STYLES["BLIND-grey"], linewidth=1.0,
                           label="BLIND-grey P(yes)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.68, 0.005), ncol=len(handles), frameon=False,
               fontsize=6.5, labelcolor=INK["secondary"], handlelength=2.2, columnspacing=1.0)
    return _save(fig, out_stem)


# ----------------------------------------------------------------------------- PLUS Figure 2
def figure2(
    curves: Mapping[str, pd.DataFrame],
    verbal: pd.DataFrame | None,
    out_stem: str | Path,
    min_answered: int = 10,
) -> dict[str, Path]:
    """PLUS Figure 2: risk-coverage of the logit rule ranked by |z| (P0) per model, with the verbal
    abstention points (coverage = 1 - abstention, risk = error among answered) under P1 and P2-hi.

    ``curves`` maps model key -> ``metrics.risk_coverage`` frame; ``verbal`` has columns
    ``model_key``, ``prompt_id``, ``coverage``, ``risk``. Curve points with fewer than
    ``min_answered`` answered items are not drawn (the risk of a handful of items is noise).
    """
    fig = _new_figure(3.6, 3.0)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.92, bottom=0.27)
    _style_axis(ax)
    models = M.ordered_models(list(curves) + ([] if verbal is None else list(verbal["model_key"])))
    handles = []
    markers = {"P1": "o", "P2hi": "^", "P2lo": "s"}
    for i, mk in enumerate(models):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        cur = curves.get(mk)
        if cur is not None and len(cur):
            cur = cur.loc[cur["n_answered"] >= min_answered]
            ax.plot(cur["coverage"].to_numpy(), cur["risk"].to_numpy(), color=color, linewidth=1.6,
                    solid_joinstyle="round", solid_capstyle="round", zorder=2)
        handles.append(Line2D([0], [0], color=color, linewidth=1.6, label=f"{MODEL_LABELS.get(mk, mk)}, |z| ranking (P0)"))
        if verbal is not None and len(verbal):
            for r in verbal.loc[verbal["model_key"].astype(str) == mk].itertuples(index=False):
                if math.isfinite(float(r.coverage)) and math.isfinite(float(r.risk)):
                    ax.plot([float(r.coverage)], [float(r.risk)], marker=markers.get(str(r.prompt_id), "D"), markersize=6,
                            color=color, markeredgecolor=INK["surface"], markeredgewidth=1.5, linestyle="none", zorder=3)
    for pid, mkr in markers.items():
        if verbal is not None and len(verbal) and pid in set(verbal["prompt_id"].astype(str)):
            handles.append(Line2D([0], [0], marker=mkr, color=INK["secondary"], linestyle="none", markersize=5,
                                  label=f"verbal abstention, {PROMPT_LABELS.get(pid, pid)}"))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Coverage", fontsize=_FONT["size"], color=INK["secondary"])
    ax.set_ylabel("Risk (error among answered)", fontsize=_FONT["size"], color=INK["secondary"])
    fig.legend(handles=handles, loc="lower center", ncol=1 if len(handles) <= 3 else 2, frameon=False, fontsize=6,
               labelcolor=INK["secondary"])
    return _save(fig, out_stem)
