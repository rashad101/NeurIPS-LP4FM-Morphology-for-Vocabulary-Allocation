"""The paper's five figures.

Design rules applied (see the dataviz method): the form follows the data's job;
three validated categorical hues assigned in fixed order to *morphological type*
(the paper's variable), never cycled; one sequential blue ramp for the premium
heatmap; secondary encoding (marker shape + direct labels) so identity is never
colour-alone; recessive grid and axes; values direct-labelled selectively.

Every figure is written as PNG (300 dpi) and PDF (vector, for camera-ready).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .fonts import ensure_fonts, register_matplotlib
from .logging_utils import get_logger
from .progress import pbar

log = get_logger("figures")

# --- validated palette (light surface) -------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e4e3df"

# categorical slots 1-3 (all-pairs validated) mapped to morphological type
MORPH_COLORS = {
    "fusional": "#2a78d6",       # slot 1 blue
    "agglutinative": "#eb6834",  # slot 2 orange
    "analytic": "#1baf7a",       # slot 3 aqua
}
GROUP_MARKERS = {"mainstream": "o", "specialized": "D", "byte": "s"}
LEADER_RED = "#c1121f"  # label leader lines; distinct from the orange slot
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def _style() -> None:
    import matplotlib
    from matplotlib import rcParams

    matplotlib.use("Agg")
    ensure_fonts()
    families = register_matplotlib()
    rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["DejaVu Sans"] + families,
            "font.size": 9,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def _save(fig, name: str) -> list[Path]:
    paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = paths.FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(p)
        out.append(p)
    log.info("figure -> %s", paths.relpath(out[0]))
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("morphtok_blue", SEQ_BLUE)


def _place_labels(ax, items, *, fontsize=7.5, color=None, pad=2.0, avoid=(),
                  leader=None, leader_lw=0.6):
    """Annotate points with labels that do not overlap each other.

    ``items`` is a sequence of ``(x, y, text)`` in data coordinates. Each label
    is tried at a ring of candidate offsets (nearest first) and placed at the
    first one whose rendered box clears every label already placed and stays
    inside the axes; a leader line keeps it tied to its point. Fixed offsets
    collide whenever two labelled points are close, which is what made the
    top-left cluster of this figure unreadable.
    """
    color = INK_2 if color is None else color
    leader = INK_MUTED if leader is None else leader
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    candidates = [(10, 8), (-10, 8), (10, -12), (-10, -12), (14, 22), (-14, 22),
                  (14, -26), (-14, -26), (0, 32), (0, -34), (30, 4), (-30, 4),
                  (24, 40), (-24, 40), (24, -44), (-24, -44), (0, 52), (0, -54),
                  (38, 16), (-38, 16), (38, -20), (-38, -20)]
    placed: list = list(avoid)
    ax_box = ax.get_window_extent(renderer)

    from matplotlib.transforms import Bbox

    def _area(b) -> float:
        return max(0.0, b.width) * max(0.0, b.height)

    def _overlap(b1, b2) -> float:
        inter = Bbox.intersection(b1, b2)
        return 0.0 if inter is None else _area(inter)

    def _crosses(p1, p2, p3, p4) -> bool:
        # do segments p1p2 and p3p4 intersect (orientation test)
        def side(a, b, c):
            # float(): the coordinates arrive as numpy scalars, and bool
            # subtraction on those is an error rather than a sign
            v = float((b[0] - a[0]) * (c[1] - a[1])
                      - (b[1] - a[1]) * (c[0] - a[0]))
            return (v > 0) - (v < 0)
        d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
        d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
        return d1 != d2 and d3 != d4

    def _cost(box, seg) -> float:
        # collisions with other labels, anything spilling outside the axes, and
        # crossed leader lines. Two labels on nearby points can each be legible
        # yet still be ambiguous if their leaders tangle, so crossings are
        # penalised heavily rather than merely discouraged.
        c = (sum(_overlap(box, b) for b in placed)
             + 2.0 * (_area(box) - _overlap(box, ax_box)))
        c += 4.0 * _area(box) * sum(_crosses(seg[0], seg[1], s[0], s[1])
                                    for s in placed_segs)
        return c

    placed_segs: list = []
    dpi_scale = fig.dpi / 72.0
    # label the higher points first: they have the most empty space above them
    for x, y, text in sorted(items, key=lambda t: -t[1]):
        best: tuple[float, object, object, tuple] | None = None
        anchor = tuple(ax.transData.transform((x, y)))
        for dx, dy in candidates:
            ann = ax.annotate(
                text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=fontsize, color=color,
                ha="right" if dx < 0 else "left",
                va="top" if dy < 0 else "bottom",
                # panel-coloured plate: these labels sit over dense markers
                bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none",
                          alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=leader, lw=leader_lw,
                                shrinkA=0, shrinkB=2),
                zorder=6,
            )
            fig.canvas.draw()
            box = ann.get_window_extent(renderer).padded(pad)
            seg = (anchor, (anchor[0] + dx * dpi_scale, anchor[1] + dy * dpi_scale))
            cost = _cost(box, seg)
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, ann, box, seg)
            else:
                ann.remove()
            if best[0] <= 0.0:  # a clear spot; no need to look further out
                break
        # never fall back to a fixed offset: that stacks labels on dense points
        placed.append(best[2])
        placed_segs.append(best[3])


# ---------------------------------------------------------------------------
# Figure 1 — the thesis figure
# ---------------------------------------------------------------------------
def fig1_boundary_vs_fertility(merged: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    df = merged.dropna(subset=["boundary_f1", "fertility"])
    df = df[df.is_byte == 0]
    if df.empty:
        log.warning("fig1 skipped: no Part B metrics")
        return

    # This figure is set beside the body text in a ~2.9in column, so it is
    # drawn at that physical width and its type sizes are the sizes the reader
    # actually sees. Drawing it large and scaling down shrinks the text with it.
    fig, ax = plt.subplots(figsize=(2.95, 3.35))
    slopes: dict[str, float] = {}
    for mtype, sub in df.groupby("morph_type"):
        color = MORPH_COLORS.get(mtype, INK_2)
        for group, gsub in sub.groupby("group"):
            ax.scatter(
                gsub.boundary_f1, gsub.fertility,
                s=44 if group == "specialized" else 26,
                c=color, marker=GROUP_MARKERS.get(group, "o"),
                linewidths=1.0, edgecolors=SURFACE,
                alpha=0.95 if group == "specialized" else 0.75, zorder=3,
            )
        if len(sub) >= 4:  # simple slope per morphological type
            b, a = np.polyfit(sub.boundary_f1, sub.fertility, 1)
            slopes[mtype] = b
            xs = np.linspace(sub.boundary_f1.min(), sub.boundary_f1.max(), 50)
            ax.plot(xs, a + b * xs, color=color, lw=2, alpha=0.9, zorder=2)

    # Selective direct labels: the three worst-taxed cells, plus the frontier
    # corner. The specialized frontier points sit almost on top of each other,
    # so only two are named; at this figure's width more labels collide, and
    # the diamond marker already identifies the rest of the group.
    worst = df.nlargest(3, "fertility")
    best = df[(df.group == "specialized")
              & df.family.isin(["Indo-Aryan", "Dravidian"])].nsmallest(2, "fertility")
    labels = [
        (r.boundary_f1, r.fertility, f"{r.tokenizer}/{r.language}")
        for _, r in pd.concat([worst, best]).drop_duplicates(
            subset=["tokenizer", "language"]).iterrows()
    ]
    ax.margins(x=0.10, y=0.20)
    ax.set_ylim(bottom=0)  # fertility is a count ratio; negative space is dead

    ax.set_xlabel("morpheme-boundary F1 vs. gold morphemes", fontsize=7)
    ax.set_ylabel("fertility (tokens per word)", fontsize=7)
    ax.set_title("Boundary misalignment tracks the token tax,\nsteeply for agglutinative morphology",
                 fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    handles = [
        plt.Line2D([], [], color=c, marker="o", ls="-", lw=1.5, ms=3.6,
                   label=f"{k}  (slope {slopes[k]:+.1f})" if k in slopes else k)
        for k, c in MORPH_COLORS.items() if k in set(df.morph_type)
    ] + [
        plt.Line2D([], [], color=INK_2, marker=m, ls="", ms=3.6,
                   label=f"{g} tokenizer")
        for g, m in GROUP_MARKERS.items() if g in set(df.group)
    ]
    # Legend below the axes: inside the frame it covered the analytic cluster.
    # Marker size must stay well under the line spacing or adjacent handles
    # touch, which is what made the two group markers read as one shape.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncols=2, fontsize=6.2, handletextpad=0.6, columnspacing=1.2,
              labelspacing=1.0, handlelength=1.8, borderaxespad=0.2)
    # Red leaders: in the dense lower-right the grey lines did not read as
    # belonging to any particular marker, so which point a label named was
    # ambiguous. Red is used for nothing else in this figure.
    _place_labels(ax, labels, fontsize=6.0, pad=2.0, leader=LEADER_RED,
                  leader_lw=1.0)
    _save(fig, "fig1_boundary_f1_vs_fertility")


# ---------------------------------------------------------------------------
# Figure 2 — audit heatmap
# ---------------------------------------------------------------------------
def fig2_premium_heatmap(merged: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    value = "premium_corpus" if "premium_corpus" in merged.columns else "fertility"
    df = merged.dropna(subset=[value])
    if df.empty:
        log.warning("fig2 skipped")
        return
    order_groups = ["mainstream", "specialized", "byte"]
    rows: list[str] = []
    for g in order_groups:
        sub = df[df.group == g]
        rows += list(
            sub.groupby("tokenizer")[value].mean().sort_values().index
        )
    langs = (
        df.groupby(["family", "language"])[value].mean()
        .sort_values(ascending=False).reset_index()
    )
    lang_order = list(dict.fromkeys(langs.language))
    mat = df.pivot_table(index="tokenizer", columns="language", values=value,
                         aggfunc="mean").reindex(index=rows, columns=lang_order)

    fig, ax = plt.subplots(figsize=(1.05 + 0.52 * len(lang_order), 0.42 * len(rows) + 1.8))
    im = ax.imshow(mat.values, cmap=_seq_cmap(), aspect="auto",
                   vmin=1.0, vmax=float(np.nanpercentile(mat.values, 98)))
    ax.set_xticks(range(len(lang_order)), lang_order, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isnan(v):
                continue
            lo = v < np.nanpercentile(mat.values, 55)
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color=INK if lo else "#ffffff")
    # 2px surface gaps between tokenizer groups
    seen = 0
    for g in order_groups[:-1]:
        seen += df[df.group == g].tokenizer.nunique()
        if 0 < seen < len(rows):
            ax.axhline(seen - 0.5, color=SURFACE, lw=3)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("tokens relative to English (aligned sentences)")
    cbar.outline.set_visible(False)
    ax.set_title("The token tax by tokenizer, Indic first — specialized rows mark the attainable frontier")
    _save(fig, "fig2_premium_heatmap")


# ---------------------------------------------------------------------------
# Figure 3 — morpheme overlay panel (+ HTML twin with correct shaping)
# ---------------------------------------------------------------------------
def fig3_morpheme_overlay(qual_path: Path, languages: tuple[str, ...] = ("tam", "hin"),
                          tokenizers: tuple[str, ...] = ("o200k", "llama31", "gemma3",
                                                         "qwen3", "sarvam1")) -> None:
    import matplotlib.pyplot as plt

    if not Path(qual_path).is_file():
        log.warning("fig3 skipped: %s missing", paths.relpath(qual_path))
        return
    data = json.loads(Path(qual_path).read_text(encoding="utf-8"))
    picks = []
    for lang in languages:
        for ex in data.get(lang, [])[:1]:
            picks.append((lang, ex))
    if not picks:
        picks = [(k, v[0]) for k, v in list(data.items())[:2] if v]
    if not picks:
        log.warning("fig3 skipped: no qualitative examples")
        return

    fig, axes = plt.subplots(len(picks), 1, figsize=(8.4, 1.5 + 1.35 * len(picks)),
                             squeeze=False)
    for ax, (lang, ex) in zip(axes[:, 0], picks):
        word = ex["word"]
        n = len(word)
        toks = [t for t in tokenizers if t in ex["segmentations"]] or list(
            ex["segmentations"])[:5]
        ax.set_xlim(-0.02 * n, n * 1.02)
        ax.set_ylim(-0.6, len(toks) + 0.9)
        ax.axis("off")
        # gold morpheme rules
        for b in ex["gold_boundaries"]:
            ax.axvline(b, color=INK, lw=1.4, ls=(0, (3, 2)), zorder=1, alpha=0.65)
        # the word itself, with gold morphemes spaced
        ax.text(0, len(toks) + 0.35, "  ·  ".join(ex["gold_morphs"]), fontsize=13,
                color=INK, va="center")
        ax.text(n * 1.0, len(toks) + 0.35, f"gold: {ex['n_morphs'] if 'n_morphs' in ex else len(ex['gold_morphs'])} morphemes",
                fontsize=8, color=INK_MUTED, va="center", ha="right")
        for row, tk in enumerate(toks):
            seg = ex["segmentations"][tk]
            y = len(toks) - 1 - row
            cursor = 0
            for k, piece in enumerate(seg["pieces"]):
                w = len(piece)
                color = SEQ_BLUE[3 if k % 2 == 0 else 7]
                ax.add_patch(
                    plt.Rectangle((cursor + 0.04, y + 0.12), max(w - 0.08, 0.2), 0.62,
                                  facecolor=color, edgecolor=SURFACE, lw=1.5, zorder=2)
                )
                ax.text(cursor + w / 2, y + 0.43, piece, ha="center", va="center",
                        fontsize=11, color="#ffffff" if k % 2 else INK, zorder=3)
                cursor += w
            ax.text(-0.015 * n, y + 0.43, tk, ha="right", va="center", fontsize=8.5,
                    color=INK_2)
            ax.text(n * 1.015, y + 0.43, f"{seg['n_tokens']}", ha="left", va="center",
                    fontsize=8.5, color=INK_MUTED)
        ax.set_title(f"{lang}: {word}", loc="left", fontsize=10, pad=14)
    fig.suptitle("Where tokenizers cut, and where the morphemes are (dashed = gold boundary)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "fig3_morpheme_overlay")
    _fig3_html(picks, tokenizers)


def _fig3_html(picks: list[tuple[str, dict]], tokenizers: tuple[str, ...]) -> None:
    """Same panel as self-contained HTML — the browser shapes Indic text correctly."""
    css = f"""
    body {{ background:{SURFACE}; color:{INK}; font-family: system-ui, sans-serif;
            margin:24px; }}
    h1 {{ font-size:16px; }} h2 {{ font-size:13px; color:{INK_2}; margin:18px 0 6px; }}
    .row {{ display:flex; align-items:center; gap:10px; margin:4px 0; }}
    .tk {{ width:110px; text-align:right; font-size:12px; color:{INK_2}; }}
    .word {{ display:flex; }}
    .tok {{ padding:4px 6px; margin-right:2px; border-radius:4px; font-size:18px; }}
    .a {{ background:{SEQ_BLUE[3]}; color:{INK}; }}
    .b {{ background:{SEQ_BLUE[7]}; color:#fff; }}
    .gold .tok {{ background:transparent; border-bottom:2px dashed {INK}; color:{INK}; }}
    .n {{ font-size:12px; color:{INK_MUTED}; }}
    """
    parts = [f"<style>{css}</style>",
             "<h1>Morpheme–token overlay (browser-shaped text)</h1>"]
    for lang, ex in picks:
        parts.append(f"<h2>{lang} — {ex['word']}</h2>")
        gold = "".join(f"<span class='tok'>{m}</span>" for m in ex["gold_morphs"])
        parts.append(f"<div class='row gold'><span class='tk'>gold morphemes</span>"
                     f"<span class='word'>{gold}</span></div>")
        for tk in [t for t in tokenizers if t in ex["segmentations"]] or list(
                ex["segmentations"]):
            seg = ex["segmentations"][tk]
            spans = "".join(
                f"<span class='tok {'a' if i % 2 == 0 else 'b'}'>{p}</span>"
                for i, p in enumerate(seg["pieces"])
            )
            parts.append(f"<div class='row'><span class='tk'>{tk}</span>"
                         f"<span class='word'>{spans}</span>"
                         f"<span class='n'>{seg['n_tokens']} tokens</span></div>")
    out = paths.FIGURES_DIR / "fig3_morpheme_overlay.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    log.info("figure -> %s (open in a browser for correct Indic shaping)",
             paths.relpath(out))


# ---------------------------------------------------------------------------
# Figure 4 — Part C
# ---------------------------------------------------------------------------
def fig4_capability(joined_csv: Path) -> None:
    import matplotlib.pyplot as plt

    if not Path(joined_csv).is_file():
        log.warning("fig4 skipped: %s missing", paths.relpath(joined_csv))
        return
    df = pd.read_csv(joined_csv).dropna(subset=["fertility", "accuracy_loss"])
    if df.empty:
        log.warning("fig4 skipped: empty")
        return

    # drawn at the width it is shown at, so the type sizes are what a reader sees
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.8))
    ax = axes[0]
    for mtype, sub in df.groupby("morph_type"):
        ax.scatter(sub.fertility, sub.accuracy_loss, s=22,
                   c=MORPH_COLORS.get(mtype, INK_2), edgecolors=SURFACE, lw=1,
                   label=mtype, zorder=3)
    if len(df) >= 4:
        b, a = np.polyfit(df.fertility, df.accuracy_loss, 1)
        xs = np.linspace(df.fertility.min(), df.fertility.max(), 40)
        ax.plot(xs, a + b * xs, color=INK_2, lw=2, zorder=2)
    # Label only the cells that actually lost accuracy. Labelling all of them
    # piles ~50 names onto the cluster at the origin, where they are unreadable
    # and say nothing: those are the languages the budget did not bite.
    notable = df[df.accuracy_loss >= 0.05]
    loss_labels = [(r.fertility, r.accuracy_loss, r.language)
                   for _, r in notable.iterrows()]
    ax.set_xlabel("fertility (model's own tokenizer)", fontsize=7)
    ax.set_ylabel("accuracy loss under a fixed budget", fontsize=7)
    ax.set_title("Fertility predicts budgeted loss", fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.5, labelspacing=0.7, loc="upper left", handletextpad=0.5)

    ax = axes[1]
    if "boundary_f1" in df.columns and df.boundary_f1.notna().sum() >= 5:
        sub = df.dropna(subset=["boundary_f1"])
        # added-variable plot: residualise both sides on fertility
        rx = sub.boundary_f1 - np.poly1d(np.polyfit(sub.fertility, sub.boundary_f1, 1))(sub.fertility)
        ry = sub.accuracy_loss - np.poly1d(np.polyfit(sub.fertility, sub.accuracy_loss, 1))(sub.fertility)
        for mtype, idx in sub.groupby("morph_type").groups.items():
            ax.scatter(rx.loc[idx], ry.loc[idx], s=22,
                       c=MORPH_COLORS.get(mtype, INK_2), edgecolors=SURFACE, lw=1,
                       label=mtype, zorder=3)
        if len(sub) >= 4:
            b, a = np.polyfit(rx, ry, 1)
            xs = np.linspace(rx.min(), rx.max(), 40)
            ax.plot(xs, a + b * xs, color=INK_2, lw=2, zorder=2)
            ax.annotate(f"partial slope {b:+.3f}", xy=(0.03, 0.93),
                        xycoords="axes fraction", fontsize=7, color=INK)
        ax.axhline(0, color=GRID, lw=1)
        ax.axvline(0, color=GRID, lw=1)
        ax.set_xlabel("boundary F1 | fertility (residual)", fontsize=7)
        ax.set_ylabel("accuracy loss | fertility (residual)", fontsize=7)
        ax.set_title("Morphology beyond token count", fontsize=8)
        ax.tick_params(labelsize=6.5)
    else:
        ax.axis("off")
    # lay out first: the panels were close enough that the right y-label ran
    # into the left panel's plotting area
    fig.tight_layout(pad=0.3, w_pad=2.0)
    _place_labels(axes[0], loss_labels, fontsize=6.5)
    _save(fig, "fig4_capability_vs_fertility")


# ---------------------------------------------------------------------------
# Figure 5 — H4 English counter-cost
# ---------------------------------------------------------------------------
def fig5_english_cost(merged: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    eng = merged[(merged.language == "eng") & (merged.is_byte == 0)].copy()
    if eng.empty:
        log.warning("fig5 skipped: no English row")
        return
    indic = (merged[merged.family.isin(["Indo-Aryan", "Dravidian"])]
             .groupby("tokenizer").fertility.mean().rename("indic"))
    eng = eng.join(indic, on="tokenizer").sort_values(["group", "fertility"])
    x = np.arange(len(eng))
    # sized for full text width; at 0.7\linewidth this was shown at 38% scale
    fig, ax = plt.subplots(figsize=(5.5, 3.1))
    ax.bar(x - 0.19, eng.fertility, width=0.36, color=MORPH_COLORS["analytic"],
           label="English fertility", zorder=3)
    ax.bar(x + 0.19, eng.indic, width=0.36, color=MORPH_COLORS["agglutinative"],
           label="mean Indic fertility", zorder=3)
    for xi, (e, i) in enumerate(zip(eng.fertility, eng.indic)):
        ax.text(xi - 0.19, e + 0.06, f"{e:.2f}", ha="center", fontsize=5.2, color=INK_2)
        if not np.isnan(i):
            ax.text(xi + 0.19, i + 0.06, f"{i:.2f}", ha="center", fontsize=5.2, color=INK_2)
    # Tokenizer name only, rotated: "name\n(group)" is several times wider than
    # a bar slot, so the labels ran into each other. The group split is shown
    # once, as a divider, instead of being repeated under every bar.
    ax.set_xticks(x, list(eng.tokenizer), fontsize=6.2, rotation=30, ha="right",
                  rotation_mode="anchor")
    groups = list(eng.group)
    for gi in range(1, len(groups)):
        if groups[gi] != groups[gi - 1]:
            ax.axvline(gi - 0.5, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    top = float(np.nanmax([eng.fertility.max(), eng.indic.max()]))
    for g in dict.fromkeys(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        ax.text(sum(idx) / len(idx), top * 1.10, g, ha="center", fontsize=6.5,
                color=INK_2, style="italic")
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel("tokens per word", fontsize=7)
    ax.set_title("What specialization costs English", fontsize=8)
    ax.tick_params(axis="y", labelsize=6.5)
    # below the axes: at the top it ran into the group captions
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncols=2, frameon=False)
    _save(fig, "fig5_english_cost")


# ---------------------------------------------------------------------------
# Figure 6 — the prototype allocator (Part D)
# ---------------------------------------------------------------------------
ARM_COLORS = {
    "baseline": "#2a78d6",     # slot 1 — allocation by frequency
    "morph_train": "#eb6834",  # slot 2 — constrained learning only
    "morph_full": "#1baf7a",   # slot 3 — morphology-governed segmentation
}
ARM_LABELS = {
    "baseline": "frequency BPE",
    "morph_train": "morpheme-constrained training",
    "morph_full": "+ morphology-aware segmentation",
}


def fig6_prototype(arms_csv: Path) -> None:
    import matplotlib.pyplot as plt

    if not Path(arms_csv).is_file():
        log.warning("fig6 skipped: %s missing", paths.relpath(arms_csv))
        return
    df = pd.read_csv(arms_csv)
    if df.empty:
        return
    langs = sorted(df.language.unique())
    arms = [a for a in ARM_COLORS if a in set(df.arm)]
    x = np.arange(len(langs))
    width = 0.8 / max(1, len(arms))

    # drawn at the width it is shown at, so its type is the size a reader sees
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.9))
    for ax, metric, title, ylab in (
        (axes[0], "boundary_f1", "Morpheme-boundary alignment",
         "boundary F1 vs gold morphemes"),
        (axes[1], "fertility", "Cost of that alignment", "tokens per word"),
    ):
        for k, arm in enumerate(arms):
            sub = df[df.arm == arm].set_index("language").reindex(langs)
            vals = sub[metric].values
            pos = x + (k - (len(arms) - 1) / 2) * width
            ax.bar(pos, vals, width=width * 0.92, color=ARM_COLORS[arm],
                   label=ARM_LABELS[arm], zorder=3)
        # Per-bar value labels are dropped. The baseline and morph-train bars
        # differ in the third decimal, so their labels collided into an
        # unreadable run of digits; the values are in Table 4 instead.
        ax.set_xticks(x, langs, fontsize=6.5)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel(ylab, fontsize=7)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.margins(y=0.16)
    # below the panels: inside the left axes it sat on the tallest bars
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, fontsize=6.5, ncols=3, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.06),
               handletextpad=0.5, handlelength=1.4, columnspacing=1.6)
    fig.suptitle("A morphology-governed allocator, at matched vocabulary budget",
                 fontsize=8.5, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.3, w_pad=1.6, rect=(0, 0.07, 1, 1))
    _save(fig, "fig6_prototype_allocator")


# ---------------------------------------------------------------------------
def make_all(run_id: str | None = None) -> None:
    from .analysis import build_merged, find_stage_dir

    _style()
    merged_csv = paths.TABLES_DIR / "merged_tokenizer_language.csv"
    merged = pd.read_csv(merged_csv) if merged_csv.is_file() else build_merged(run_id)
    steps = [
        ("fig1", lambda: fig1_boundary_vs_fertility(merged)),
        ("fig2", lambda: fig2_premium_heatmap(merged)),
        ("fig5", lambda: fig5_english_cost(merged)),
    ]
    b_dir = find_stage_dir("part_b", run_id)
    if b_dir is not None:
        steps.append(("fig3", lambda: fig3_morpheme_overlay(
            b_dir / "qualitative_examples.json")))
    h3 = paths.TABLES_DIR / "h3_budget_loss_joined.csv"
    steps.append(("fig4", lambda: fig4_capability(h3)))
    d_dir = find_stage_dir("part_d", run_id)
    if d_dir is not None:
        steps.append(("fig6", lambda: fig6_prototype(d_dir / "prototype_arms.csv")))

    for name, fn in pbar(steps, desc="figures", unit="fig"):
        try:
            fn()
        except Exception as exc:
            log.exception("%s failed: %s", name, exc)
    log.info("figures written to %s", paths.relpath(paths.FIGURES_DIR))


# ---------------------------------------------------------------------------
# Figure 0 — the teaser: what frequency allocation does, and what we propose
# ---------------------------------------------------------------------------
def fig0_teaser() -> None:
    """Compact teaser for the introduction, set beside the opening text.

    Uses the real Tamil example of Part B, whose three gold morphemes span
    characters 0-11, 11-15 and 15-18. Qwen3 produces 19 tokens over 14 character
    spans, so it cuts the word in 13 places, only one of which is a gold
    boundary. Sarvam-1 produces two tokens, cutting once and on a gold boundary.
    Both segmentations are the measured ones, and the cut counts are derived
    from the drawn spans rather than written by hand. Text is plain rather than
    LaTeX, since matplotlib draws it directly.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow, Rectangle

    WORD = ("\u0b95\u0bc1\u0b9f\u0bbf\u0baf\u0bbf\u0bb0\u0bc1\u0baa\u0bcd"
            "\u0baa\u0bbe\u0bb3\u0bb0\u0bcd\u0b95\u0bb3\u0bcd")
    N = len(WORD)                                   # 18 characters
    GOLD_CHARS = [11, 15]
    GOLD = [g / N for g in GOLD_CHARS]
    QWEN = [(0,1),(1,2),(2,3),(3,5),(5,7),(7,8),(8,9),(9,11),
            (11,12),(12,13),(13,14),(14,16),(16,17),(17,18)]
    SARVAM = [(0, 11), (11, 18)]
    # Measured token counts. Qwen3 exceeds its 14 character spans because five
    # of its splits fall inside a codepoint and so are not character boundaries.
    QWEN_TOKENS, SARVAM_TOKENS = 19, 2

    def cut_note(spans: list[tuple[int, int]]) -> str:
        """How many of a vocabulary's cuts miss a gold boundary."""
        cuts = sorted({b for _, b in spans if 0 < b < N})
        missed = [c for c in cuts if c not in GOLD_CHARS]
        if not missed:
            return ("its cut lands on a boundary" if len(cuts) == 1
                    else "every cut lands on a boundary")
        return f"{len(missed)} of {len(cuts)} cuts miss"
    PROBLEM = MORPH_COLORS["agglutinative"]
    PROPOSE = MORPH_COLORS["analytic"]

    fig, ax = plt.subplots(figsize=(2.55, 2.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 96); ax.axis("off")
    X0, W = 6, 74

    def band(y0, accent, kicker, rule, spans, ntok, note):
        ax.add_patch(Rectangle((0, y0), 100, 42, facecolor=SURFACE,
                               edgecolor=GRID, lw=0.7, zorder=1))
        ax.add_patch(Rectangle((0, y0), 2.2, 42, facecolor=accent,
                               edgecolor="none", zorder=2))
        ax.text(X0, y0 + 36.5, kicker, fontsize=5.3, color=accent,
                weight="bold", va="center")
        ax.text(X0, y0 + 30.5, rule, fontsize=6.6, color=INK, weight="bold",
                va="center")
        ax.text(X0, y0 + 21.5, WORD, fontsize=8.5, color=INK, va="center")
        # token pieces, with a visible gutter so adjacent pieces stay distinct
        for a, b in spans:
            ax.add_patch(Rectangle((X0 + W * a / N + 0.6, y0 + 9.5),
                                   W * (b - a) / N - 1.2, 5.0,
                                   facecolor=accent, alpha=0.9,
                                   edgecolor=SURFACE, lw=0.6, zorder=3))
        for g in GOLD:
            ax.plot([X0 + W * g] * 2, [y0 + 7.8, y0 + 16.4], ls=(0, (2, 1.5)),
                    lw=1.0, color=INK, zorder=5)
        ax.text(X0, y0 + 3.8, ntok, fontsize=6.2, color=INK, weight="bold",
                va="center")
        ax.text(X0 + W, y0 + 3.8, note, fontsize=5.2, color=INK_2,
                va="center", ha="right")

    band(52, PROBLEM, "CURRENT PRACTICE", "Allocate by frequency",
         QWEN, f"{QWEN_TOKENS} tokens", cut_note(QWEN))
    band(2, PROPOSE, "ALREADY ATTAINABLE", "Respect the morphemes",
         SARVAM, f"{SARVAM_TOKENS} tokens", cut_note(SARVAM))
    ax.add_patch(FancyArrow(50, 50.5, 0, -3.4, width=0.5, head_width=2.6,
                            head_length=1.9, length_includes_head=True,
                            color=INK_MUTED, zorder=6))
    _save(fig, "fig0_teaser")
