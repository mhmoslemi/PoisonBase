#!/usr/bin/env python3
"""
Best-so-far comparison plot across runs.

Reads the per-rollout .meta.json files a run writes under
runs/<name>/step<NN>/, takes the running maximum over valid rollouts, and
draws one curve per run on a single axis.

The best-so-far starts at 0: before any step has produced a valid solution
there is nothing found, so the curve begins at zero at step zero and steps up
at each discovery. Only steps where the running max actually increases get a
marker and a label with the new value; steps that repeat the previous value
stay bare.

Usage:

    python tmp.py \
        runs/circle_packing_n26_Qwen3-8B_0812-1928 \
        runs/circle_packing_n26_Qwen3-8B_0812-1933 \
        --labels "no memory" "memory" \
        --out best_so_far.png

Labels are optional. With none given, each run is labelled from its
config.json (`memory: true/false`) when that file is there, and from the
directory name otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # write files without needing a display
import matplotlib.pyplot as plt

_STEP_DIR_RE = re.compile(r"^step(\d+)$")

# Goal for circle packing n=26 (sum of radii).
DEFAULT_TARGET = 2.635983

# Two-series categorical palette, validated for a light surface.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"


# ----------------------------------------------------------------------
# Reading a run
# ----------------------------------------------------------------------
def read_config(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def step_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    """Numerically sorted (step_index, path) pairs. step10 must not sort before step2."""
    out = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = _STEP_DIR_RE.match(child.name)
        if m:
            out.append((int(m.group(1)), child))
    return sorted(out)


def score_of(meta: dict, metric: str) -> Optional[float]:
    """
    The number to maximize. `raw_score` is the problem's own metric (sum of
    radii for circle packing); `reward` is what the trainer optimizes. They
    differ whenever a problem rescales, so which one you plot matters and is
    a flag rather than a guess.
    """
    if not meta.get("valid"):
        return None
    value = meta.get(metric)
    if value is None and metric == "raw_score":
        value = meta.get("reward")     # problems that never set raw_score
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value


def read_run(run_dir: Path, metric: str) -> Dict[str, object]:
    """
    Returns steps, the per-step max, the running max, and the rollout counts.
    The running max starts at 0.0 -- nothing found yet -- so it is defined at
    every step, including the ones before the first valid rollout.
    """
    steps, step_best, running, n_valid, n_total = [], [], [], [], []
    best = 0.0

    for step_idx, sdir in step_dirs(run_dir):
        scores = []
        total = 0
        for meta_path in sorted(sdir.glob("*.meta.json")):
            total += 1
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            s = score_of(meta, metric)
            if s is not None:
                scores.append(s)

        this = max(scores) if scores else None
        if this is not None:
            best = max(best, this)

        steps.append(step_idx)
        step_best.append(this)
        running.append(best)
        n_valid.append(len(scores))
        n_total.append(total)

    return {"steps": steps, "step_best": step_best, "running": running,
            "n_valid": n_valid, "n_total": n_total}


def improvements(steps: List[int], running: List[float], min_delta: float,
                 precision: int) -> List[Tuple[int, float]]:
    """
    Steps where the running max strictly improved, with the new value.

    An improvement too small to show at the printed precision is tracked but
    not marked -- otherwise the plot gets a row of identical-looking labels
    for gains in the eighth decimal.
    """
    out, prev = [], 0.0
    for step, value in zip(steps, running):
        if value <= prev:
            continue
        visible = (f"{value:.{precision}f}" != f"{prev:.{precision}f}"
                   and value > prev + min_delta)
        if visible:
            out.append((step, value))
        prev = value
    return out


def auto_label(run_dir: Path) -> str:
    cfg = read_config(run_dir)
    if "memory" in cfg:
        return "memory" if cfg.get("memory") else "no memory"
    return run_dir.name


# ----------------------------------------------------------------------
# Terminal report
# ----------------------------------------------------------------------
def print_run(label: str, run: Dict[str, object], imps: List[Tuple[int, float]],
              precision: int) -> None:
    """Per-step best-so-far table, with a * on the steps that improved it."""
    imp_steps = {s for s, _ in imps}
    w = max(precision + 4, 14)

    print(f"\n=== {label} ===")
    print(f"{'step':>5}  {'best this step':>{w}}  {'best so far':>{w}}  "
          f"{'valid':>9}  {'new':>4}")
    for step, this, best, nv, nt in zip(run["steps"], run["step_best"],
                                        run["running"], run["n_valid"],
                                        run["n_total"]):
        this_s = f"{this:.{precision}f}" if this is not None else "-"
        best_s = f"{best:.{precision}f}"
        mark = "*" if step in imp_steps else ""
        print(f"{step:>5}  {this_s:>{w}}  {best_s:>{w}}"
              f"  {f'{nv}/{nt}':>9}  {mark:>4}")

    final = run["running"][-1] if run["running"] else 0.0
    last = imps[-1][0] if imps else "-"
    print(f"  -> {len(imps)} discoveries, best {final:.{precision}f} "
          f"(last improved at step {last}), "
          f"valid {sum(run['n_valid'])}/{sum(run['n_total'])} rollouts")


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def label_gaps(figsize, fontsize, precision, x_span, y_span):
    """
    How much room one label takes, in data units. Derived from the figure
    geometry rather than guessed, so labels only get pushed apart when they
    would genuinely overlap.
    """
    ax_w_pt = figsize[0] * 72 * 0.80       # the axes take ~80% of the width
    ax_h_pt = figsize[1] * 72 * 0.72       # ...and ~72% of the height
    text_w_pt = (precision + 2) * fontsize * 0.62
    row_h_pt = fontsize + 3.0
    return x_span * text_w_pt / ax_w_pt, y_span * row_h_pt / ax_h_pt


def place_labels(ax, imps, color, fontsize, precision, gaps, above):
    """
    Label each improvement with its new value. No box -- just the number, in
    the series colour. Every series keeps to one side of its own curve so
    labels never wander onto a neighbouring curve, and a label that would land
    on the previous one is pushed out another row.
    """
    x_gap, y_gap = gaps
    prev = None                            # (x, y, level)

    for sx, sy in imps:
        level = 0
        if prev is not None and abs(sx - prev[0]) < x_gap and abs(sy - prev[1]) < y_gap:
            level = prev[2] + 1
        prev = (sx, sy, level)

        dy = (fontsize + 2.5) * level
        # A label pushed off its own point gets a hairline back to it.
        arrow = dict(arrowstyle="-", color=color, alpha=0.45,
                     linewidth=0.6, shrinkA=1.0, shrinkB=3.0) if level else None
        ax.annotate(
            f"{sy:.{precision}f}",
            xy=(sx, sy),
            # Nudged right of the point: with steps-post the curve rises
            # vertically through the point, and a centred label sits on it.
            xytext=(4, (7 + dy) if above else -(10 + dy)),
            textcoords="offset points",
            ha="left",
            va="bottom" if above else "top",
            fontsize=fontsize,
            color=color,
            zorder=5,
            arrowprops=arrow,
        )


def draw_curves(ax, data, labels, imps_per_run, annotate, fontsize, precision,
                gaps, label_sides):
    """Draw every run on `ax`; label_sides=None means markers only (inset)."""
    for i, (label, run) in enumerate(zip(labels, data)):
        steps, running = run["steps"], run["running"]
        if not steps:
            continue
        color = SERIES_COLORS[i % len(SERIES_COLORS)]

        # Zero before anything is found: the curve leaves the origin only when
        # the first valid solution shows up.
        xs = [steps[0]] + list(steps)
        ys = [0.0] + list(running)

        # steps-post: the value holds until the next improvement, which is what
        # "best so far" actually means between discoveries.
        ax.plot(xs, ys, drawstyle="steps-post", linewidth=2.0, color=color,
                label=label, zorder=3, solid_joinstyle="round")

        imps = imps_per_run[i]
        if imps:
            ix, iy = zip(*imps)
            ax.scatter(ix, iy, s=28, color=color, zorder=4,
                       edgecolors="white", linewidths=1.2)

        if label_sides is not None and annotate != "none" and imps:
            chosen = {"all": imps, "first": imps[:1], "last": imps[-1:]}[annotate]
            place_labels(ax, chosen, color, fontsize, precision,
                         gaps, label_sides[i])


def main():
    ap = argparse.ArgumentParser(
        description="Plot best-so-far per step for one or more runs.")
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="one label per run; defaults to memory on/off from config.json")
    ap.add_argument("--metric", default="raw_score",
                    choices=["raw_score", "reward"],
                    help="which field to maximize (default: raw_score)")
    ap.add_argument("--out", default="best_so_far.png", help="output image path")
    ap.add_argument("--target", type=float, default=None,
                    help=f"horizontal goal line (default: config.json, else {DEFAULT_TARGET})")
    ap.add_argument("--no-target", action="store_true",
                    help="never draw the target line")
    ap.add_argument("--annotate", default="all",
                    choices=["all", "first", "last", "none"],
                    help="which improvements to label (default: all)")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="skip annotating improvements smaller than this")
    ap.add_argument("--precision", type=int, default=6,
                    help="decimals used for printing and labelling (default: 6)")
    ap.add_argument("--fontsize", type=float, default=7.0)
    ap.add_argument("--figsize", nargs=2, type=float, default=[10.0, 5.5])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default=None)
    ap.add_argument("--ylim", nargs=2, type=float, default=None,
                    help="force the y range instead of the automatic one")
    ap.add_argument("--inset", action="store_true",
                    help="add a zoomed inset over the top of the range, where "
                         "the curves crowd together")
    ap.add_argument("--inset-frac", type=float, default=0.08,
                    help="fraction of the y range the inset covers (default: 0.08)")
    args = ap.parse_args()

    run_dirs = [Path(r).expanduser() for r in args.runs]
    for d in run_dirs:
        if not d.is_dir():
            raise SystemExit(f"not a directory: {d}")

    labels = args.labels or [auto_label(d) for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit(f"got {len(labels)} labels for {len(run_dirs)} runs")

    data = [read_run(d, args.metric) for d in run_dirs]

    target = args.target
    if target is None and not args.no_target:
        for d in run_dirs:
            t = read_config(d).get("target")
            if t is not None:
                target = float(t)
                break
        if target is None:
            target = DEFAULT_TARGET

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # Ranges are needed before labelling so collisions can be judged in data
    # units; compute them from the curves, then draw.
    all_x = [s for run in data for s in run["steps"]]
    all_y = [v for run in data for v in run["running"]] + [0.0]
    if target is not None:
        all_y.append(target)
    if not all_x:
        raise SystemExit("no step directories found in any run")
    x_span = max(max(all_x) - min(all_x), 1)
    y_span = max(all_y) - min(all_y) or 1.0

    per_run_imps = []
    for label, run in zip(labels, data):
        if not run["steps"]:
            print(f"[warn] {label}: no step directories, skipping")
            per_run_imps.append([])
            continue
        imps = improvements(run["steps"], run["running"],
                            args.min_delta, args.precision)
        per_run_imps.append(imps)
        print_run(label, run, imps, args.precision)

    # Each series labels one side of its own curve, so labels stay off the
    # neighbouring curves: the run that finishes highest writes above, the
    # rest write below.
    finals = [run["running"][-1] if run["running"] else 0.0 for run in data]
    best_run = max(range(len(finals)), key=lambda i: finals[i])
    label_sides = [i == best_run for i in range(len(data))]

    gaps = label_gaps(args.figsize, args.fontsize, args.precision,
                      x_span, y_span)
    draw_curves(ax, data, labels, per_run_imps, args.annotate, args.fontsize,
                args.precision, gaps, label_sides)

    if target is not None:
        ax.axhline(target, linestyle=(0, (5, 4)), linewidth=1.1,
                   color=INK_MUTED, zorder=2)
        ax.annotate(f"goal {target:.{args.precision}f}", xy=(0.997, target),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points",
                    ha="right", va="bottom", fontsize=args.fontsize + 0.5,
                    color=INK_SOFT)

    ax.set_xlabel("search step", fontsize=args.fontsize + 2.5, color=INK_SOFT)
    ax.set_ylabel("best valid " + ("metric" if args.metric == "raw_score" else "reward")
                  + " so far", fontsize=args.fontsize + 2.5, color=INK_SOFT)
    ax.set_title(args.title or "Best-so-far per step",
                 fontsize=args.fontsize + 4, color=INK, pad=12, loc="left")

    ax.grid(True, axis="y", color=INK_MUTED, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=args.fontsize + 1.5, length=3)

    x_lo, x_hi = min(all_x) - x_span * 0.02, max(all_x) + x_span * 0.06
    ax.set_xlim(x_lo, x_hi)
    top = max(all_y)
    if args.ylim:
        ax.set_ylim(*args.ylim)
    else:
        ax.set_ylim(-y_span * 0.04, top + y_span * 0.10)

    if args.inset:      # keep clear of the inset and of the zero baseline
        leg = ax.legend(loc="center left", bbox_to_anchor=(0.55, 0.60),
                        frameon=False, fontsize=args.fontsize + 2.5)
    else:
        leg = ax.legend(loc="lower right", frameon=False,
                        fontsize=args.fontsize + 2.5)
    for text in leg.get_texts():
        text.set_color(INK_SOFT)

    # A zero baseline leaves the bottom of the axis empty while every late
    # discovery piles up in the last few percent; the inset puts that band
    # back at a readable scale without hiding the zero.
    if args.inset:
        zoom_lo = top - y_span * args.inset_frac
        axz = ax.inset_axes([0.34, 0.09, 0.64, 0.40], facecolor="white")
        axz.set_zorder(6)
        axz.patch.set_alpha(0.97)
        draw_curves(axz, data, labels, per_run_imps, "none", args.fontsize,
                    args.precision, gaps, None)
        if target is not None:
            axz.axhline(target, linestyle=(0, (5, 4)), linewidth=1.0,
                        color=INK_MUTED, zorder=2)
        axz.set_xlim(x_lo, x_hi)
        axz.set_ylim(zoom_lo, top + y_span * 0.012)
        axz.grid(True, axis="y", color=INK_MUTED, alpha=0.22, linewidth=0.6)
        axz.set_axisbelow(True)
        for side in ("top", "right"):
            axz.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axz.spines[side].set_color(INK_MUTED)
            axz.spines[side].set_linewidth(0.8)
        axz.tick_params(colors=INK_SOFT, labelsize=args.fontsize, length=2)
        axz.set_title(f"zoom: top {y_span * args.inset_frac:.3f}",
                      fontsize=args.fontsize, color=INK_SOFT, loc="left", pad=4)

    fig.tight_layout()
    out = Path(args.out).expanduser()
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(f"\nwrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
