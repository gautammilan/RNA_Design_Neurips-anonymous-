#!/usr/bin/env python3
import os
import json
import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Iterable, Any, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import math  # currently unused, but kept as in your version

# ──────────────────────────────────────────────────────────────────────────────
# Global Matplotlib font sizes
# ──────────────────────────────────────────────────────────────────────────────
BASE_FONT_SIZE = 14          # overall default
AXIS_LABEL_FONT_SIZE = 24    # x/y labels
TITLE_FONT_SIZE = 24         # plot titles
TICK_LABEL_FONT_SIZE = 20    # tick labels
LEGEND_FONT_SIZE = 18       # legend text
ANNOTATION_FONT_SIZE = 12    # e.g. SOTA text labels

# Apply to matplotlib rcParams so ticks, labels, etc., follow these defaults
plt.rcParams.update({
    "font.size": BASE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "xtick.labelsize": TICK_LABEL_FONT_SIZE,
    "ytick.labelsize": TICK_LABEL_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})

# ──────────────────────────────────────────────────────────────────────────────
# External: eval function
# ──────────────────────────────────────────────────────────────────────────────
from evaluation import eval_design

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

SOTA_DEFAULT = {
    "solve_rate":       0.80,
    "umfe_rate":        0.79,
    "structural_dist":  0.60,
    "ensemble_defect":  0.013250552874116384,
    "probability":      0.722988425960639,
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _safe_prefix(prefix: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in prefix)

def is_valid_seq(seq: str, structure: str) -> bool:
    if len(seq) != len(structure):
        return False
    valid_pairs = {"CG", "GC", "AU", "UA", "GU", "UG"}
    stack = []
    for idx, ch in enumerate(structure):
        if ch == "(":
            stack.append(idx)
        elif ch == ")":
            if not stack:
                return False
            j = stack.pop()
            if seq[j] + seq[idx] not in valid_pairs:
                return False
        elif ch != ".":
            return False
    return not stack

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_records(path: str) -> List[Dict[str, Any]]:
    out = []
    for rec in iter_jsonl(path):
        if "best_design_per_turn" not in rec and "designed_sequence" in rec:
            rec["best_design_per_turn"] = [rec["designed_sequence"]]
        out.append(rec)
    return out

def group_by_id(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    d = collections.defaultdict(list)
    for rec in records:
        key = rec.get("id", rec.get("target_structure"))
        d[key].append(rec)
    return d

# ──────────────────────────────────────────────────────────────────────────────
# Global cache checkpoint load/save
# ──────────────────────────────────────────────────────────────────────────────
def load_cache(cache_path: str) -> pd.DataFrame:
    """
    Load a shared global parquet cache if it exists, else return an empty DataFrame.
    Cache is deduplicated on (target_structure, sequence).
    """
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            before = len(df)
            df = df.drop_duplicates(subset=["target_structure", "sequence"])
            after = len(df)
            if after != before:
                logger.info(f"Deduplicated cache {cache_path}: {before} -> {after} rows")
            logger.info(f"Loaded global cache: {cache_path} ({after} rows)")
            return df
        except Exception as e:
            logger.warning(f"Failed loading global cache {cache_path}: {e}")
    else:
        logger.info(f"No existing global cache at {cache_path}; starting fresh.")
    return pd.DataFrame(columns=[
        "target_structure", "sequence", "mfe_structures",
        "structural_dist", "probability", "ensemble_defect",
        "energy_diff", "is_mfe", "is_umfe"
    ])

def save_cache(df: pd.DataFrame, cache_path: str):
    """
    Save the shared global cache, always deduplicated on (target_structure, sequence).
    """
    df = df.drop_duplicates(subset=["target_structure", "sequence"])
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Saved global cache: {cache_path} ({len(df)} rows)")

# <<< helper for geometric mean of positive values
def _geom_mean(values: List[float]) -> Optional[float]:
    vals = [float(v) for v in values if v > 0.0]
    if not vals:
        return None
    logs = np.log(vals)
    return float(np.exp(np.mean(logs)))

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation (with shared global cache)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_pair(args: Tuple[str, str]) -> Dict[str, Any]:
    seq, ss = args
    return eval_design(seq, ss)

def _row_from_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_structure": m["target_structure"],
        "sequence":         m["sequence"],
        "mfe_structures":   m.get("mfe_structures", []),
        "structural_dist":  m["structural_dist"],
        "probability":      m["probability"],
        "ensemble_defect":  m["ensemble_defect"],
        "energy_diff":      m.get("energy_diff", None),
        "is_mfe":           bool(m.get("is_mfe", False)),
        "is_umfe":          bool(m.get("is_umfe", False)),
    }

def evaluate_all_with_cache(records: List[Dict[str, Any]],
                                  desc: str,
                                  exp_cache_path: str,
                                  timing_log_path: str,
                                  n_workers: int,
                                  batch_size: int = 10000):
    """
    Evaluate only missing UNIQUE (target_structure, sequence) pairs using a
    shared global parquet cache at exp_cache_path. Append new rows to the
    global cache as needed, and log per-batch timing to timing_log_path.
    """

    # Load global cache (deduped)
    cache_df = load_cache(exp_cache_path)
    logger.info(f"Loaded {cache_df.shape[0]} rows from global cache {exp_cache_path}")
    cache_index = set(zip(
        cache_df["target_structure"].astype(str),
        cache_df["sequence"].astype(str)
    ))

    num_input = len(records)
    logger.info(f"{desc}: Loaded records: {num_input}, global cache has {len(cache_df)} rows")

    # Separate valid/invalid + fallback
    invalid = [rec for rec in records if not is_valid_seq(rec.get("designed_sequence",""), rec["target_structure"])]
    valid   = [rec for rec in records if     is_valid_seq(rec.get("designed_sequence",""), rec["target_structure"])]

    valid_ids = {rec.get("id", rec["target_structure"]) for rec in valid}
    all_ids   = {rec.get("id", rec["target_structure"]) for rec in records}
    missing_ids = all_ids - valid_ids

    if missing_ids:
        logger.info(f"{desc}: Adding {len(missing_ids)} all-A fallbacks")
        for mid in missing_ids:
            placeholder = next(r for r in invalid if r.get("id", r.get("target_structure")) == mid)
            L = len(placeholder["target_structure"])
            seqA = "A" * L
            new_rec = {
                **placeholder,
                "designed_sequence": seqA,
                "_was_allA_fallback": True,
                "best_design_per_turn": [seqA],
            }
            valid.append(new_rec)

    # Collect all UNIQUE (target_structure, sequence) pairs
    all_pairs: set[Tuple[str, str]] = set()
    for rec in valid:
        tgt = rec["target_structure"]
        main = rec["designed_sequence"]
        all_pairs.add((tgt, main))
        for s in rec.get("best_design_per_turn", [main]):
            all_pairs.add((tgt, s))

    # Identify unique pairs missing from global cache
    missing_pairs = sorted(
        (tgt, seq)
        for (tgt, seq) in all_pairs
        if (tgt, seq) not in cache_index
    )
    logger.info(
        f"{desc}: {len(missing_pairs)} unique (structure, sequence) pairs "
        f"missing from global cache out of {len(all_pairs)} unique pairs total"
    )

    # Evaluate missing unique pairs in batches
    new_rows: List[Dict[str, Any]] = []
    if missing_pairs:
        total_batches = (len(missing_pairs) + batch_size - 1) // batch_size
        for b in range(total_batches):
            start = b * batch_size
            end   = min(start + batch_size, len(missing_pairs))
            batch_pairs = missing_pairs[start:end]
            logger.info(f"{desc}: Batch {b+1}/{total_batches} size={len(batch_pairs)}")

            t0 = time.time()
            # evaluate_pair expects (seq, ss)
            args_list = [(seq, tgt) for (tgt, seq) in batch_pairs]
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                for metrics in tqdm(ex.map(evaluate_pair, args_list),
                                    total=len(batch_pairs),
                                    desc=f"{desc} batch {b+1}/{total_batches}"):
                    row = _row_from_metrics(metrics)
                    new_rows.append(row)
                    cache_index.add((row["target_structure"], row["sequence"]))

            t1 = time.time()
            total_time = t1 - t0
            time_per_seq = total_time / len(batch_pairs) if batch_pairs else float("nan")

            # timing log (per-experiment / per-SOTA, but shared cache)
            os.makedirs(os.path.dirname(timing_log_path) or ".", exist_ok=True)
            with open(timing_log_path, "a") as f:
                rec_log = {
                    "desc": desc,
                    "batch_index": b + 1,
                    "num_batches": total_batches,
                    "num_pairs": len(batch_pairs),
                    "total_time_sec": total_time,
                    "time_per_seq_sec": time_per_seq,
                }
                f.write(json.dumps(rec_log) + "\n")

    # append to global cache and save
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cache_df = pd.concat([cache_df, new_df], ignore_index=True)
        save_cache(cache_df, exp_cache_path)

    # Build metrics lookup from deduped cache
    cache_df = cache_df.drop_duplicates(subset=["target_structure", "sequence"])
    lookup = {
        (row.target_structure, row.sequence): {
            "mfe_structures": row.mfe_structures,
            "structural_dist": row.structural_dist,
            "probability": row.probability,
            "ensemble_defect": row.ensemble_defect,
            "energy_diff": row.energy_diff,
            "is_mfe": bool(row.is_mfe),
            "is_umfe": bool(row.is_umfe),
            "sequence": row.sequence,
            "target_structure": row.target_structure,
        }
        for row in cache_df.itertuples()
    }

    # Attach metrics
    for rec in valid:
        tgt = rec["target_structure"]
        main = rec["designed_sequence"]
        key = (tgt, main)
        rec["metrics"] = lookup[key]
        rec["turn_metrics"] = [lookup[(tgt, s)] for s in rec.get("best_design_per_turn", [main])]

    return valid

# ──────────────────────────────────────────────────────────────────────────────
# SOTA loaders (two formats)
# ──────────────────────────────────────────────────────────────────────────────
def detect_sota_format(path: str) -> str:
    """Return 'summary' if nested {'value', ...} under metrics-like keys; else 'runs'."""
    first = next(iter_jsonl(path), None)
    if not first:
        return "empty"
    prob = first.get("probability")
    if isinstance(prob, dict) and "value" in prob:
        return "summary"
    return "runs"

def _get_nested_value(rec: Dict[str, Any], keys: List[str]) -> Optional[float]:
    """Fetch nested {'value': x} for any canonical key variant."""
    for k in keys:
        v = rec.get(k)
        if isinstance(v, dict) and "value" in v:
            return float(v["value"])
    return None

def aggregate_sota_summary(path: str,
                           gm_exclude_ids: Optional[set] = None
                           ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Read SOTA summary file and compute mean metrics + per-structure probability map.
    Returns (sota_vals, sota_prob_by_structure).
    """
    if gm_exclude_ids is None:
        gm_exclude_ids = set()
    gm_exclude_ids = {str(x) for x in gm_exclude_ids}

    prob_keys = ["probability", "prob"]
    ned_keys  = ["ensemble_defect", "ned"]
    sd_keys   = ["structural_dist", "sd", "structural_distance"]
    mfe_keys  = ["mfe", "solve_rate"]
    umfe_keys = ["umfe", "umfe_rate"]

    probs, neds, sds, mfes, umfes = [], [], [], [], []
    prob_by_struct: Dict[str, float] = {}

    # for geometric mean
    gm_probs: List[float] = []

    for rec in iter_jsonl(path):
        ss = rec["target_structure"]
        ss_str = str(ss)

        p  = _get_nested_value(rec, prob_keys)
        ed = _get_nested_value(rec, ned_keys)
        sd = _get_nested_value(rec, sd_keys)
        mf = _get_nested_value(rec, mfe_keys)
        uf = _get_nested_value(rec, umfe_keys)

        if p is not None:
            probs.append(p)
            prob_by_struct[ss] = p
            # include in geometric mean only if not excluded and > 0
            if ss_str not in gm_exclude_ids and p > 0.0:
                gm_probs.append(p)
        if ed is not None:
            neds.append(ed)
        if sd is not None:
            sds.append(sd)
        if mf is not None:
            mfes.append(mf)
        if uf is not None:
            umfes.append(uf)

    def safe_mean(x: List[float]) -> Optional[float]:
        return float(np.mean(x)) if x else None

    sota_vals: Dict[str, float] = {
        "probability":      safe_mean(probs),
        "ensemble_defect":  safe_mean(neds),
        "structural_dist":  safe_mean(sds),
        "solve_rate":       safe_mean(mfes),
        "umfe_rate":        safe_mean(umfes),
    }

    # Fill with defaults if missing
    for k, v in list(sota_vals.items()):
        if v is None and k in SOTA_DEFAULT:
            sota_vals[k] = SOTA_DEFAULT[k]

    # geometric mean for probability (SOTA summary)
    gmean = _geom_mean(gm_probs)
    sota_vals["probability_gmean"] = gmean
    sota_vals["probability_gmean_count"] = len(gm_probs)
    logger.info(f"[geom-mean] SOTA(summary): geometric mean over {len(gm_probs)} structures")

    return sota_vals, prob_by_struct

def compute_sota_metrics(sota_groups: Dict[str, List[Dict[str, Any]]],
                         gm_exclude_ids: Optional[set] = None,
                         name: str = "experiment"
                         ) -> Dict[str, float]:
    """
    Aggregate metrics across IDs: best solve, min dist/defect, max prob; avg umfe.
    Also compute geometric mean of best probabilities across IDs, excluding any IDs
    or target structures listed in gm_exclude_ids.
    """
    if gm_exclude_ids is None:
        gm_exclude_ids = set()
    gm_exclude_ids = {str(x) for x in gm_exclude_ids}

    solve_rates, dists, probs, neds, umfes = [], [], [], [], []
    gm_probs: List[float] = []

    for key, runs in tqdm(sota_groups.items(), desc=f"Computing SOTA metrics ({name})"):
        ml = [r["metrics"] for r in runs]

        # standard aggregate metrics
        solve_rates.append(max(m["is_mfe"] for m in ml))
        dists.append(min(m["structural_dist"] for m in ml))
        neds.append(min(m["ensemble_defect"] for m in ml))
        probs.append(max(m["probability"] for m in ml))
        if any("is_umfe" in m for m in ml):
            umfes.append(max(bool(m.get("is_umfe", False)) for m in ml))

        # geometric-mean candidates: best prob per id/structure
        key_str = str(key)
        struct = str(ml[0]["target_structure"])
        if key_str in gm_exclude_ids or struct in gm_exclude_ids:
            continue
        best_p = max(m["probability"] for m in ml)
        if best_p > 0.0:
            gm_probs.append(best_p)

    out = {
        "solve_rate":      float(np.mean(solve_rates)) if solve_rates else float("nan"),
        "structural_dist": float(np.mean(dists))       if dists       else float("nan"),
        "ensemble_defect": float(np.mean(neds))        if neds        else float("nan"),
        "probability":     float(np.mean(probs))       if probs       else float("nan"),
    }
    if umfes:
        out["umfe_rate"] = float(np.mean(umfes))

    gmean = _geom_mean(gm_probs)
    out["probability_gmean"] = gmean
    out["probability_gmean_count"] = len(gm_probs)
    logger.info(f"[geom-mean] {name}: geometric mean over {len(gm_probs)} ids/structures")

    return out

def best_prob_by_structure(groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    From runs (already evaluated), get baseline best probability per structure.
    """
    out: Dict[str, float] = {}
    for runs in groups.values():
        ss = runs[0]["metrics"]["target_structure"]
        best_p = max(r["metrics"]["probability"] for r in runs)
        out[ss] = best_p
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def best_of_n_curve(groups: Dict[str, List[Dict[str, Any]]],
                    max_n: int,
                    metric: str,
                    log_base: Optional[int] = None,
                    num_points: int = 100) -> Tuple[List[int], List[float]]:
    """Compute (possibly subsampled) best-of-N curve for groups."""
    min_runs = min(len(runs) for runs in groups.values())
    max_plot = min(max_n, min_runs)

    if log_base:
        max_exp = np.log(max_plot) / np.log(log_base)
        raw = np.logspace(0, max_exp, num=num_points, base=log_base)
        n_list = np.unique(np.round(raw).astype(int))
        n_list = n_list[(n_list >= 1) & (n_list <= max_plot)]
    else:
        n_list = np.arange(1, max_plot + 1, dtype=int)

    xs: List[int] = []
    ys: List[float] = []

    for n in tqdm(n_list, desc="Best-of-N points", leave=False):
        vals = []
        for runs in groups.values():
            subset = runs[:n]
            if metric == "solve_rate":
                v = max(r["metrics"]["is_mfe"] for r in subset)
            elif metric == "umfe_rate":
                v = max(r["metrics"].get("is_umfe", False) for r in subset)
            elif metric == "structural_dist":
                v = min(r["metrics"]["structural_dist"] for r in subset)
            elif metric == "ensemble_defect":
                v = min(r["metrics"]["ensemble_defect"] for r in subset)
            else:  # "probability"
                v = max(r["metrics"]["probability"] for r in subset)
            vals.append(v)
        xs.append(int(n))
        ys.append(float(np.mean(vals)))
    return xs, ys

def _beautify(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        n = len(handles)

        new_handles = handles
        new_labels = labels
        ncol = 1 if n <= 3 else 2

        ax.legend(
            new_handles,
            new_labels,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            loc="center",
            borderaxespad=0.6,
            ncol=ncol,
        )

    # leave extra space at bottom so legend isn't cut off
    plt.tight_layout(rect=[0, 0.08, 1, 1])

def plot_best_of_n(groups_list: List[Dict[str, List[Dict[str, Any]]]],
                   exp_names: List[str],
                   sota_vals: Dict[str, float],
                   max_n: int,
                   out_dir: str,
                   log_base: Optional[int] = None,
                   line_styles: Optional[List[str]] = None,
                   marker_styles: Optional[List[str]] = None,
                   colors: Optional[List[str]] = None):
    # If no colors are provided, fall back to a default palette
    if colors is None:
        colors = [
            "#0072B2", "#451267", "#CF5D41", "#F0AA4F",
            "#F0C84F", "#01010A", "#892F64",
        ]

    metric_specs = [
        ("solve_rate",      "Solve Rate"),
        ("umfe_rate",       "UMFE Rate"),
        ("structural_dist", "Structural Distance"),
        ("ensemble_defect", "Normalized Ensemble Defect"),
        ("probability",     "Boltzmann Probability"),
    ]

    for metric, title in metric_specs:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot experiments
        for i, (groups, name) in enumerate(zip(groups_list, exp_names)):
            xs, ys = best_of_n_curve(
                groups,
                max_n,
                metric,
                log_base=log_base,
                num_points=50,
            )

            # linestyle: mostly for aesthetics, keep all solid by default
            if line_styles is not None and len(line_styles) > 0:
                ls = line_styles[i % len(line_styles)]
            else:
                ls = "-"

            # marker: main way to distinguish experiments
            if marker_styles is not None and len(marker_styles) > 0:
                mk = marker_styles[i % len(marker_styles)]
            else:
                mk = "o"   # default marker

            color = colors[i % len(colors)]

            ax.plot(
                xs,
                ys,
                marker=mk,
                markersize=4,
                color=color,
                linestyle=ls,
                linewidth=1.5,
                alpha=0.8,
                label=name,
            )

        # Draw SOTA horizontal line (no legend entry)
        sota_value = sota_vals.get(metric, None)
        has_sota = sota_value is not None
        if has_sota:
            ax.axhline(
                sota_value,
                linestyle="--",
                linewidth=1.5,
                color="red",
            )

        # Invert y for "lower is better" metrics
        if metric in ["structural_dist", "ensemble_defect"]:
            ax.invert_yaxis()

        # Log-scale on x if requested
        if log_base:
            ax.set_xscale("log", base=log_base)

        # Common styling + legend
        _beautify(
            ax,
            xlabel="Number of Samples (N)",
            ylabel=title,
            title=f"Best-of-N {title}",
        )

        # Add SOTA text label just above (or below, for inverted y) the line
        if has_sota:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            dy = (y_max - y_min) * 0.005  # 0.5% of axis range

            if ax.yaxis_inverted():
                # For inverted y-axis, visually above is a smaller y value
                y_text = sota_value - dy
                va = "top"
            else:
                y_text = sota_value + dy
                va = "bottom"

            dx = (x_max - x_min) * 0.01
            x_text = x_max - dx

            ax.text(
                x_text,
                y_text,
                f"SOTA = {sota_value:.3f}",
                fontsize=ANNOTATION_FONT_SIZE,
                color="red",
                ha="right",
                va=va,
            )

        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"best_of_N_{metric}.png"))
        fig.savefig(os.path.join(out_dir, f"best_of_N_{metric}.pdf"))
        plt.close(fig)

def plot_turns_histogram(groups_list: List[Dict[str, List[Dict[str, Any]]]],
                         exp_names: List[str],
                         out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for groups, name in zip(groups_list, exp_names):
        lengths = [
            len(r.get("best_design_per_turn", [r["metrics"]["sequence"]]))
            for runs in groups.values()
            for r in runs
        ]
        ax.hist(lengths, bins=20, alpha=0.5, edgecolor="black", label=name)
    _beautify(ax,
              xlabel="Number of Turns",
              ylabel="Frequency",
              title="Distribution of Number of Turns")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "turns_histogram.png"))
    plt.close(fig)

def plot_prob_diff_vs_length(groups: Dict[str, List[Dict[str, Any]]],
                             sota_prob_by_structure: Dict[str, float],
                             out_dir: str,
                             exp_safe_name: str):
    xs_len, ys_delta = [], []
    for runs in groups.values():
        ss = runs[0]["metrics"]["target_structure"]
        L = len(ss)
        if ss not in sota_prob_by_structure:
            continue
        exp_best = max(r["metrics"]["probability"] for r in runs)
        base = sota_prob_by_structure[ss]
        xs_len.append(L)
        ys_delta.append(exp_best - base)

    if not xs_len:
        logger.info(f"[Δprob vs length] No overlaps for {exp_safe_name}. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(xs_len, ys_delta, s=16, alpha=0.7, label=exp_safe_name)
    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="red", label="Δ=0")
    _beautify(ax,
              xlabel="Target Length",
              ylabel="Δ Probability (Exp − SOTA)",
              title=f"Δ Probability vs Length — {exp_safe_name}")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"delta_prob_vs_length__{exp_safe_name}.png"))
    fig.savefig(os.path.join(out_dir, f"delta_prob_vs_length__{exp_safe_name}.pdf"))
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Metrics summary & JSON
# ──────────────────────────────────────────────────────────────────────────────
def collect_experiment_metrics(exp_names: List[str],
                               groups_list: List[Dict[str, List[Dict[str, Any]]]],
                               sota_vals: Optional[Dict[str, float]] = None,
                               gm_exclude_ids: Optional[set] = None
                               ) -> Dict[str, Dict[str, float]]:
    exp_metrics = {
        name: compute_sota_metrics(
            groups,
            gm_exclude_ids=gm_exclude_ids,
            name=name,
        )
        for name, groups in zip(exp_names, groups_list)
    }
    if sota_vals is not None:
        return {"SOTA": sota_vals, **exp_metrics}
    return exp_metrics

def save_metrics_json(out_dir: str,
                      metrics: Dict[str, Dict[str, float]],
                      filename: str = "metrics_summary.json") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics summary to {path}")
    return path

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare multiple RNA-design experiments (shared global cache)")
    parser.add_argument(
        "--results", nargs="+",
        default=[],
        help="JSONL files for each experiment (raw runs or pre-evaluated)",
    )
    parser.add_argument(
        "--exp_names", nargs="+",
        default=[],
        help="Names for each experiment, same length/order as --results",
    )
    parser.add_argument(
        "--sota", type=str,
        default="/nfs/stak/users/gautammi/my-hpc-share/workspace/research/research/RNADesign/results/eterna100_SAMFEO_bo10.jsonl",
        help="Path to SOTA JSONL (either summary or runs)",
    )
    parser.add_argument(
        "--out_dir", type=str,
        default="./decoding_results/eterna100_SAMFEO.jsonl",
        help="Directory to save plots, caches, and timing logs",
    )
    parser.add_argument(
        "--cache_path", type=str,
        default="../eval_cache_sept.parquet",
        help=(
            "Path to a single global parquet cache shared across all experiments "
            "and SOTA. Default: <out_dir>/eval_cache_global.parquet"
        ),
    )
    parser.add_argument(
        "--max_n", type=int, default=100000,
        help="Max runs for Best-of-N",
    )
    parser.add_argument(
        "--max_turns", type=int, default=10,
        help="Max turns to plot (unused here)",
    )
    parser.add_argument(
        "--n_workers", type=int, default=96,
        help="Processes for evaluation",
    )
    parser.add_argument(
        "--log-base", type=int, choices=[2, 10], default=None,
        help="Plot x-axis in log scale with the given base",
    )
    parser.add_argument(
        "--line-styles", nargs="+",
        default=["--", "-", "-", "--", "-"],
        help=(
            'Matplotlib line styles for each experiment '
            '(e.g. "--", "-", "-", "--", "-"). If fewer than experiments are '
            'provided, they will be cycled.'
        ),
    )
    parser.add_argument(
        "--markers", nargs="+",
        default=["", "o", "", "", "o"],
        help=(
            "Matplotlib marker styles for each experiment "
            "(e.g. 'o', '^', 's', 'D', 'v', 'P'). If fewer than experiments "
            "are provided, they will be cycled."
        ),
    )
    parser.add_argument(
        "--colors", nargs="+",
        default=[
            "#0072B2", "#0072B2", "#CF5D41", "#451267",
            "#451267",
        ],
        help=(
            "Hex color codes for each experiment (e.g. '#0072B2'). "
            "If fewer than experiments are provided, they will be cycled."
        ),
    )
    # IDs / structures to exclude from geometric mean
    parser.add_argument(
        "--gm-exclude", nargs="+",
        default=[50, 52, 57, 60, 61, 67, 72, 78, 80, 81, 86, 87, 88, 90, 91, 92, 96, 99],
        help="IDs or target structures to exclude from geometric mean of probability",
    )

    args = parser.parse_args()

    if len(args.results) != len(args.exp_names):
        parser.error("`--results` and `--exp_names` must have the same number of entries")

    os.makedirs(args.out_dir, exist_ok=True)

    # Global cache path
    if args.cache_path is None:
        global_cache_path = os.path.join(args.out_dir, "eval_cache_global.parquet")
    else:
        global_cache_path = args.cache_path
    logger.info(f"[cache] Using global cache at: {global_cache_path}")

    # build line_styles list aligned with experiments
    if args.line_styles is None:
        line_styles = None  # all solid by default
    else:
        raw_ls = args.line_styles
        if len(raw_ls) < len(args.exp_names):
            logger.info(
                f"[line-styles] Only {len(raw_ls)} styles provided for "
                f"{len(args.exp_names)} experiments; cycling."
            )
        line_styles = [raw_ls[i % len(raw_ls)] for i in range(len(args.exp_names))]
        logger.info(f"[line-styles] Using styles per experiment: {line_styles}")

    # build marker_styles list aligned with experiments
    if args.markers is None:
        marker_styles = None
    else:
        raw_mk = args.markers
        if len(raw_mk) < len(args.exp_names):
            logger.info(
                f"[markers] Only {len(raw_mk)} markers provided for "
                f"{len(args.exp_names)} experiments; cycling."
            )
        marker_styles = [raw_mk[i % len(raw_mk)] for i in range(len(args.exp_names))]
        logger.info(f"[markers] Using markers per experiment: {marker_styles}")

    # build color_styles list aligned with experiments
    if args.colors is None:
        color_styles = None
    else:
        raw_colors = args.colors
        if len(raw_colors) < len(args.exp_names):
            logger.info(
                f"[colors] Only {len(raw_colors)} colors provided for "
                f"{len(args.exp_names)} experiments; cycling."
            )
        color_styles = [raw_colors[i % len(raw_colors)] for i in range(len(args.exp_names))]
        logger.info(f"[colors] Using colors per experiment: {color_styles}")

    # build gm_exclude_ids set
    gm_exclude_ids = set(str(x) for x in args.gm_exclude)
    if gm_exclude_ids:
        logger.info(f"[geom-mean] Excluding {len(gm_exclude_ids)} ids/structures from geometric mean: {gm_exclude_ids}")

    # ── Evaluate experiments using shared global cache ────────────────────────
    groups_list: List[Dict[str, List[Dict[str, Any]]]] = []

    for path, exp_name in zip(args.results, args.exp_names):
        safe_name = exp_name  # no mangling

        # timing log goes in out_dir, keyed by experiment name
        timing_log_path  = os.path.join(args.out_dir, f"eval_timing__{_safe_prefix(exp_name)}.jsonl")

        records = load_records(path)
        logger.info(f"Experiment '{exp_name}' ({safe_name}) with {len(records)} records")
        logger.info(f"Using global cache file: {global_cache_path}")

        valid_records = evaluate_all_with_cache(
            records=records,
            desc=f"Evaluating {os.path.basename(path)} ({exp_name})",
            exp_cache_path=global_cache_path,
            timing_log_path=timing_log_path,
            n_workers=args.n_workers,
            batch_size=10000,
        )

        groups_list.append(group_by_id(valid_records))

    # ── SOTA handling (also with the shared global cache) ─────────────────────
    sota_vals = dict(SOTA_DEFAULT)
    sota_prob_by_structure: Optional[Dict[str, float]] = None



    if args.sota:
        fmt = detect_sota_format(args.sota)
        logger.info(f"SOTA format detected: {fmt}")

        if fmt == "summary":
            sota_vals, sota_prob_by_structure = aggregate_sota_summary(
                args.sota,
                gm_exclude_ids=gm_exclude_ids,
            )

        elif fmt == "runs":
            # important: use SAME global cache path
            sota_timing_path = os.path.join(args.out_dir, "eval_timing__SOTA.jsonl")

            sota_records = load_records(args.sota)
            valid_sota_records = evaluate_all_with_cache(
                records=sota_records,
                desc="Evaluating SOTA runs",
                exp_cache_path=global_cache_path,
                timing_log_path=sota_timing_path,
                n_workers=args.n_workers,
                batch_size=10000,
            )
            sota_groups = group_by_id(valid_sota_records)
            sota_vals = compute_sota_metrics(
                sota_groups,
                gm_exclude_ids=gm_exclude_ids,
                name="SOTA",
            )
            sota_prob_by_structure = best_prob_by_structure(sota_groups)

        elif fmt == "empty":
            logger.warning("SOTA file appears empty; using defaults.")
        else:
            logger.warning("Unknown SOTA format; using defaults.")




    # ── Metrics, plots, Δ-prob vs length ──────────────────────────────────────
    all_metrics = collect_experiment_metrics(
        args.exp_names,
        groups_list,
        sota_vals=sota_vals,
        gm_exclude_ids=gm_exclude_ids,
    )

    save_metrics_json(args.out_dir, all_metrics)

    plot_best_of_n(
        groups_list,
        args.exp_names,
        sota_vals,
        args.max_n,
        args.out_dir,
        log_base=args.log_base,
        line_styles=line_styles,
        marker_styles=marker_styles,
        colors=color_styles,
    )

    # plot_turns_histogram(groups_list, args.exp_names, args.out_dir)

    if sota_prob_by_structure:
        for groups, name in zip(groups_list, args.exp_names):
            safe_name = _safe_prefix(name)
            plot_prob_diff_vs_length(groups, sota_prob_by_structure, args.out_dir, safe_name)
    else:
        logger.info("Skipping Δ-probability vs length (no SOTA per-structure baseline available).")

if __name__ == "__main__":
    main()
