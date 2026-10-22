# หน้าที่: คำนวณ metrics จาก probability cache ด้วย threshold ที่ freeze จาก dev
# ใช้กับ: canonical evaluation ทุกชุด cache

"""Evaluate cached TUSZ probabilities with frozen dev thresholds.

This scorer reports global window metrics, patient-level metrics, and an
episode-based false-alarm rate.  A false alarm is a contiguous run of positive
predictions, bounded by recording seams, that overlaps no positive label.

Usage:
    python evaluation/compute_metrics.py \
        save/probs_eval_deformer_run4.npz \
        --out-prefix save/eval_deformer_run4

    python evaluation/compute_metrics.py \
        save/probs_eval_deformer_run4.npz \
        save/probs_eval_deformer_alpha4_run5.npz \
        --out-prefix save/eval_deformer_run4_vs_run5
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


# นับช่วง alarm ต่อเนื่องที่ไม่ทับ seizure และไม่เชื่อมข้าม recording seam
def count_false_alarm_episodes(preds, labels, seam):
    """Count positive prediction episodes with no positive label overlap."""
    preds = np.asarray(preds, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    seam = np.asarray(seam, dtype=bool)
    if not (len(preds) == len(labels) == len(seam)):
        raise ValueError("preds, labels, and seam must have equal length")

    false_alarms = 0
    predicted_episodes = 0
    start = None
    for i in range(len(preds) + 1):
        boundary = i == len(preds) or (i < len(seam) and seam[i])
        if start is not None and (boundary or not preds[i]):
            predicted_episodes += 1
            if not labels[start:i].any():
                false_alarms += 1
            start = None
        if i < len(preds) and preds[i] and start is None:
            start = i
    return false_alarms, predicted_episodes


def safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def compute_metrics(name, probs, labels, subject_ids, seam, threshold, segment_sec):
    preds = probs >= threshold
    positives = labels == 1
    negatives = ~positives

    tp = int(np.sum(preds & positives))
    tn = int(np.sum(~preds & negatives))
    fp = int(np.sum(preds & negatives))
    fn = int(np.sum(~preds & positives))
    false_alarms, predicted_episodes = count_false_alarm_episodes(
        preds, labels, seam
    )
    hours = len(labels) * segment_sec / 3600.0

    global_metrics = {
        "model_run": name,
        "threshold_from_dev": float(threshold),
        "n_windows": int(len(labels)),
        "duration_hours": hours,
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "seizure_f1": float(
            f1_score(labels, preds, pos_label=1, zero_division=0)
        ),
        "precision": safe_rate(tp, tp + fp),
        "sensitivity": safe_rate(tp, tp + fn),
        "specificity": safe_rate(tn, tn + fp),
        "balanced_accuracy": 0.5 * (
            safe_rate(tp, tp + fn) + safe_rate(tn, tn + fp)
        ),
        "false_alarm_episodes": false_alarms,
        "predicted_episodes": predicted_episodes,
        "fa_per_24h": safe_rate(false_alarms, hours) * 24.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    rows = []
    for subject_id in np.unique(subject_ids):
        mask = subject_ids == subject_id
        y = labels[mask]
        p = probs[mask]
        pred = preds[mask]
        subject_seam = seam[mask].copy()
        subject_seam[0] = True
        n_pos = int(np.sum(y == 1))
        subject_fa, subject_episodes = count_false_alarm_episodes(
            pred, y, subject_seam
        )
        subject_hours = len(y) * segment_sec / 3600.0
        subject_tp = int(np.sum(pred & (y == 1)))
        subject_tn = int(np.sum(~pred & (y == 0)))
        subject_fp = int(np.sum(pred & (y == 0)))
        subject_fn = int(np.sum(~pred & (y == 1)))
        rows.append({
            "model_run": name,
            "subject_id": int(subject_id),
            "n_windows": int(len(y)),
            "n_positive_windows": n_pos,
            "duration_hours": subject_hours,
            "auprc": (
                float(average_precision_score(y, p))
                if 0 < n_pos < len(y) else float("nan")
            ),
            "seizure_f1": (
                float(f1_score(y, pred, pos_label=1, zero_division=0))
                if n_pos > 0 else float("nan")
            ),
            "sensitivity": safe_rate(subject_tp, subject_tp + subject_fn),
            "specificity": safe_rate(subject_tn, subject_tn + subject_fp),
            "false_alarm_episodes": subject_fa,
            "predicted_episodes": subject_episodes,
            "fa_per_24h": safe_rate(subject_fa, subject_hours) * 24.0,
        })

    patient_summary = {}
    for metric in (
        "auprc", "seizure_f1", "sensitivity", "specificity", "fa_per_24h"
    ):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        valid = values[np.isfinite(values)]
        patient_summary[metric] = {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "median": float(np.median(valid)),
            "n": int(len(valid)),
        }
    return global_metrics, patient_summary, rows


def load_caches(path):
    cache = np.load(path, allow_pickle=False)
    required = {
        "model_names", "probs", "labels", "subject_ids", "seam", "thresholds"
    }
    missing = required - set(cache.files)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    if not bool(np.any(cache["seam"])):
        raise ValueError(f"{path} has no recording seams; FA/24h would be invalid")
    names = cache["model_names"]
    probs = cache["probs"]
    thresholds = cache["thresholds"]
    if not (len(names) == len(probs) == len(thresholds)):
        raise ValueError(f"{path} has inconsistent model/probability/threshold counts")
    return [
        {
            "name": str(names[index]),
            "probs": probs[index].astype(np.float64),
            "labels": cache["labels"].astype(np.int8),
            "subject_ids": cache["subject_ids"].astype(np.int32),
            "seam": cache["seam"].astype(bool),
            "threshold": float(thresholds[index]),
        }
        for index in range(len(names))
    ]


def paired_summary(all_rows):
    if len(all_rows) != 2:
        return None
    left = {row["subject_id"]: row for row in all_rows[0]}
    right = {row["subject_id"]: row for row in all_rows[1]}
    paired = {}
    for metric in ("auprc", "seizure_f1", "sensitivity", "specificity"):
        diffs = []
        for subject_id in sorted(set(left) & set(right)):
            a = left[subject_id][metric]
            b = right[subject_id][metric]
            if np.isfinite(a) and np.isfinite(b):
                diffs.append(b - a)
        diffs = np.asarray(diffs, dtype=float)
        result = {
            "mean_delta_second_minus_first": float(np.mean(diffs)),
            "std_delta": float(np.std(diffs)),
            "n": int(len(diffs)),
        }
        if len(diffs) > 0 and np.any(diffs != 0):
            result["wilcoxon_p"] = float(wilcoxon(diffs).pvalue)
        else:
            result["wilcoxon_p"] = float("nan")
        paired[metric] = result
    return paired


# ค่าเฉลี่ย/sample SD ข้าม runs; ไม่ใช่ population SD ข้ามผู้ป่วย
def aggregate_summary(summaries):
    """Aggregate run-level metrics as mean and sample standard deviation."""
    metrics = (
        "auroc", "auprc", "seizure_f1", "precision", "sensitivity",
        "specificity", "balanced_accuracy", "fa_per_24h",
    )
    aggregate = {"n_runs": len(summaries), "global": {}, "per_subject_mean": {}}
    for metric in metrics:
        values = np.asarray(
            [summary["global"][metric] for summary in summaries], dtype=float
        )
        aggregate["global"][metric] = {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    for metric in (
        "auprc", "seizure_f1", "sensitivity", "specificity", "fa_per_24h"
    ):
        values = np.asarray(
            [summary["per_subject"][metric]["mean"] for summary in summaries],
            dtype=float,
        )
        aggregate["per_subject_mean"][metric] = {
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return aggregate


def aggregate_per_subject(all_rows):
    """Aggregate each subject's metrics across runs with sample SD."""
    metrics = (
        "auprc", "seizure_f1", "sensitivity", "specificity", "fa_per_24h"
    )
    grouped = {}
    for rows in all_rows:
        for row in rows:
            grouped.setdefault(row["subject_id"], []).append(row)

    aggregate_rows = []
    for subject_id in sorted(grouped):
        rows = grouped[subject_id]
        reference = rows[0]
        aggregate = {
            "subject_id": subject_id,
            "n_runs": len(rows),
            "n_windows": reference["n_windows"],
            "n_positive_windows": reference["n_positive_windows"],
            "duration_hours": reference["duration_hours"],
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            valid = values[np.isfinite(values)]
            aggregate[f"{metric}_mean"] = (
                float(np.mean(valid)) if len(valid) else float("nan")
            )
            aggregate[f"{metric}_sample_std"] = (
                float(np.std(valid, ddof=1)) if len(valid) > 1
                else (0.0 if len(valid) == 1 else float("nan"))
            )
            aggregate[f"{metric}_n"] = int(len(valid))
        aggregate_rows.append(aggregate)
    return aggregate_rows


def format_mean_std(row, metric, decimals=4):
    mean = row[f"{metric}_mean"]
    sample_std = row[f"{metric}_sample_std"]
    if not (np.isfinite(mean) and np.isfinite(sample_std)):
        return "N/A"
    return f"{mean:.{decimals}f} +/- {sample_std:.{decimals}f}"


def write_per_subject_aggregate_report(path, rows, report_title):
    lines = [
        f"{report_title} — PER-SUBJECT METRICS ACROSS SEEDS",
        "=" * 119,
        "Values are mean +/- sample SD across seeds 2023, 2024, and 2025.",
        "AUPRC, F1_sz, and sensitivity are N/A for subjects with no positive window.",
        "",
        "SEIZURE-POSITIVE SUBJECTS",
        "-" * 119,
        (
            f"{'ID':>3} {'Pos':>6} {'AUPRC':>21} {'F1_sz':>21} "
            f"{'Sensitivity':>21} {'Specificity':>21} {'FA/24h':>21}"
        ),
    ]
    for row in rows:
        if row["n_positive_windows"] == 0:
            continue
        lines.append(
            f"{row['subject_id']:>3} {row['n_positive_windows']:>6} "
            f"{format_mean_std(row, 'auprc'):>21} "
            f"{format_mean_std(row, 'seizure_f1'):>21} "
            f"{format_mean_std(row, 'sensitivity'):>21} "
            f"{format_mean_std(row, 'specificity'):>21} "
            f"{format_mean_std(row, 'fa_per_24h', 2):>21}"
        )
    lines.extend([
        "",
        "SEIZURE-FREE SUBJECTS",
        "-" * 74,
        f"{'ID':>3} {'Windows':>9} {'Specificity':>25} {'FA/24h':>25}",
    ])
    for row in rows:
        if row["n_positive_windows"] != 0:
            continue
        lines.append(
            f"{row['subject_id']:>3} {row['n_windows']:>9} "
            f"{format_mean_std(row, 'specificity'):>25} "
            f"{format_mean_std(row, 'fa_per_24h', 2):>25}"
        )
    lines.extend([
        "",
        "NOTES",
        "-" * 74,
        "- SD is variability across the three random-seed runs for the same subject.",
        "- It is not variability across subjects.",
        "- Thresholds are frozen separately from each run's dev seizure-F1 optimum.",
        "- Eval labels were not used to select checkpoints or thresholds.",
        "",
    ])
    with open(path, "w") as report_file:
        report_file.write("\n".join(lines))


def write_text_report(path, summaries, aggregate, definition, report_title):
    lines = [
        f"{report_title} — CANONICAL EVALUATION REPORT",
        "=" * 96,
        f"Runs: {aggregate['n_runs']}",
        f"Threshold: {definition['threshold']}",
        f"FA/24h: {definition['fa_per_24h']}",
        f"Segment duration: {definition['segment_seconds']:.1f} seconds",
        "",
        "GLOBAL METRICS PER RUN",
        "-" * 96,
        (
            f"{'Run':<26} {'Thr':>7} {'AUROC':>8} {'AUPRC':>8} {'F1_sz':>8} "
            f"{'Prec':>8} {'Sens':>8} {'Spec':>8} {'BalAcc':>8} {'FA/24h':>9}"
        ),
    ]
    for summary in summaries:
        metric = summary["global"]
        lines.append(
            f"{metric['model_run']:<26} {metric['threshold_from_dev']:>7.4f} "
            f"{metric['auroc']:>8.4f} {metric['auprc']:>8.4f} "
            f"{metric['seizure_f1']:>8.4f} {metric['precision']:>8.4f} "
            f"{metric['sensitivity']:>8.4f} {metric['specificity']:>8.4f} "
            f"{metric['balanced_accuracy']:>8.4f} {metric['fa_per_24h']:>9.2f}"
        )
    lines.extend(["", "GLOBAL METRICS - MEAN +/- SAMPLE STD", "-" * 96])
    for metric in (
        "auroc", "auprc", "seizure_f1", "precision", "sensitivity",
        "specificity", "balanced_accuracy", "fa_per_24h",
    ):
        values = aggregate["global"][metric]
        lines.append(
            f"{metric:<20} {values['mean']:.6f} +/- {values['sample_std']:.6f}"
        )
    lines.extend(["", "PER-SUBJECT MEAN PER RUN", "-" * 96])
    lines.append(
        f"{'Run':<26} {'AUPRC':>12} {'F1_sz':>12} {'Sensitivity':>12} "
        f"{'Specificity':>12} {'FA/24h':>12}"
    )
    for summary in summaries:
        patient = summary["per_subject"]
        lines.append(
            f"{summary['global']['model_run']:<26} "
            f"{patient['auprc']['mean']:>12.4f} {patient['seizure_f1']['mean']:>12.4f} "
            f"{patient['sensitivity']['mean']:>12.4f} "
            f"{patient['specificity']['mean']:>12.4f} "
            f"{patient['fa_per_24h']['mean']:>12.2f}"
        )
    lines.extend(["", "CONFUSION MATRICES", "-" * 96])
    lines.append(f"{'Run':<26} {'TN':>10} {'FP':>10} {'FN':>10} {'TP':>10}")
    for summary in summaries:
        metric = summary["global"]
        lines.append(
            f"{metric['model_run']:<26} {metric['tn']:>10d} {metric['fp']:>10d} "
            f"{metric['fn']:>10d} {metric['tp']:>10d}"
        )
    lines.extend([
        "",
        "NOTES",
        "-" * 96,
        "- Checkpoints were selected by maximum dev AUPRC.",
        "- Each threshold was frozen from that checkpoint's dev seizure-F1 optimum.",
        "- Eval labels were not used to select checkpoints or thresholds.",
        "- AUROC/AUPRC are threshold-free; other metrics use the frozen dev threshold.",
        "",
    ])
    with open(path, "w") as report_file:
        report_file.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("caches", nargs="+", help="One cache file per model/run")
    parser.add_argument("--segment-sec", type=float, default=4.0)
    parser.add_argument("--out-prefix", default="save/eval_cached_runs")
    parser.add_argument("--text-out", default=None)
    parser.add_argument("--report-title", default="A4-E20")
    args = parser.parse_args()

    caches = [cache for path in args.caches for cache in load_caches(path)]
    reference = caches[0]
    for cache in caches[1:]:
        for field in ("labels", "subject_ids", "seam"):
            if not np.array_equal(cache[field], reference[field]):
                raise ValueError(
                    f"{cache['name']} does not align with {reference['name']} "
                    f"for field {field}"
                )

    summaries = []
    all_rows = []
    for cache in caches:
        global_metrics, patient_summary, rows = compute_metrics(
            cache["name"],
            cache["probs"],
            cache["labels"],
            cache["subject_ids"],
            cache["seam"],
            cache["threshold"],
            args.segment_sec,
        )
        summaries.append({
            "global": global_metrics,
            "per_subject": patient_summary,
        })
        all_rows.append(rows)

    definition = {
            "threshold": "frozen optimal_threshold stored in checkpoint (dev)",
            "fa_per_24h": (
                "contiguous positive prediction episodes with no positive-label "
                "overlap, split at verified recording seams, per 24 recording hours"
            ),
            "segment_seconds": args.segment_sec,
    }
    aggregate = aggregate_summary(summaries)
    per_subject_aggregate = aggregate_per_subject(all_rows)
    output = {
        "definition": definition,
        "runs": summaries,
        "aggregate": aggregate,
        "paired": paired_summary(all_rows),
        "per_subject_across_runs": per_subject_aggregate,
    }

    output_dir = os.path.dirname(args.out_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.out_prefix + "_summary.json", "w") as file:
        json.dump(output, file, indent=2, allow_nan=True)
    with open(args.out_prefix + "_per_subject.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=all_rows[0][0].keys())
        writer.writeheader()
        for rows in all_rows:
            writer.writerows(rows)
    per_subject_aggregate_path = args.out_prefix + "_per_subject_3seed_summary.csv"
    with open(per_subject_aggregate_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=per_subject_aggregate[0].keys())
        writer.writeheader()
        writer.writerows(per_subject_aggregate)
    per_subject_text_path = args.out_prefix + "_per_subject_3seed_summary.txt"
    write_per_subject_aggregate_report(
        per_subject_text_path, per_subject_aggregate, args.report_title
    )
    text_path = args.text_out or args.out_prefix + "_results.txt"
    write_text_report(text_path, summaries, aggregate, definition, args.report_title)

    for run in summaries:
        g = run["global"]
        p = run["per_subject"]
        print(f"\n{g['model_run']}  threshold(dev)={g['threshold_from_dev']:.4f}")
        print(
            f"  AUROC={g['auroc']:.4f}  AUPRC={g['auprc']:.4f}  "
            f"F1_sz={g['seizure_f1']:.4f}"
        )
        print(
            f"  Sens={g['sensitivity']:.4f}  Spec={g['specificity']:.4f}  "
            f"FA/24h={g['fa_per_24h']:.2f} ({g['false_alarm_episodes']} episodes)"
        )
        print(
            f"  Per-subj AUPRC={p['auprc']['mean']:.4f}±"
            f"{p['auprc']['std']:.4f} (n={p['auprc']['n']})  "
            f"F1_sz={p['seizure_f1']['mean']:.4f}±"
            f"{p['seizure_f1']['std']:.4f}"
        )
    print(f"\nSaved: {args.out_prefix}_summary.json")
    print(f"Saved: {args.out_prefix}_per_subject.csv")
    print(f"Saved: {per_subject_aggregate_path}")
    print(f"Saved: {per_subject_text_path}")
    print(f"Saved: {text_path}")


if __name__ == "__main__":
    main()
