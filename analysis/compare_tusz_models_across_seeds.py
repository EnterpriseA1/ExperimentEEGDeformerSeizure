"""Paired seed-level tests for the final TUSZ CBCE ten-seed results.

Reference is Deformer. Delta = Deformer minus comparator, so positive favors
Deformer. This is an exploratory/post-hoc analysis. Holm correction is applied
separately to the five architecture comparisons within each metric and test.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

try:
    from .compare_experiment_auprc import holm
except ImportError:  # Direct script execution.
    from compare_experiment_auprc import holm


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "save_cbce/tusz_e10_seed2032_beta99999_20261007"
OUT = ROOT / "reports/tusz_seed_tests_20261017"
MODELS = ["Deformer", "Deformer_A4", "EEGNet", "Conformer", "TSception", "EEGViT"]
COMPARATORS = MODELS[1:]
METRICS = ["accuracy", "auroc", "auprc", "seizure_f1"]
DISPLAY = {
    "Deformer": "EEG-Deformer",
    "Deformer_A4": "EEG-Deformer (A4 tuned)",
    "EEGNet": "EEGNet",
    "Conformer": "EEG-Conformer",
    "TSception": "TSception",
    "EEGViT": "EEG-ViT",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(model):
    path = SOURCE / model / "ten_seed_summary.json"
    data = json.loads(path.read_text())
    rows = {}
    for item in data["runs"]:
        g = item["global"]
        match = re.fullmatch(re.escape(model) + r"_seed(\d+)/run_1", g["model_run"])
        if match is None:
            raise ValueError(f"Unexpected run identity in {path}: {g['model_run']}")
        seed = int(match.group(1))
        if seed in rows:
            raise ValueError(f"Duplicate seed {seed}: {path}")
        total = int(g["tn"] + g["fp"] + g["fn"] + g["tp"])
        if total != int(g["n_windows"]):
            raise ValueError(f"Confusion matrix mismatch: {model} seed{seed}")
        rows[seed] = {
            "accuracy": (g["tn"] + g["tp"]) / total,
            "auroc": float(g["auroc"]),
            "auprc": float(g["auprc"]),
            "seizure_f1": float(g["seizure_f1"]),
            "n_windows": total,
            "n_positive": int(g["fn"] + g["tp"]),
        }
    expected = set(range(2023, 2033))
    if set(rows) != expected:
        raise ValueError(f"Incomplete seed set for {model}: {sorted(rows)}")
    return path, rows


def paired_test(reference, comparator):
    difference = reference - comparator
    if not np.isfinite(difference).all() or np.std(difference, ddof=1) == 0:
        raise ValueError("Invalid or degenerate paired differences")
    t_result = stats.ttest_rel(reference, comparator, alternative="two-sided")
    ci = t_result.confidence_interval(0.95)
    w_result = stats.wilcoxon(
        np.round(difference, 12), alternative="two-sided", zero_method="wilcox", method="auto"
    )
    return {
        "n": len(difference),
        "mean_reference": float(reference.mean()),
        "mean_comparator": float(comparator.mean()),
        "delta": float(difference.mean()),
        "ci95_low": float(ci.low),
        "ci95_high": float(ci.high),
        "t": float(t_result.statistic),
        "df": len(difference) - 1,
        "t_p": float(t_result.pvalue),
        "paired_dz": float(difference.mean() / difference.std(ddof=1)),
        "wilcoxon_statistic": float(w_result.statistic),
        "wilcoxon_p": float(w_result.pvalue),
        "reference_wins": int((difference > 0).sum()),
        "comparator_wins": int((difference < 0).sum()),
        "ties": int((difference == 0).sum()),
        "shapiro_p": float(stats.shapiro(difference).pvalue),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        raise ValueError(f"Output exists; preserve it and choose a new directory: {OUT}")
    OUT.mkdir(parents=True)
    plan = {
        "created": "2026-10-17",
        "analysis_type": "exploratory/post-hoc",
        "unit": "training seed paired by numeric seed ID",
        "seeds": list(range(2023, 2033)),
        "reference": "Deformer",
        "comparators": COMPARATORS,
        "metrics": METRICS,
        "tests": ["two-sided paired t-test", "two-sided Wilcoxon signed-rank"],
        "correction": "Holm separately over five architecture comparisons within each metric and test",
        "alpha": 0.05,
        "delta_direction": "Deformer minus comparator; positive favors Deformer",
        "confidence_interval": "pointwise unadjusted 95% CI for paired mean difference",
    }
    (OUT / "analysis_plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    values = {}
    audit = {}
    for model in MODELS:
        path, rows = load_model(model)
        values[model] = rows
        audit[model] = {"path": str(path), "sha256": sha256(path), "n_seeds": len(rows)}
    for seed in range(2023, 2033):
        populations = {(values[m][seed]["n_windows"], values[m][seed]["n_positive"]) for m in MODELS}
        if len(populations) != 1:
            raise ValueError(f"Eval population differs for seed {seed}: {populations}")

    rows = []
    for metric in METRICS:
        reference = np.array([values["Deformer"][s][metric] for s in range(2023, 2033)])
        metric_rows = []
        for model in COMPARATORS:
            comparator = np.array([values[model][s][metric] for s in range(2023, 2033)])
            result = {
                "metric": metric,
                "reference": DISPLAY["Deformer"],
                "comparator": DISPLAY[model],
            }
            result.update(paired_test(reference, comparator))
            metric_rows.append(result)
        for row, adjusted in zip(metric_rows, holm([r["t_p"] for r in metric_rows])):
            row["t_p_holm"] = float(adjusted)
        for row, adjusted in zip(metric_rows, holm([r["wilcoxon_p"] for r in metric_rows])):
            row["wilcoxon_p_holm"] = float(adjusted)
        for row in metric_rows:
            row["t_significant_holm"] = row["t_p_holm"] < 0.05
            row["wilcoxon_significant_holm"] = row["wilcoxon_p_holm"] < 0.05
        rows.extend(metric_rows)

    write_csv(OUT / "results.csv", rows)
    (OUT / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (OUT / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "# TUSZ CBCE ten-seed paired tests — 2026-10-17",
        "",
        "Exploratory/post-hoc. Delta is EEG-Deformer minus comparator; positive favors EEG-Deformer.",
        "Holm correction is separate across five comparisons within each metric and test.",
        "",
        "| Metric | Comparator | Delta | 95% CI | paired t p | t Holm p | Wilcoxon p | Wilcoxon Holm p |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        scale = 100.0
        lines.append(
            f"| {row['metric']} | {row['comparator']} | {row['delta']*scale:+.3f} pp | "
            f"[{row['ci95_low']*scale:+.3f}, {row['ci95_high']*scale:+.3f}] | "
            f"{row['t_p']:.6g} | {row['t_p_holm']:.6g} | "
            f"{row['wilcoxon_p']:.6g} | {row['wilcoxon_p_holm']:.6g} |"
        )
    lines += [
        "",
        "Notes:",
        "- The same numeric seed is paired across architectures, but this does not guarantee common random numbers.",
        "- All models use one fixed held-out TUSZ cohort; inference is about training-seed variability on that cohort.",
        "- Wilcoxon is a sensitivity analysis for a symmetric location shift, not a replacement test selected by normality.",
        "- Failure to reject is not evidence of equivalence.",
    ]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print((OUT / "results.md").read_text())


if __name__ == "__main__":
    main()
