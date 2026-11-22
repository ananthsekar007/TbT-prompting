import argparse
import json
from pathlib import Path

import pandas as pd

from prompting import eval_cases

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
ORDER = ["s1", "s2", "s4", "s5"]
SHORT = {"s1": "S1 zero-shot", "s2": "S2 few-shot", "s4": "S4 few-shot+SC", "s5": "S5 few-shot+repair"}


def load_results():
    rows = []
    for path in sorted(RESULTS.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["strategy", "case"], keep="last")
    df["ap_correct"] = df.validation.map(lambda v: v["autopipeline"]["is_correct"])
    df["hm_correct"] = df.validation.map(lambda v: v["hard_match"]["is_correct"])
    df["hm_partial"] = df.apply(lambda r: r.validation["hard_match"]["accuracy"] if r.executed else 0.0, axis=1)
    df["failure"] = df.apply(classify_failure, axis=1)
    return df


def classify_failure(r):
    if r.ap_correct and r.hm_correct:
        return "correct"
    if r.status == "no_code" and r.get("truncated_completions", 0):
        return "truncated_output"
    if r.status in ("no_code", "llm_error", "constraint_violation", "timeout"):
        return r.status
    if r.status == "error":
        return "runtime_error"
    if r.ap_correct and not r.hm_correct:
        return "column_naming_only"
    out_rows, tgt_rows = r.output_shape[0], r.target_shape[0]
    if out_rows != tgt_rows:
        return "row_count_mismatch"
    info = " ".join(r.validation["hard_match"]["info"])
    if "Column" in info and "Mismatch" in info:
        return "schema_mismatch"
    return "wrong_values"


def summarize(df, n_eval):
    rows = []
    for strategy, g in df.groupby("strategy"):
        rows.append({
            "strategy": strategy,
            "name": g.strategy_name.iloc[0],
            "model": g.model.iloc[0],
            "cases_run": len(g),
            "exact_acc_autopipeline": g.ap_correct.sum() / n_eval,
            "exact_acc_hard_match": g.hm_correct.sum() / n_eval,
            "mean_partial_hard_match": g.hm_partial.sum() / n_eval,
            "execution_rate": g.executed.sum() / n_eval,
            "llm_calls": int(g.calls.sum()),
            "prompt_tokens": int(g.prompt_tokens.sum()),
            "completion_tokens": int(g.completion_tokens.sum()),
            "llm_latency_min": round(g.llm_latency_s.sum() / 60, 1),
            "wall_clock_min": round(g.wall_s.sum() / 60, 1),
        })
    out = pd.DataFrame(rows)
    out["_o"] = out.strategy.map(lambda s: ORDER.index(s) if s in ORDER else 99)
    return out.sort_values("_o").drop(columns="_o")


def by_length(df):
    g = df.groupby(["strategy", "length"])
    out = g.agg(cases=("case", "count"), exact_acc_autopipeline=("ap_correct", "mean"),
                mean_partial_hard_match=("hm_partial", "mean")).reset_index()
    return out


def plot(summary, length_table):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return
    FIGURES.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = range(len(summary))
    w = 0.38
    ax.bar([i - w / 2 for i in x], summary.exact_acc_autopipeline * 100, w, label="autopipeline (official)")
    ax.bar([i + w / 2 for i in x], summary.exact_acc_hard_match * 100, w, label="hard_match")
    ax.set_xticks(list(x))
    ax.set_xticklabels([SHORT.get(s, s) for s in summary.strategy], fontsize=9)
    ax.set_ylabel("exact-match accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    for i, v in enumerate(summary.exact_acc_autopipeline * 100):
        ax.text(i - w / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_by_strategy.png", dpi=200)

    pivot = length_table.pivot(index="length", columns="strategy", values="exact_acc_autopipeline") * 100
    pivot = pivot[[s for s in ORDER if s in pivot.columns]].rename(columns=SHORT)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    pivot.plot.bar(ax=ax, width=0.8)
    ax.set_xlabel("pipeline length")
    ax.set_ylabel("exact-match accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_by_length.png", dpi=200)
    print(f"figures written to {FIGURES}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    n_eval = len(eval_cases())
    df = load_results()
    summary = summarize(df, n_eval)
    lengths = by_length(df)
    failures = df.groupby(["strategy", "failure"]).size().unstack(fill_value=0)

    summary.to_csv(ROOT / "summary.csv", index=False)
    lengths.to_csv(ROOT / "summary_by_length.csv", index=False)
    failures.to_csv(ROOT / "failure_modes.csv")

    per_case = df.pivot(index="case", columns="strategy", values="ap_correct")
    per_case.to_csv(ROOT / "per_case_correct.csv")

    pd.set_option("display.width", 200)
    print(f"Evaluation cases: {n_eval}\n")
    print(summary.to_string(index=False))
    print("\nBy pipeline length (autopipeline exact accuracy):")
    print(lengths.pivot(index="length", columns="strategy", values="exact_acc_autopipeline").round(2).to_string())
    print("\nBy pipeline length (mean partial credit):")
    print(lengths.pivot(index="length", columns="strategy", values="mean_partial_hard_match").round(2).to_string())
    print("\nFailure modes:")
    print(failures.to_string())
    if not args.no_plots:
        plot(summary, lengths)


if __name__ == "__main__":
    main()
