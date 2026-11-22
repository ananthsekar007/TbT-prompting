import argparse
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from executor import check_constraints, run_transform
from prompting import case_dir, case_length, eval_cases, list_cases
from strategies import STRATEGIES
from validate import score_in_subprocess

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
GENERATED = ROOT / "generated"
LOGS = ROOT / "logs"

write_lock = threading.Lock()


def done_cases(strategy):
    path = RESULTS / f"{strategy}.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line)["case"] for line in path.read_text().splitlines() if line.strip()}


def save_logs(strategy, case, client):
    log_dir = LOGS / strategy / case
    log_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, rec in enumerate(client.records):
        prompt_path = log_dir / f"call_{i:02d}_prompt.json"
        completion_path = log_dir / f"call_{i:02d}_completion.json"
        prompt_path.write_text(json.dumps({"messages": rec["messages"], "params": rec["params"], "tag": rec["tag"]}, indent=1))
        completion_path.write_text(json.dumps({k: rec[k] for k in
                                               ("choices", "usage", "latency_s", "response_model",
                                                "reasoning_effort_header", "cached")}, indent=1))
        paths.append({"prompt": str(prompt_path.relative_to(ROOT)), "completion": str(completion_path.relative_to(ROOT))})
    return log_dir, paths


def run_case(strategy, case, client, timeout):
    module = STRATEGIES[strategy]
    client.reset()
    t0 = time.time()
    row = {"case": case, "length": case_length(case), "strategy": strategy, "strategy_name": module.NAME,
           "model": client.model, "settings": {**module.SETTINGS, "max_tokens": client.max_tokens,
                                               "reasoning_effort": client.reasoning_effort}}

    try:
        code = module.generate(case_dir(case), client, client.model)
        row["status"] = "generated" if code else "no_code"
    except Exception as e:
        code = None
        row["status"] = "llm_error"
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    gen_dir = GENERATED / strategy / case
    gen_dir.mkdir(parents=True, exist_ok=True)
    if code:
        (gen_dir / "transform.py").write_text(code)
    for i, cand in enumerate(client.notes.get("candidates", [])):
        (gen_dir / "candidates").mkdir(exist_ok=True)
        (gen_dir / "candidates" / f"candidate_{i}.py").write_text(cand or "# no code block in completion\n")

    log_dir, call_paths = save_logs(strategy, case, client)
    row["logs"] = call_paths
    row["notes"] = {k: v for k, v in client.notes.items() if k != "candidates"}

    result_df = None
    if code:
        violations = check_constraints(code)
        if violations:
            row["status"] = "constraint_violation"
            row["error"] = "; ".join(violations)
        else:
            result_df, status, err, secs = run_transform(code, case_dir(case) / "test_0.csv", timeout=timeout)
            row["status"] = status
            row["error"] = err
            row["exec_seconds"] = round(secs, 2)

    row["executed"] = result_df is not None
    if result_df is not None:
        target_csv = case_dir(case) / "target.csv"
        row["output_shape"] = list(result_df.shape)
        row["target_shape"] = [sum(1 for _ in open(target_csv)) - 1, len(pd.read_csv(target_csv, nrows=0).columns)]
        row["validation"] = score_in_subprocess(result_df, target_csv)
    else:
        row["validation"] = {m: {"accuracy": 0.0, "is_correct": False, "similarities": [], "info": [row["status"]]}
                             for m in ("autopipeline", "hard_match")}

    (log_dir / "validation.json").write_text(json.dumps({"status": row["status"], "error": row.get("error"),
                                                         "validation": row["validation"]}, indent=1))

    row["calls"] = len(client.records)
    row["prompt_tokens"] = sum(r["usage"]["prompt_tokens"] for r in client.records)
    row["completion_tokens"] = sum(r["usage"]["completion_tokens"] for r in client.records)
    row["llm_latency_s"] = round(sum(r["latency_s"] for r in client.records), 2)
    row["cached_calls"] = sum(1 for r in client.records if r["cached"])
    row["truncated_completions"] = sum(1 for r in client.records for ch in r["choices"] if ch.get("finish_reason") == "length")
    row["wall_s"] = round(time.time() - t0, 2)
    row["transform_path"] = str((gen_dir / "transform.py").relative_to(ROOT)) if code else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, help="comma list, e.g. s1,s2,s4,s5")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--cases", default="all", help="'all' (100 eval cases) or comma list")
    ap.add_argument("--workers", type=int, default=10, help="cases processed in parallel (execution + scoring)")
    ap.add_argument("--llm-concurrency", type=int, default=6, help="max LLM requests in flight at once")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--request-timeout", type=int, default=900, help="seconds before an LLM request is abandoned and retried")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--force", action="store_true", help="re-run cases already in results/")
    args = ap.parse_args()

    from llm import LLMClient
    client = LLMClient(args.model, max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
                       max_concurrent=args.llm_concurrency, request_timeout=args.request_timeout)

    cases = eval_cases() if args.cases == "all" else [c.strip() for c in args.cases.split(",")]
    unknown = set(cases) - set(list_cases())
    if unknown:
        raise SystemExit(f"unknown cases: {sorted(unknown)}")

    RESULTS.mkdir(exist_ok=True)
    strategies = args.strategy.split(",")
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise SystemExit(f"unknown strategy {strategy}; choose from {list(STRATEGIES)}")
        out_path = RESULTS / f"{strategy}.jsonl"
        if args.force and out_path.exists():
            keep = [l for l in out_path.read_text().splitlines() if l.strip() and json.loads(l)["case"] not in cases]
            out_path.write_text("".join(l + "\n" for l in keep))

    # interleave strategies so cheap s1/s2 jobs keep flowing while s4/s5 wait on the LLM
    jobs = []
    for case in cases:
        for strategy in strategies:
            if case not in done_cases(strategy):
                jobs.append((strategy, case))
    print(f"== {len(jobs)} jobs ({', '.join(strategies)}), {args.workers} workers, "
          f"{args.llm_concurrency} concurrent LLM requests", flush=True)

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, s, c, client, args.timeout): (s, c) for s, c in jobs}
        for n, fut in enumerate(as_completed(futures), start=1):
            strategy, case = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                traceback.print_exc()
                print(f"  [{n}/{len(jobs)}] {strategy} {case}: runner crash {e}", flush=True)
                continue
            with write_lock, open(RESULTS / f"{strategy}.jsonl", "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            v = row["validation"]
            elapsed = (time.time() - start) / 60
            print(f"  [{n}/{len(jobs)} {elapsed:5.1f}m] {strategy} {case:<11} {row['status']:<20} "
                  f"ap={v['autopipeline']['is_correct']!s:<5} hm={v['hard_match']['is_correct']!s:<5} "
                  f"partial={v['hard_match']['accuracy']:.2f} ({row['wall_s']:.0f}s)", flush=True)
    print(f"== finished in {(time.time() - start) / 60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
