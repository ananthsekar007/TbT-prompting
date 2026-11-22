"""Sanity check for the runner: executes the hand-written demo programs on test_0.csv
and validates them with both methods. All should be correct before any LLM call."""
import pandas as pd

from executor import check_constraints, run_transform
from prompting import DEMO_CASES, DEMO_DIR, case_dir
from validate import score

for case in DEMO_CASES:
    code = (DEMO_DIR / case / "transform.py").read_text()
    df, status, err, secs = run_transform(code, case_dir(case) / "test_0.csv")
    if df is None:
        print(f"{case}: {status} {err}")
        continue
    r = score(df, pd.read_csv(case_dir(case) / "target.csv"))
    print(f"{case}: autopipeline={r['autopipeline']['is_correct']} hard_match={r['hard_match']['is_correct']} "
          f"violations={check_constraints(code)} ({secs:.1f}s)")
