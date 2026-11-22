from pathlib import Path

import pandas as pd

from executor import check_constraints, run_transform
from prompting import build_messages, extract_code, structure_report

NAME = "Few-shot CoT + execution-feedback repair"
MAX_REPAIRS = 2
SETTINGS = {"temperature": 0.0, "n": 1, "max_repairs": MAX_REPAIRS}

REPAIR_TEMPLATE = """I ran your program on the source sample and found problems:

{feedback}

Fix the program. Keep the same rules as before (general transformation, no file access, exact target column names). Briefly explain the fix under `## Plan`, then give the full corrected program under `## Code` in one ```python block."""


def diagnose(code, case_dir, target, n_src, timeout=120):
    if code is None:
        return ["your answer did not contain a ```python code block"]
    violations = check_constraints(code)
    if violations:
        return [f"constraint violation: {v}" for v in violations]
    df, status, err, _ = run_transform(code, Path(case_dir) / "training_0.csv", timeout=timeout)
    if status == "timeout":
        return [f"the program did not finish within {timeout}s; avoid row-by-row Python loops"]
    if status == "error":
        return [f"the program crashed:\n{err}"]
    _, problems = structure_report(df, target, n_src)
    return problems


def generate(case_dir, client, model):
    case_dir = Path(case_dir)
    target = pd.read_csv(case_dir / "target.csv", low_memory=False)
    n_src = sum(1 for _ in open(case_dir / "training_0.csv")) - 1

    messages = build_messages(case_dir.name, few_shot=True)
    text = client.chat(messages, temperature=0.0, n=1)["choices"][0]["content"]
    code = extract_code(text)
    history = []

    for round_no in range(1, MAX_REPAIRS + 1):
        problems = diagnose(code, case_dir, target, n_src)
        history.append({"round": round_no - 1, "problems": problems})
        if not problems:
            break
        feedback = "\n".join(f"- {p}" for p in problems)
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": REPAIR_TEMPLATE.format(feedback=feedback)}]
        text = client.chat(messages, temperature=0.0, n=1)["choices"][0]["content"]
        new_code = extract_code(text)
        if new_code is not None:
            code = new_code
    else:
        history.append({"round": MAX_REPAIRS, "problems": diagnose(code, case_dir, target, n_src)})

    client.notes["repair_history"] = history
    return code
