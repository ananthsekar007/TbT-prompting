import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT / "benchmarks" / "smartbuilding-pipelines-v2-split"
DEMO_DIR = ROOT / "demos"

DEMO_CASES = ["length1_1", "length2_1", "length3_1", "length7_2", "length8_1"]
PROMPT_DEMOS = ["length1_1", "length2_1", "length7_2", "length8_1"]

SYSTEM_PROMPT = """You are a senior data engineer. You write Python/pandas programs that transform a source table into a target table.

You are shown a SAMPLE of the source table and the target table. Your program will later be run on a DIFFERENT chunk of the same source data (same schema, different rows), and its output must reproduce the target table exactly. So write the general transformation, never something tied to the sample rows.

Important: the target was produced from that other chunk, not from the sample. The sample and the target can cover different dates, time periods, or entities, so their rows will not line up and their values will not match one-to-one. Do NOT shift dates, fill in missing dates, or try to map sample rows onto target rows. Infer the operators from the schemas, formats and value patterns, and apply them to whatever rows the source contains.

Rules for the program:
- Define `def transform(source: pd.DataFrame) -> pd.DataFrame` and only use pandas, numpy and the Python standard library.
- Compute everything from `source`. Do not read or write any file, do not use the network, do not call exec/eval.
- Never hard-code target data values or row positions. Column names, constants, unit factors and date formats are fine.
- The returned DataFrame must have exactly the target column names (copy them verbatim, including case and spaces) and only those columns. End with `return out[TARGET_COLUMNS]`.
- Match the target's value formats exactly, especially dates/timestamps (use an explicit strftime format) and text padding/case.
- Be robust to messy input: column names may differ only by case/whitespace, numbers may be stored as strings, cells may be NaN, timestamps may come in several formats.
- The number of rows in the sample can differ from the evaluation source. Row-preserving transforms should keep one output row per input row; fixed-size aggregations (e.g. 24 hours x 12 months) should produce the same shape as the target."""

COT_INSTRUCTION = """Think step by step before writing code. Under a heading `## Plan`, write a short numbered list (at most 8 lines, one sentence each) that infers the operator sequence:
1. row granularity: is the target row-preserving, filtered, or aggregated (group-by/pivot/unpivot)? Use timestamp granularity and key columns;
2. how each target column is produced from source columns (rename, arithmetic/unit conversion, date reformatting, constants, derived fields);
3. any aggregation function, filter, sort or reshape, and the evidence for it.
Decide and move on; do not deliberate at length. Then under a heading `## Code`, give the complete program in a single ```python fenced block. Keep code comments to a few words; do not reason inside the code."""

FORMAT_REMINDER = "Answer with `## Plan` followed by `## Code` containing exactly one ```python block that defines transform(source)."


def case_dir(case):
    return BENCHMARK_DIR / case


def case_length(case):
    return int(re.match(r"length(\d+)_", case).group(1))


def list_cases():
    cases = [p.name for p in BENCHMARK_DIR.iterdir() if p.is_dir() and p.name.startswith("length")]
    return sorted(cases, key=lambda c: (case_length(c), int(c.split("_")[1])))


def eval_cases():
    return [c for c in list_cases() if c not in DEMO_CASES]


def _fmt_value(v, width=40):
    s = repr(v) if isinstance(v, str) else str(v)
    return s if len(s) <= width else s[: width - 3] + "..."


def _looks_like_time(series):
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    return sample.str.contains(r"\d{1,4}[/-]\d{1,2}|\d{1,2}:\d{2}").mean() > 0.8


def profile_table(df, name, n_rows_total, max_sample_rows=12, max_cols_detail=60):
    lines = [f"### {name}", f"rows: {n_rows_total}, columns: {df.shape[1]}"]
    lines.append(f"exact column names: {list(df.columns)!r}")
    lines.append("column details:")
    for i, col in enumerate(df.columns):
        if i >= max_cols_detail:
            lines.append(f"  ... {df.shape[1] - max_cols_detail} more columns omitted")
            break
        s = df[col]
        info = f"  {col!r}: dtype={s.dtype}, nulls={int(s.isna().sum())}, distinct={s.nunique(dropna=True)}"
        nonnull = s.dropna()
        if s.nunique(dropna=True) <= 12 and len(nonnull):
            info += ", values=" + "[" + ", ".join(_fmt_value(v, 30) for v in sorted(nonnull.unique(), key=str)) + "]"
        elif pd.api.types.is_numeric_dtype(s) and len(nonnull):
            info += f", min={nonnull.min():.6g}, max={nonnull.max():.6g}"
        elif _looks_like_time(s):
            info += f", first={_fmt_value(nonnull.iloc[0])}, last={_fmt_value(nonnull.iloc[-1])}"
        elif len(nonnull):
            info += ", e.g. " + ", ".join(_fmt_value(v, 30) for v in nonnull.unique()[:3])
        lines.append(info)

    lines.append(f"sample rows (CSV, {min(len(df), max_sample_rows)} rows):")
    lines.append(sample_rows(df, max_sample_rows).to_csv(index=False).strip())
    return "\n".join(lines)


def sample_rows(df, k):
    """Head, a few evenly spaced middle rows and the tail, so group boundaries are visible."""
    if len(df) <= k:
        return df
    head = max(k // 2, 1)
    tail = max(k // 4, 1)
    mid = k - head - tail
    idx = list(range(head))
    idx += [int(x) for x in pd.Series(range(head, len(df) - tail)).sample(mid, random_state=0).sort_values()]
    idx += list(range(len(df) - tail, len(df)))
    return df.iloc[idx]


def load_case(case):
    d = case_dir(case)
    source = pd.read_csv(d / "training_0.csv", low_memory=False)
    target = pd.read_csv(d / "target.csv", low_memory=False)
    return source, target


def case_block(case, max_sample_rows=12):
    source, target = load_case(case)
    wide = source.shape[1] > 40
    src = profile_table(source, "Source table (sample: training chunk)", len(source),
                        max_sample_rows=6 if wide else max_sample_rows)
    tgt = profile_table(target, "Target table", len(target), max_sample_rows=max_sample_rows)
    return f"{src}\n\n{tgt}\n\nTARGET_COLUMNS = {list(target.columns)!r}"


@lru_cache(maxsize=1)
def demo_block():
    parts = ["Here are worked examples from other cases of the same benchmark.\n"]
    for i, case in enumerate(PROMPT_DEMOS, start=1):
        reasoning = (DEMO_DIR / case / "reasoning.md").read_text().strip()
        code = (DEMO_DIR / case / "transform.py").read_text().strip()
        parts.append(f"=== Example {i} ===\n{case_block(case, max_sample_rows=5)}\n\n"
                     f"## Plan\n{reasoning}\n\n## Code\n```python\n{code}\n```\n")
    parts.append("=== End of examples ===\n")
    return "\n".join(parts)


def build_messages(case, few_shot=False):
    user = ""
    if few_shot:
        user += demo_block() + "\n"
    user += "Now solve this case.\n\n" + case_block(case) + "\n\n" + COT_INSTRUCTION + "\n\n" + FORMAT_REMINDER
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def extract_code(text):
    if not text:
        return None
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        return None
    code = blocks[-1].strip() + "\n"
    if "import pandas" not in code:
        code = "import pandas as pd\nimport numpy as np\n\n" + code
    return code


def _shape(v):
    s = re.sub(r"[0-9]", "9", str(v))
    s = re.sub(r"[A-Z]", "A", s)
    return re.sub(r"[a-z]", "a", s)


def structure_report(out, target, sample_rows_in):
    """Compare an output computed on the sample against the target's structure.
    Returns (score, list of problems). Only looks at schema, formats and shape."""
    problems = []
    tcols = list(target.columns)
    ocols = [str(c) for c in out.columns]
    missing = [c for c in tcols if c not in ocols]
    extra = [c for c in ocols if c not in tcols]
    if missing:
        problems.append(f"missing target columns: {missing!r}")
    if extra:
        problems.append(f"unexpected extra columns: {extra!r}")

    n_out, n_tgt = len(out), len(target)
    if n_tgt <= 400 and n_out != n_tgt:
        problems.append(f"output has {n_out} rows but the target has {n_tgt}; a fixed-size aggregation should match exactly")
    elif n_tgt > 400 and abs(n_out - n_tgt) > 0.2 * n_tgt:
        problems.append(f"output has {n_out} rows from a {sample_rows_in}-row sample, target has {n_tgt} rows "
                        f"from an evaluation chunk of similar size; check filtering/grouping")

    for c in tcols:
        if c not in ocols:
            continue
        t, o = target[c], out[c]
        t_num = pd.api.types.is_numeric_dtype(t)
        o_num = pd.api.types.is_numeric_dtype(o)
        if t_num and not o_num:
            problems.append(f"column {c!r} should be numeric but is {o.dtype} (e.g. {_fmt_value(o.dropna().iloc[0]) if o.notna().any() else 'all NaN'})")
        elif not t_num and t.notna().any() and o.notna().any():
            t_shapes = set(t.dropna().astype(str).head(200).map(_shape))
            o_vals = o.dropna().astype(str).head(200)
            bad = o_vals[~o_vals.map(_shape).isin(t_shapes)]
            if len(bad) > 0.5 * len(o_vals):
                problems.append(f"column {c!r} format differs: output value {bad.iloc[0]!r} vs target value {t.dropna().astype(str).iloc[0]!r}")
        if o.isna().all() and t.notna().any():
            problems.append(f"column {c!r} is entirely NaN in the output")

    score = (1.0 if not missing and not extra else 0.0) + (len(tcols) - len(missing)) / max(len(tcols), 1) - 0.1 * len(problems)
    return score, problems
