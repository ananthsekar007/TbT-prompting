import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

DRIVER = """
import pickle, sys
import pandas as pd
sys.path.insert(0, ".")
from transform import transform

source = pd.read_csv("source.csv")
out = transform(source)
if not isinstance(out, pd.DataFrame):
    raise TypeError(f"transform returned {type(out).__name__}, not a DataFrame")
with open("result.pkl", "wb") as f:
    pickle.dump(out, f)
"""

FORBIDDEN = [
    (r"\bopen\s*\(", "opens a file"),
    (r"read_(csv|excel|json|parquet|pickle|table|sql)\s*\(", "reads a file"),
    (r"to_(csv|excel|json|parquet|pickle)\s*\(", "writes a file"),
    (r"(target|test_0|training_0)\.csv", "references a benchmark file"),
    (r"\bimport\s+(requests|urllib|socket|http|subprocess|openai)\b", "network/subprocess import"),
    (r"\bfrom\s+(requests|urllib|socket|http|subprocess|openai)\b", "network/subprocess import"),
    (r"\b(exec|eval)\s*\(", "exec/eval"),
    (r"os\.system|os\.popen", "shell call"),
]


def check_constraints(code):
    problems = [msg for pat, msg in FORBIDDEN if re.search(pat, code)]
    if not re.search(r"def\s+transform\s*\(", code):
        problems.append("no transform() function")
    return problems


def short_traceback(stderr):
    """Keep the frames inside transform.py and the final exception line."""
    lines = stderr.strip().splitlines()
    keep = []
    for i, line in enumerate(lines):
        if 'File "' in line and "transform.py" in line:
            keep.extend(lines[i:i + 2])
    tail = [l for l in lines[-3:] if l not in keep]
    return "\n".join(keep + tail) if keep else "\n".join(lines[-8:])


def run_transform(code, source_csv, timeout=120):
    """Run code on source_csv in a fresh process. Returns (df, status, error, seconds)."""
    with tempfile.TemporaryDirectory(prefix="tbt_") as tmp:
        tmp = Path(tmp)
        (tmp / "transform.py").write_text(code)
        (tmp / "driver.py").write_text(DRIVER)
        try:
            os.symlink(Path(source_csv).resolve(), tmp / "source.csv")
        except OSError:
            import shutil
            shutil.copy(source_csv, tmp / "source.csv")

        env = {k: v for k, v in os.environ.items() if "KEY" not in k.upper()}
        env.update(HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
                   NO_PROXY="", PYTHONDONTWRITEBYTECODE="1")
        start = time.time()
        try:
            proc = subprocess.run([sys.executable, "-I", "driver.py"], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "timeout", f"exceeded {timeout}s", time.time() - start
        elapsed = time.time() - start

        if proc.returncode != 0:
            return None, "error", short_traceback(proc.stderr), elapsed
        with open(tmp / "result.pkl", "rb") as f:
            df = pickle.load(f)
        return df, "ok", "", elapsed


def frame_signature(df):
    """Order-insensitive hash of a DataFrame, used for self-consistency voting."""
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    d = d[sorted(d.columns)]
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(6)
    d = d.astype(str)
    d = d.sort_values(list(d.columns)).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(d, index=False)
    header = "|".join(d.columns)
    return f"{header}#{len(d)}#{int(row_hashes.sum()) & 0xFFFFFFFFFFFF:x}"
