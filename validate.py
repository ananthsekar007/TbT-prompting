import contextlib
import io
import sys
import threading
import types
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)  # invalid escape in the provided util/utils.py


def _stub_missing(name, **attrs):
    try:
        __import__(name)
    except ImportError:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod


# util/utils.py pulls in DB and sklearn helpers we never call; only are_elements_equal is used
_stub_missing("padding_match", pad_comp=None)
_stub_missing("psycopg2")
_stub_missing("sklearn")
_stub_missing("sklearn.metrics")
_stub_missing("sklearn.metrics.pairwise", cosine_similarity=None)
_stub_missing("sklearn.feature_extraction")
_stub_missing("sklearn.feature_extraction.text", CountVectorizer=None)
_stub_missing("Levenshtein", distance=lambda a, b: 0)

from validation.hard_match import compare_lists_matching, compare_tables_matching  # noqa: E402


def validate_autopipeline_style(sql_result_df, target_df, method="hard_match"):
    if method == "hard_match":
        target_cols_by_lower = {c.lower(): c for c in target_df.columns}
        rename = {c: target_cols_by_lower[c.lower()]
                  for c in sql_result_df.columns
                  if c not in target_df.columns and c.lower() in target_cols_by_lower}
        sql_result_df = sql_result_df.rename(columns=rename)
        return compare_lists_matching(sql_result_df, target_df)
    elif method == "autopipeline":
        return compare_tables_matching(sql_result_df, target_df)
    raise ValueError(f"Unknown validation method: {method!r}")


_lock = threading.Lock()  # validators print a lot; redirect_stdout is process-wide


def score(result_df, target_df):
    with _lock:
        return _score(result_df, target_df)


def _score(result_df, target_df):
    out = {}
    for method in ("autopipeline", "hard_match"):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                acc, ok, sims, info = validate_autopipeline_style(result_df.copy(), target_df.copy(), method)
            out[method] = {"accuracy": float(acc), "is_correct": bool(ok),
                           "similarities": [float(s) for s in sims],
                           "info": [str(x) for x in info][:50]}
        except Exception as e:
            out[method] = {"accuracy": 0.0, "is_correct": False, "similarities": [],
                           "info": [f"validator error: {type(e).__name__}: {e}"]}
        out[method]["log_tail"] = buf.getvalue()[-2000:]
    return out


def score_in_subprocess(result_df, target_csv, timeout=1800):
    """Score in a separate process so several cases can be validated in parallel."""
    import json
    import pickle
    import subprocess
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="tbt_val_") as tmp:
        pkl = Path(tmp) / "result.pkl"
        out_json = Path(tmp) / "score.json"
        with open(pkl, "wb") as f:
            pickle.dump(result_df, f)
        try:
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(pkl), str(target_csv), str(out_json)],
                                  cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=timeout)
            if proc.returncode == 0 and out_json.exists():
                return json.loads(out_json.read_text())
            err = proc.stderr.strip().splitlines()[-3:]
        except subprocess.TimeoutExpired:
            err = [f"validation exceeded {timeout}s"]
    return {m: {"accuracy": 0.0, "is_correct": False, "similarities": [],
                "info": ["validator crashed: " + " | ".join(err)]} for m in ("autopipeline", "hard_match")}


if __name__ == "__main__":
    import json
    import pickle

    import pandas as pd

    with open(sys.argv[1], "rb") as f:
        result = pickle.load(f)
    target = pd.read_csv(sys.argv[2], low_memory=False)
    with open(sys.argv[3], "w") as f:
        json.dump(_score(result, target), f)
