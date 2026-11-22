from collections import defaultdict
from pathlib import Path

import pandas as pd

from executor import check_constraints, frame_signature, run_transform
from prompting import build_messages, extract_code, structure_report

NAME = "Few-shot CoT + self-consistency"
K = 5
TEMPERATURE = 0.7
SETTINGS = {"temperature": TEMPERATURE, "n": K}


def sample_candidates(messages, client, k=K, temperature=TEMPERATURE):
    texts = [c["content"] for c in client.chat(messages, temperature=temperature, n=k)["choices"]]
    extra = 0
    while len(texts) < k:  # some backends ignore n
        texts.append(client.chat(messages, temperature=temperature, n=1, tag=f"extra-{extra}")["choices"][0]["content"])
        extra += 1
    return texts[:k]


def vote(codes, case_dir, timeout=120):
    """Run every candidate on the source sample, cluster by output hash, return the
    winner of the largest cluster. Ties: best structural match to the target, then
    the lowest sample index."""
    case_dir = Path(case_dir)
    target = pd.read_csv(case_dir / "target.csv", low_memory=False)
    n_src = sum(1 for _ in open(case_dir / "training_0.csv")) - 1

    candidates = []
    clusters = defaultdict(list)
    for i, code in enumerate(codes):
        info = {"index": i, "has_code": code is not None}
        if code is None:
            info["status"] = "no_code"
        elif check_constraints(code):
            info["status"] = "constraint_violation"
            info["error"] = check_constraints(code)
        else:
            df, status, err, secs = run_transform(code, case_dir / "training_0.csv", timeout=timeout)
            info.update(status=status, error=err, seconds=round(secs, 2))
            if df is not None:
                sig = frame_signature(df)
                info["signature"] = sig
                info["structure_score"], _ = structure_report(df, target, n_src)
                clusters[sig].append(i)
        candidates.append(info)

    if not clusters:
        fallback = next((i for i, c in enumerate(codes) if c is not None), None)
        return fallback, candidates, {}

    def rank(item):
        sig, members = item
        return (len(members), candidates[members[0]]["structure_score"], -members[0])

    best_sig, members = max(clusters.items(), key=rank)
    sizes = {sig[-12:]: len(m) for sig, m in clusters.items()}
    return members[0], candidates, sizes


def generate(case_dir, client, model):
    case = Path(case_dir).name
    messages = build_messages(case, few_shot=True)
    texts = sample_candidates(messages, client)
    codes = [extract_code(t) for t in texts]
    chosen, candidates, sizes = vote(codes, case_dir)

    client.notes["candidates"] = codes
    client.notes["vote"] = {"chosen": chosen, "candidates": candidates, "cluster_sizes": sizes}
    return codes[chosen] if chosen is not None else None
