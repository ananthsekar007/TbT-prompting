# A4: Transformation-by-Target with prompting strategies

ASURITE: 1237228176. Model: `qwen38-27b` on the ASU RC gateway (`https://openai.rc.asu.edu/v1`).

## Layout

| Path | What |
|---|---|
| `prompting.py` | shared prompt pieces: system prompt, table profiles, CoT instruction, demo block, code extraction, structural checks |
| `strategies/` | `s1_zeroshot_cot.py`, `s2_fewshot_cot.py`, `s4_fewshot_cot_sc.py`, `s5_fewshot_repair.py` (bonus); each exposes `generate(case_dir, client, model) -> str` |
| `llm.py` | OpenAI-compatible client: retries with backoff on 429/503, on-disk cache, per-call logging |
| `executor.py` | runs a generated `transform.py` in a fresh subprocess (timeout, no API key in env, proxy set to a dead port) plus static constraint checks |
| `validate.py` | wraps the provided `validation/hard_match.py` + `validation/autopipeline_match.py` (both methods) |
| `run_benchmark.py` | the runner, writes `results/<strategy>.jsonl`, `generated/`, `logs/` |
| `summarize.py` | turns the JSONL files into `summary.csv`, `summary_by_length.csv`, `failure_modes.csv`, `per_case_correct.csv`, `figures/` |
| `demos/` | the 5 held-out demonstration cases, hand-written programs and plans, `fewshot_block.txt` |
| `check_harness.py` | runs the hand-written demo programs through the executor + validators (sanity check) |

## Strategies

| ID | Strategy | Calls/case | Decoding |
|---|---|---|---|
| s1 | Zero-shot CoT | 1 | temperature 0 |
| s2 | Few-shot CoT (4 demos) | 1 | temperature 0 |
| s4 | Few-shot CoT + self-consistency | 1 request with n=5 | temperature 0.7, k=5 |
| s5 | Few-shot CoT + execution-feedback repair (bonus) | 1-3 | temperature 0, up to 2 repairs |

All strategies share the same system prompt, table-profile block, CoT instruction and output format;
only the demonstrations (s2/s4/s5), sampling + voting (s4) or the repair loop (s5) are added.

**Self-consistency vote (s4).** Each of the 5 candidates is run on the source sample (`training_0.csv`, never on `test_0.csv`).
Candidates with no code block, a constraint violation, a crash or a timeout are dropped. The rest are clustered by an
order-insensitive hash of their output (columns sorted, rows sorted, floats rounded to 6 decimals). The largest cluster wins.
Tie-break: the cluster whose output best matches the target's structure (exact column set, then fraction of target columns
present, minus 0.1 per structural problem), then the lowest sample index. If every candidate fails, the first candidate
with code is returned (and scored as-is).

**Repair loop (s5).** The few-shot answer is run on `training_0.csv`; feedback is limited to crashes/timeouts, constraint
violations and structural mismatches against the target (missing/extra columns, numeric vs text dtype, value-format
pattern such as `2016-01-01` vs `01/01/2016`, all-NaN columns, wrong row count for small fixed-size targets). No
evaluation data is used. Up to 2 repair rounds.

**Demonstration split.** `length1_1`, `length2_1`, `length3_1`, `length7_2`, `length8_1` are excluded from scoring for
every strategy; the prompt uses `length1_1`, `length2_1`, `length7_2`, `length8_1`. All metrics are over the remaining 100 cases.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`.env` (not included in the zip) must contain:

```
OPENAI_API_KEY=<your RC key>
YODB_LLM_ENDPOINT=https://openai.rc.asu.edu/v1
```

The benchmark goes in `benchmarks/smartbuilding-pipelines-v2-split/`.

## Reproduce

```bash
python check_harness.py                                   # 5/5 demo programs should be correct
python run_benchmark.py --strategy s1 --cases length1_2,length4_3,length8_2   # smoke test
python run_benchmark.py --strategy s1,s2,s4,s5 --model qwen38-27b --cases all --workers 10 --llm-concurrency 6
python summarize.py
./make_submission.sh
```

Runner flags: `--timeout 120` (seconds per execution), `--max-tokens 4000`, `--request-timeout 900`, `--reasoning-effort none`,
`--workers 10` (strategy x case jobs in parallel; execution and scoring run in subprocesses), `--llm-concurrency 6`
(max requests in flight to the gateway),
`--force` (re-run cases already in the results file). The run is resumable: finished cases are skipped, and every LLM
response is cached in `cache/` keyed by the full request, so re-running reproduces the same completions without new calls.

Settings used for every reported number: model `qwen38-27b`, `max_tokens=4000`, `reasoning_effort=none`,
temperature 0 (s1, s2, s5) / 0.7 with n=5 (s4), header `x-obleth-boons: off`. Each JSONL row also records them.

## Outputs per case

- `results/<strategy>.jsonl`: case, status, both validator results (accuracy, is_correct, per-column similarities), tokens, latency, log paths
- `generated/<strategy>/<case>/transform.py` (+ `candidates/` for s4)
- `logs/<strategy>/<case>/call_XX_prompt.json`, `call_XX_completion.json`, `validation.json`
