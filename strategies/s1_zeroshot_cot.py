from pathlib import Path

from prompting import build_messages, extract_code

NAME = "Zero-shot CoT"
SETTINGS = {"temperature": 0.0, "n": 1}


def generate(case_dir, client, model):
    case = Path(case_dir).name
    messages = build_messages(case, few_shot=False)
    result = client.chat(messages, temperature=0.0, n=1)
    return extract_code(result["choices"][0]["content"])
