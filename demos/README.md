# Demonstration split

Held out from evaluation (never scored): `length1_1`, `length2_1`, `length3_1`, `length7_2`, `length8_1`.

The few-shot prompt (S2, S4, S5) uses four of them: `length1_1`, `length2_1`, `length7_2`, `length8_1`.
`length3_1` is held out but not shown, because it has the same source as `length1_1` and would add little.

Each folder has a hand-written `transform.py` (verified correct on `test_0.csv` with both validators by
`check_harness.py`) and, for the prompt demos, the `reasoning.md` plan shown to the model.
`fewshot_block.txt` is the exact demonstration text inserted into the prompt.
