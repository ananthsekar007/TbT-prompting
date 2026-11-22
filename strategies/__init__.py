from strategies import s1_zeroshot_cot, s2_fewshot_cot, s4_fewshot_cot_sc, s5_fewshot_repair

STRATEGIES = {
    "s1": s1_zeroshot_cot,
    "s2": s2_fewshot_cot,
    "s4": s4_fewshot_cot_sc,
    "s5": s5_fewshot_repair,
}
