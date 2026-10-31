"""Quick checks for the answer extractors. Runs on CPU in a second:
    python test_common.py
If anything here is wrong, every accuracy number you report is wrong."""

from common import (extract_gsm8k, gsm8k_correct, gsm8k_gold, gsm8k_rewards,
                    last_boxed, math_correct, extract_letter)

# GSM8K
assert gsm8k_gold("She has 3 + 4 = 7 apples.\n#### 7") == "7"
assert gsm8k_gold("...\n#### 1,200") == "1200"
assert extract_gsm8k("so the total is 18.\n#### 18") == 18
assert extract_gsm8k("The answer is $1,250.") == 1250
assert extract_gsm8k("#### -3") == -3
assert extract_gsm8k("no numbers here") is None
assert gsm8k_correct("Step 1 ... 12 + 6 = 18\n#### 18", "18")
assert not gsm8k_correct("#### 17", "18")
assert gsm8k_rewards(["#### 18", "the answer is 18", "#### 5", "no idea"], "18") == [1.1, 1.0, 0.1, 0.0]

# MATH-500
assert last_boxed("so \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"
assert last_boxed("first \\boxed{1} then \\boxed{2}") == "2"
assert last_boxed("cut off \\boxed{\\frac{1}{") is None
assert math_correct("the answer is \\boxed{\\dfrac{1}{2}}", "\\frac{1}{2}")
assert math_correct("\\boxed{ 5 }", "5")
assert not math_correct("\\boxed{6}", "5")
assert not math_correct("I think it's 5", "5")  # no \boxed{} means no credit

# MMLU
assert extract_letter("B") == "B"
assert extract_letter("The answer is (C).") == "C"
assert extract_letter("Answer: d") == "D"

print("All checks passed")
