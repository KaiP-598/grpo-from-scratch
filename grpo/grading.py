"""
Math answer grading with high recall.

Adapted from the understand-r1-zero project (Apache 2.0 license):
https://github.com/sail-sg/understand-r1-zero

Combines multiple strategies: string normalization, sympy symbolic
comparison, and math_verify for robust answer checking.
"""
# Copyright 2025 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0

import re
import signal
from itertools import islice, zip_longest
from math import isclose
from typing import Optional

import sympy
from latex2sympy2_extended import latex2sympy
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from pylatexenc import latex2text
from sympy import N, simplify
from sympy.parsing import sympy_parser
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


# ---------------------------------------------------------------------------
# String normalization (from Hendrycks' MATH evaluation code)
# ---------------------------------------------------------------------------

def _normalize_mathd(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


_UNIT_TEXTS = [
    "east", "degree", "mph", "kmph", "ft", "m sqaure", " m east", "sq m",
    "deg", "mile", "q .", "monkey", "prime", "ratio", "profit of rs", "rd",
    "o", "gm", "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab",
    "way", "west", "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .",
    "t", "a", "h", "no change", "men", "soldier", "pie", "bc", "excess",
    "st", "inches", "noon", "percent", "by", "gal", "kmh", "c", "acre",
    "rise", "a . m", "th", "π r 2", "sq", "mark", "l", "toy", "coin",
    "sq . m", "gallon", "° f", "profit", "minw", "yr", "women", "feet",
    "am", "pm", "hr", "cu cm", "square", "v â € ™", "are", "rupee",
    "rounds", "cubic", "cc", "mtr", "s", "ohm", "number", "kmph", "day",
    "hour", "minute", "min", "second", "man", "woman", "sec", "cube", "mt",
    "sq inch", "mp", "∏ cm ³", "hectare", "more", "sec", "unit", "cu . m",
    "cm 2", "rs .", "rs", "kg", "g", "month", "km", "m", "cm", "mm",
    "apple", "liter", "loss", "yard", "pure", "year", "increase", "decrease",
    "d", "less", "Surface", "litre", "pi sq m", "s .", "metre", "meter", "inch",
]
_UNIT_TEXTS.extend([t + "s" for t in _UNIT_TEXTS])


def _strip_string(string):
    def _fix_fracs(s):
        substrs = s.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            for substr in substrs[1:]:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return s
                    a, b = substr[0], substr[1]
                    if b != "{":
                        post = substr[2:] if len(substr) > 2 else ""
                        new_str += "{" + a + "}{" + b + "}" + post
                    else:
                        post = substr[2:] if len(substr) > 2 else ""
                        new_str += "{" + a + "}" + b + post
        return new_str

    def _fix_a_slash_b(s):
        if len(s.split("/")) != 2:
            return s
        a, b = s.split("/")
        try:
            a, b = int(a), int(b)
            assert s == f"{a}/{b}"
            return "\\frac{" + str(a) + "}{" + str(b) + "}"
        except:
            return s

    def _fix_sqrt(s):
        if "\\sqrt" not in s:
            return s
        splits = s.split("\\sqrt")
        new_s = splits[0]
        for sp in splits[1:]:
            if sp[0] != "{":
                new_s += "\\sqrt{" + sp[0] + "}" + sp[1:]
            else:
                new_s += "\\sqrt" + sp
        return new_s

    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    string = string.replace("\\left", "").replace("\\right", "")

    _s = re.sub(r"\\text{.*?}$", "", string).strip()
    if _s != "" and _s != string:
        string = _s

    for _ in range(2):
        for unit in _UNIT_TEXTS:
            _s = re.sub(r"(^|\W)" + unit + r"($|\W)", r"\1\2", string)
            if _s != "":
                string = _s

    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    if "\\text{ " in string:
        string = string.split("\\text{ ")[0]
    string = string.replace("\\%", "").replace("\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if len(string) > 0 and string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


# ---------------------------------------------------------------------------
# Substitutions and normalization (from Minerva / Hendrycks)
# ---------------------------------------------------------------------------

_SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

_REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours",
    "km", "units", "\\ldots", "sue", "points", "feet", "minutes", "digits",
    "cents", "degrees", "cm", "gm", "pounds", "meters", "meals", "edges",
    "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}",
    "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def _normalize_final(answer: str) -> str:
    for before, after in _SUBSTITUTIONS:
        answer = answer.replace(before, after)
    for expr in _REMOVED_EXPRESSIONS:
        answer = answer.replace(expr, "")
    answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", answer)
    answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", answer)
    answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", answer)
    answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", answer)
    answer = answer.replace("$", "")
    if answer.replace(",", "").isdigit():
        answer = answer.replace(",", "")
    return answer


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

class _Timeout:
    def __init__(self, seconds=1):
        self.seconds = seconds
    def handle_timeout(self, signum, frame):
        raise TimeoutError
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, *args):
        signal.alarm(0)


def _repeatness(s: str) -> bool:
    def ranks(l):
        idx = {v: i for i, v in enumerate(sorted(set(l)))}
        return [idx[v] for v in l]
    def suffix_array(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa
    def lcp(arr, sa, inv):
        n, ans, k = len(arr), [0] * len(arr), 0
        for i in range(n):
            if inv[i] == n - 1:
                k = 0
                continue
            j = sa[inv[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1
            ans[inv[i]] = k
            if k > 0:
                k -= 1
        return ans
    arr = [ord(c) for c in s]
    n = len(arr)
    if n <= 1:
        return False
    c, sa = suffix_array(arr)
    cnt = sum(lcp(arr, sa, c))
    return (cnt * 2 / (n * (n + 1))) > 0.2


def _numeric_equal(pred: float, ref: float) -> bool:
    return isclose(ref, pred, rel_tol=1e-4)


def _symbolic_equal(a_str, b_str) -> bool:
    def _try_parse(s):
        for fn in [parse_latex, parse_expr, latex2sympy]:
            try:
                return fn(s.replace("\\\\", "\\"))
            except:
                try:
                    return fn(s)
                except:
                    pass
        return s

    a, b = _try_parse(a_str), _try_parse(b_str)
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass
    try:
        if _numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass
    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass
    return False


def _parse_latex_text(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    for old, new in [("√", "sqrt"), ("π", "pi"), ("∞", "inf"),
                     ("∪", "U"), ("·", "*"), ("×", "*")]:
        expr = expr.replace(old, new)
    return expr.strip()


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_commas(x)
        return _is_int(float(x))
    except:
        return False


def _strip_commas(expr: str) -> str:
    p = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> Optional[str]:
    if expr is None:
        return None
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")
    expr = expr.replace("\\%", "%").replace("\\$", "$").replace("$", "").replace("%", "")
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")
    for unit in ["degree", "cm", "centimeter", "meter", "mile", "second",
                 "minute", "hour", "day", "week", "month", "year", "foot",
                 "feet", "inch", "yard"]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub("\\^ *\\\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex_text(expr)
        except:
            pass
    expr = re.sub("- *", "-", expr)
    expr = re.sub("([0-9]) +([0-9])", "\\1+\\2", expr)
    expr = expr.replace(" ", "").replace("{", "").replace("}", "").lower()
    if _str_is_int(expr):
        expr = str(int(round(float(_strip_commas(expr)))))
    return expr


_BAD_SUBSTRINGS = ["^{", "^("]
_BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
_TUPLE_CHARS = "()[]"


def _should_allow_eval(expr: str) -> bool:
    expr_clean = expr.replace("sqrt", "").replace("frac", "")
    if len(set(c for c in expr_clean if c.isalpha())) > 2:
        return False
    for bad in _BAD_SUBSTRINGS:
        if bad in expr:
            return False
    for pat in _BAD_REGEXES:
        if re.search(pat, expr):
            return False
    return True


def _sympy_equal(gt_norm: str, pred_norm: str) -> bool:
    try:
        expr = f"({gt_norm})-({pred_norm})"
        if _should_allow_eval(expr):
            py_expr = expr.replace("^", "**")
            diff = sympy_parser.parse_expr(
                py_expr,
                transformations=sympy_parser.standard_transformations
                + (sympy_parser.implicit_multiplication_application,),
            )
            if sympy.simplify(diff) == 0:
                return True
    except:
        pass
    return False


def _split_tuple(expr: str) -> list[str]:
    expr = _strip_commas(expr)
    if len(expr) == 0:
        return []
    if (len(expr) > 2 and expr[0] in _TUPLE_CHARS and expr[-1] in _TUPLE_CHARS
            and all(ch not in expr[1:-1] for ch in _TUPLE_CHARS)):
        return [e.strip() for e in expr[1:-1].split(",")]
    return [expr]


# ---------------------------------------------------------------------------
# Top-level answer extraction and grading
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i, depth, end = idx, 0, None
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
        i += 1
    if end is None:
        return None
    boxed = text[idx:end + 1]
    left = "\\boxed{"
    if boxed[:len(left)] == left and boxed[-1] == "}":
        return boxed[len(left):-1]
    return None


def grade(model_answer: str, gt_answer: str) -> bool:
    """Check if model_answer matches gt_answer using multiple strategies."""
    if "\\boxed" in gt_answer:
        gt_answer = extract_boxed(gt_answer) or gt_answer

    # Strategy 1: normalized string match (Hendrycks MATH)
    gt_mathd = _normalize_mathd(gt_answer)
    pred_mathd = _normalize_mathd(model_answer)
    if gt_mathd == pred_mathd:
        return True
    try:
        if float(gt_mathd) == float(pred_mathd):
            return True
    except:
        pass

    # Strategy 2: sympy symbolic comparison
    gt_norm = _normalize(gt_answer)
    pred_norm = _normalize(model_answer)
    if gt_norm is None:
        return False
    if gt_norm == pred_norm:
        return True
    if len(pred_norm) == 0:
        return False

    gt_elems = _split_tuple(gt_norm)
    pred_elems = _split_tuple(pred_norm)
    if len(gt_elems) > 1 and (gt_norm[0] != pred_norm[0] or gt_norm[-1] != pred_norm[-1]):
        return False
    if len(gt_elems) != len(pred_elems):
        return False

    for ge, pe in zip(gt_elems, pred_elems):
        if _is_frac(ge) and _is_frac(pe):
            if ge != pe:
                return False
        elif _str_is_int(ge) != _str_is_int(pe):
            return False
        elif not _sympy_equal(ge, pe):
            return False
    return True


def _latex_equal_safe(given: str, gt: str) -> bool:
    try:
        with _Timeout(1):
            if (len(given) > 128 and _repeatness(given)) or (len(gt) > 128 and _repeatness(gt)):
                return False
            gt_norm = _normalize(gt)
            pred_norm = _normalize(given)
            if gt_norm is None:
                return False
            if gt_norm == pred_norm:
                return True
            g = given.replace("\n", "")
            t = gt.replace("\n", "")
            if "$" not in g:
                g = f"${g}$"
            if "$" not in t:
                t = f"${t}$"
            return verify(
                parse(t, extraction_config=(LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()),
                      fallback_mode="no_fallback", extraction_mode=["first_match"], parsing_timeout=1),
                parse(g, extraction_config=(LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()),
                      fallback_mode="no_fallback", extraction_mode=["first_match"], parsing_timeout=1),
                timeout_seconds=1,
            )
    except:
        return False


# ---------------------------------------------------------------------------
# Reward functions for different prompt formats
# ---------------------------------------------------------------------------

def reasoning_reward(response: str, ground_truth: str) -> dict:
    """
    Reward function for the reasoning prompt format.
    Expects <think>...</think> <answer>...</answer> structure.
    """
    if "</think> <answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[-1].replace("</answer>", "")
        if "\\boxed" in answer:
            answer = extract_boxed(answer)
            if answer is None:
                return {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}
        if isinstance(ground_truth, (float, int)):
            ground_truth = str(ground_truth)
        if isinstance(ground_truth, str):
            correct = grade(answer, ground_truth)
        elif isinstance(ground_truth, list):
            correct = any(grade(answer, gt) for gt in ground_truth)
        return {
            "format_reward": 1.0,
            "answer_reward": 1.0 if correct else 0.0,
            "reward": 1.0 if correct else 0.0,
        }
    return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}


def direct_answer_reward(response: str, ground_truth: str) -> dict:
    """
    Reward function for the minimal prompt format (no reasoning tags).
    Extracts answer from \\boxed{} directly.
    """
    answer = extract_boxed(response)
    if answer is None:
        return {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}
    if isinstance(ground_truth, (float, int)):
        ground_truth = str(ground_truth)
    if isinstance(ground_truth, str):
        correct = grade(answer, ground_truth)
    elif isinstance(ground_truth, list):
        correct = any(grade(answer, gt) for gt in ground_truth)
    return {
        "format_reward": 1.0,
        "answer_reward": 1.0 if correct else 0.0,
        "reward": 1.0 if correct else 0.0,
    }
