import re

import pandas as pd

WORD_NUMS = {
    "one": 1,
    "single": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
NUM = r"(?P<num>\d{1,4}|" + "|".join(WORD_NUMS) + r")"
HOME = (
    r"(?:dwelling\s?houses?|dwellings?|houses?|homes?|flats?|apartments?|bungalows?|"
    r"cottages?|maisonettes?|villas?|townhouses?|units?)"
)
# number, optional "no."/"x", up to 3 descriptive words, then a home noun
# e.g. "22no. houses", "4 no. detached dwellinghouses", "14 flatted dwellings"
COUNT_RE = re.compile(
    rf"(?<![\w/.]){NUM}(?:\s*(?:no|nos|x)\b\.?)?\s+(?:[\w\-.½]+\s+){{0,3}}?{HOME}\b"
)
# number is a storey/bedroom/size descriptor, not a count of homes
NOT_COUNT_AFTER = re.compile(
    r"[\s\-]*(?:½|storey|story|bed|bedroom|apartment\s+block|kw|m\b|%)"
)
# number belongs to a condition, plot, class, phase etc.
NOT_COUNT_BEFORE = re.compile(
    r"(?:class|plot|plots|condition|phase|ref|no|from|of existing)\s*$"
)
# demolition clause, runs from "demoli..." until the next building verb or end of text,
# homes demolished are not homes created e.g. "demolition of 24 flats, erection of 12 dwellings"
DEMOLITION_RE = re.compile(
    r"\bdemoli\w*.*?(?=\b(?:erect|construct|form|creat|build\b|redevelop|convers|convert|"
    r"change of use|subdivi|provi|replac)|;|$)",
    re.DOTALL,
)
PLOT_RANGE_RE = re.compile(r"(\d+)\s*(?:-|to)\s*(\d+)")
PLOTS_RE = re.compile(r"\bplots\s+([\d\s\-,and to]+)")
# unquantified schemes, can't infer a number from text alone
UNKNOWN_RE = re.compile(
    r"residential development|housing development|mixed[\s\-]use|"
    r"\b(?:dwelling\s?houses|dwellings|houses|flats|apartments|homes|units)\b"
)
# singular mention without a number, e.g. "erection of dwellinghouse"
SINGLE_RE = re.compile(
    r"\b(?:dwelling\s?house|dwelling|house|home|flat|bungalow|cottage|residential)\b"
)


def _to_int(s):
    return int(s) if s.isdigit() else WORD_NUMS[s]


def _first_count(t):
    for m in COUNT_RE.finditer(t):
        if NOT_COUNT_AFTER.match(t, m.end("num")):
            continue
        if NOT_COUNT_BEFORE.search(t[: m.start("num")].rstrip()):
            continue
        n = _to_int(m.group("num"))
        if 0 < n < 5000:
            return n
    return None


def _plot_count(t):
    """'plots 22-25 and 31-32' -> 6"""
    m = PLOTS_RE.search(t)
    if not m:
        return None
    total = 0
    for a, b in PLOT_RANGE_RE.findall(m.group(1)):
        if int(b) > int(a):
            total += int(b) - int(a) + 1
    return total or None


def estimate_n_homes(text):
    """
    Estimate the number of homes from proposal text.
    Returns (n_homes, method), n_homes is None when it can't be determined.
    method: count (explicit number), plots (plot range), single (no number,
    singular home noun), unknown (plural/outline scheme, no number), none
    """
    if not isinstance(text, str):
        return None, "none"
    t = DEMOLITION_RE.sub(" ", text.lower())
    n = _first_count(t)
    if n:
        return n, "count"
    n = _plot_count(t)
    if n:
        return n, "plots"
    if UNKNOWN_RE.search(t):
        return None, "unknown"
    if SINGLE_RE.search(t):
        return 1, "single"
    return None, "none"


# the 11 output classes, in order
N_HOMES_CLASSES = [str(i) for i in range(1, 11)] + ["more than 10"]


def n_homes_class(n):
    """Map a number of homes to one of N_HOMES_CLASSES, None if no estimate"""
    if pd.isna(n) or n < 1:
        return None
    return "more than 10" if n > 10 else str(int(n))
