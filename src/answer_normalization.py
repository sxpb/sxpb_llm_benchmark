import re
from typing import Dict, Any, Tuple, Literal, Optional, Set, Union, List
from datetime import datetime

AnswerType = Literal[
    'integer',
    'number',
    'boolean',
    'date',
    'string',
    'csv-list-ordered',
    'csv-list-unordered',
]

NormalizationOptions = Dict[str, Any]

DEFAULT_OPTIONS: NormalizationOptions = {
    "tolerance": 1e-6,
    "case_sensitive": False,
    "allow_currency": True,
    "allow_percent": True,
    "decimal_places": None,
}

INTEGER_PATTERN_WITH_CURRENCY = r"[$€£¥]?\s*-?\d[\d,]*"
INTEGER_PATTERN = r"-?\d[\d,]*"
NUMBER_PATTERN_WITH_CURRENCY = r"[$€£¥]?\s*-?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?%?"
NUMBER_PATTERN = r"-?\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?%?"
WRAPPING_QUOTES_PATTERN = r"""^["']|["']$"""
CODE_FENCE_PATTERN = r"^```[\s\S]*?```"
LANGUAGE_IDENTIFIER_PATTERN = r"^\w+\n"
CURRENCY_AND_FORMATTING_CHARS = r"[$€£¥,\s]"
NUMBER_CLEANUP_CHARS = r"[$€£¥,%\s]"

TRUE_VALUES: Set[str] = {"true", "yes", "y", "1"}
FALSE_VALUES: Set[str] = {"false", "no", "n", "0"}

def strip_wrapping_quotes(text: str) -> str:
    return re.sub(WRAPPING_QUOTES_PATTERN, "", text.strip())

def normalize_integer(text: str, options: NormalizationOptions) -> Tuple[bool, Union[int, str]]:
    pattern = INTEGER_PATTERN_WITH_CURRENCY if options["allow_currency"] else INTEGER_PATTERN
    match = re.search(pattern, text)
    if not match:
        return False, f"No integer found in: \"{text}\""

    normalized_value = re.sub(CURRENCY_AND_FORMATTING_CHARS, "", match.group(0))
    try:
        return True, int(normalized_value)
    except ValueError:
        return False, f"Failed to parse integer: \"{match.group(0)}\""

def normalize_number(text: str, options: NormalizationOptions) -> Tuple[bool, Union[float, str]]:
    pattern = NUMBER_PATTERN_WITH_CURRENCY if options["allow_currency"] else NUMBER_PATTERN
    match = re.search(pattern, text)
    if not match:
        return False, f"No number found in: \"{text}\""

    token = match.group(0)
    has_percent_sign = options["allow_percent"] and token.endswith('%')
    normalized_token = re.sub(NUMBER_CLEANUP_CHARS, "", token)

    try:
        parsed_number = float(normalized_token)
    except ValueError:
        return False, f"Failed to parse number: \"{token}\""

    if has_percent_sign:
        parsed_number /= 100

    if options["decimal_places"] is not None:
        factor = 10 ** options["decimal_places"]
        parsed_number = round(parsed_number * factor) / factor

    return True, parsed_number

def normalize_boolean(text: str, options: NormalizationOptions) -> Tuple[bool, Union[bool, str]]:
    normalized_value = text.strip().lower()
    if normalized_value in TRUE_VALUES:
        return True, True
    if normalized_value in FALSE_VALUES:
        return True, False
    return False, f"Not a boolean: \"{text}\""

def normalize_date(text: str) -> Tuple[bool, Union[str, str]]:
    cleaned = strip_wrapping_quotes(text)
    try:
        # This is a simplification. The JS version is more robust.
        parsed_date = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        return True, parsed_date.strftime("%Y-%m-%d")
    except ValueError:
        return False, f"Invalid date: \"{text}\""

def normalize_string(text: str, options: NormalizationOptions) -> Tuple[bool, str]:
    trimmed_text = text.strip()
    trimmed_text = re.sub(WRAPPING_QUOTES_PATTERN, "", trimmed_text)

    def repl(match):
        inner = match.group(0)[3:-3].strip()
        return re.sub(LANGUAGE_IDENTIFIER_PATTERN, "", inner)

    trimmed_text = re.sub(CODE_FENCE_PATTERN, repl, trimmed_text)
    trimmed_text = trimmed_text.strip()

    value = trimmed_text if options["case_sensitive"] else trimmed_text.lower()
    return True, value

def normalize_csv_list_ordered(text: str, options: NormalizationOptions) -> Tuple[bool, Union[List[str], str]]:
    stripped_text = strip_wrapping_quotes(text)
    items = [item.strip() for item in stripped_text.split(',') if item.strip()]
    normalized_items = [item if options["case_sensitive"] else item.lower() for item in items]
    return True, normalized_items

def normalize_csv_list_unordered(text: str, options: NormalizationOptions) -> Tuple[bool, Union[List[str], str]]:
    success, value = normalize_csv_list_ordered(text, options)
    if not success:
        return False, value
    return True, sorted(value)


def normalize_answer(
    text: str,
    kind: AnswerType,
    options: Optional[NormalizationOptions] = None,
) -> Tuple[bool, Any]:
    resolved_options = {**DEFAULT_OPTIONS, **(options or {})}

    normalizers = {
        "integer": normalize_integer,
        "number": normalize_number,
        "boolean": normalize_boolean,
        "date": normalize_date,
        "string": normalize_string,
        "csv-list-ordered": normalize_csv_list_ordered,
        "csv-list-unordered": normalize_csv_list_unordered,
    }

    if kind in normalizers:
        return normalizers[kind](text, resolved_options)
    return False, f"Unknown answer kind: {kind}"

def compare_values(actual: Any, expected: Any, kind: AnswerType, options: NormalizationOptions) -> bool:
    if kind in ["integer", "boolean", "date", "string"]:
        return actual == expected
    if kind == "number":
        if options["decimal_places"] is not None:
            return actual == expected
        return abs(actual - expected) <= options["tolerance"]
    if kind in ["csv-list-ordered", "csv-list-unordered"]:
        return actual == expected
    return False

def compare_answers(
    actual: str,
    expected: str,
    kind: AnswerType,
    options: Optional[NormalizationOptions] = None,
) -> Tuple[bool, Optional[str]]:
    resolved_options = {**DEFAULT_OPTIONS, **(options or {})}

    actual_success, actual_value = normalize_answer(actual, kind, resolved_options)
    if not actual_success:
        return False, f"Failed to normalize actual answer: {actual_value}"

    expected_success, expected_value = normalize_answer(expected, kind, resolved_options)
    if not expected_success:
        return False, f"Failed to normalize expected answer: {expected_value}"

    match = compare_values(actual_value, expected_value, kind, resolved_options)

    details = None
    if not match:
        details = f"Mismatch: actual=\"{actual_value}\" vs expected=\"{expected_value}\""

    return match, details
