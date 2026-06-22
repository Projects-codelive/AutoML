import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from automl.logger import get_logger
import re

logger = get_logger("Eda")
def basic_info(df: pd.DataFrame):
    number_of_row = df.shape[0]
    number_of_column = df.shape[1]
    duplicates = df.duplicated().sum()
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f'Number of rows: {number_of_row}')
    logger.info(f'Number of columns: {number_of_column}')
    logger.info(f'Duplicates: {duplicates}')
    logger.info(f'Memory usage: {mem_mb:.2f} MB')

_BOOL_MAP = {
    "yes": 1, "no": 0, "true": 1, "false": 0,
    "y": 1, "n": 0, "1": 1, "0": 0,
}
_SYMBOL_PATTERN = re.compile(
    r"""
    ^\s*                    # leading whitespace
    [₹$€£¥₩%+\-]?          # optional currency or sign prefix
    [\s]?                   # optional space after symbol
    |                       # OR
    \s*                     # trailing whitespace
    (mi\.?|km\.?|kg\.?|lbs?\.?|mph|kph|hp|cc|°[cf]?|%|[₹$€£¥₩])
    \s*$                    # end
    """,
    re.IGNORECASE | re.VERBOSE,
)
_DATE_CHAR_PATTERN = re.compile(r"[/\-:]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", re.IGNORECASE)

def _strip_to_numeric(value: str) -> str:
    """Remove currency symbols, units, commas from a string value."""
    v = _SYMBOL_PATTERN.sub("", str(value))         # strip prefix/suffix symbols
    v = v.replace(",", "")                          # remove thousand separators
    return v.strip()

def _is_dirty_numeric(series: pd.Series, sample_size: int = 50) -> bool:
    """
    Returns True if the column looks like a number wearing a costume.
    Tests on a sample: strip symbols → try float cast → check success rate.
    """
    sample = series.dropna().head(sample_size)
    if len(sample) == 0:
        return False
    success = 0
    for val in sample:
        try:
            float(_strip_to_numeric(str(val)))
            success += 1
        except ValueError:
            pass
    return (success / len(sample)) >= 0.90   # 90% must parse cleanly

def _is_dirty_boolean(series: pd.Series) -> bool:
    """
    Returns True if all unique non-null values map to yes/no/true/false etc.
    """
    unique_vals = set(series.dropna().astype(str).str.lower().unique())
    return unique_vals.issubset(_BOOL_MAP.keys())

def _is_mostly_numeric(series: pd.Series, threshold: float = 0.85) -> bool:
    """
    Returns True if pd.to_numeric coerces successfully on >threshold fraction.
    Catches columns like ["35", "40", "N/A", "55"] — mostly numeric, few bad rows.
    """
    coerced = pd.to_numeric(series, errors="coerce")
    non_null_original = series.notna().sum()
    if non_null_original == 0:
        return False
    success_rate = coerced.notna().sum() / non_null_original
    return success_rate >= threshold

def _clean_numeric_column(series: pd.Series) -> pd.Series:
    """Strip symbols and cast to float. Unparseable → NaN."""
    def _convert(val):
        try:
            return float(_strip_to_numeric(str(val)))
        except ValueError:
            return np.nan

    return series.apply(_convert)

def _clean_boolean_column(series: pd.Series) -> pd.Series:
    """Map yes/no/true/false → 1/0. Unknowns → NaN."""
    return series.astype(str).str.lower().map(_BOOL_MAP)

def _is_datetime_like(series: pd.Series, sample_size: int = 50, threshold: float = 0.85) -> bool:
    """
    Returns True if the column looks like dates.
    Guards against pure numeric columns (e.g. "2021") being misread as dates
    by requiring date-separator characters or month names in the raw text.
    """
    sample = series.dropna().astype(str).head(sample_size)
    if sample.empty:
        return False

    # require at least one value to contain a date-like character/word
    has_date_chars = sample.str.contains(_DATE_CHAR_PATTERN, regex=True).mean()
    if has_date_chars < 0.5:
        return False

    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    success_rate = parsed.notna().mean()
    return success_rate >= threshold


def _clean_datetime_column(series: pd.Series) -> pd.Series:
    """Parse to datetime. Unparseable → NaT."""
    return pd.to_datetime(series, errors="coerce", format="mixed")

def run_basic_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Detects and fixes dirty columns automatically.

    Pass in the raw loaded DataFrame. Returns:
        - cleaned_df  : DataFrame with corrected dtypes
        - cleaning_report : dict logging what was changed and why
    """
    df = df.copy()
    report = {
        "converted_to_numeric": [],
        "converted_to_boolean": [],
        "coerced_mostly_numeric": [],
        "converted_to_datetime": [],
        "unchanged": [],
    }

    object_cols = df.select_dtypes(include=["object", "string", "category"]).columns

    for col in object_cols:
        series = df[col]

        # ── Priority 0: Datetime check (must run before boolean/numeric) ──────────
        if _is_datetime_like(series):
            df[col] = _clean_datetime_column(series)
            report["converted_to_datetime"].append(col)

        # ── Priority 1: Boolean check first (yes/no cols are also "numeric-ish") ──
        elif _is_dirty_boolean(series):
            df[col] = _clean_boolean_column(series)
            report["converted_to_boolean"].append(col)

        # ── Priority 2: Dirty numeric (currency, units, commas) ───────────────────
        elif _is_dirty_numeric(series):
            df[col] = _clean_numeric_column(series)
            report["converted_to_numeric"].append(col)

        # ── Priority 3: Mostly numeric with a few bad rows ────────────────────────
        elif _is_mostly_numeric(series):
            df[col] = pd.to_numeric(series, errors="coerce")
            report["coerced_mostly_numeric"].append(col)

        # ── No pattern matched: leave as-is for EDA to handle ────────────────────
        else:
            report["unchanged"].append(col)

    return df, report


