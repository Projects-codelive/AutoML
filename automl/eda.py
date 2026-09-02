import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from automl.logger import get_logger
import re
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
    return {"rows": number_of_row, "columns": number_of_column,
            "duplicates": int(duplicates), "memory_mb": round(mem_mb, 2)}

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

def get_column_types(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    numerical_continuous = [c for c in num_cols if df[c].nunique() > 15]
    numerical_discrete   = [c for c in num_cols if df[c].nunique() <= 15]

    return {
        "numerical_continuous": numerical_continuous,
        "numerical_discrete": numerical_discrete,
        "categorical": cat_cols,
        "boolean": bool_cols,
        "datetime": datetime_cols,
    }

def missing_value(df: pd.DataFrame) -> pd.DataFrame:
    report = []
    most_missing = []
    for col in df.columns:
        missing_val = df[col].isnull().sum()
        if missing_val > 0:
            missing_pct = (missing_val / len(df)) * 100
            if missing_pct > 30:
                most_missing.append(col)
            report.append({
                "Column": col,
                "Missing Count": missing_val,
                "Missing Percentage": missing_pct
            })
    logger.warning(f"Columns with >30% missing (need special treatment): {most_missing}")
    report_df = pd.DataFrame(report)
    if not report_df.empty:
        report_df = report_df.sort_values(by="Missing Percentage", ascending=False).reset_index(drop=True)
    return report_df.to_dict(orient="records"), most_missing

def plot_missing_values(df: pd.DataFrame, save_path: str = None):
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if missing_pct.empty:
        print("No missing values to plot")
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_pct.values, y=missing_pct.index, color="steelblue")
    plt.title("Missing Values by Column (%)")
    plt.xlabel("Missing (%)")
    plt.ylabel("Column")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def descriptive_statics(df: pd.DataFrame):
    numerical_col = df.select_dtypes(include=["number", "float", "int64"]).columns.tolist()
    categorical_col = df.select_dtypes(include=["object", "category"]).columns.tolist()
    stats_report = {"numerical": {}, "categorical": {}}
    highly_skewd = []
    # Statics extraction for the numerical columns
    for col in numerical_col:
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()
        min = df[col].min()
        max = df[col].max()
        skewness = df[col].skew()
        if(abs(skewness) > 1):
            highly_skewd.append(col)
        kurtosis = df[col].kurt()
        logger.info(f"Column: {col}")
        logger.info(f"mean: {mean:.4f}")      # The :.4f rounds the number to 4 decimal places for cleaner output
        logger.info(f"median: {median:.4f}")
        logger.info(f"std: {std:.4f}")
        logger.info(f"min: {min:.4f}")
        logger.info(f"max: {max:.4f}")
        logger.info(f"skewness: {skewness:.4f}")
        logger.info(f"kurtosis: {kurtosis:.4f}")
        logger.info("-" * 20)
        stats_report["numerical"][col] = {
            "mean": mean,
            "median": median,
            "std": std,
            "min": min,
            "max": max,
            "skew": skewness,
            "kurtosis": kurtosis
        }
    logger.info(f"{highly_skewd} this column has more skweness So we need Special treatment like log/sqrt transform")
    logger.info("-" * 20)
    # Statics extraction for the Categorial columns
    for col in categorical_col:
        counts = df[col].value_counts()
        top_value = counts.index[0]
        top_frequency = counts.iloc[0]
        total_non_null = df[col].notna().sum()
        top_pct = round((top_frequency / total_non_null) * 100, 2)
        logger.info(f"Column: {col}")
        logger.info(f"unique count: {df[col].nunique()}")
        logger.info(f"top value: {top_value}")
        logger.info(f"top value frequency: {top_frequency} ({top_pct}%)")
        if top_pct > 95:
            logger.warning(f"'{col}' is quasi-constant — {top_pct}% rows are '{top_value}'")
        logger.info("-" * 20)
        stats_report["categorical"][col] = {
            "unique_count": df[col].nunique(),
            "top_value": top_value,
            "top_frequency": int(top_frequency),
            "top_pct": top_pct,
        }
    return stats_report, highly_skewd

def Outlier_detection(df: pd.DataFrame):
    numerical_col = df.select_dtypes(include=["number", "float", "int64"]).columns.tolist()
    outlier_report = {}
    for col in numerical_col:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3-q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        mask = (df[col] < lower_limit) | (df[col] > upper_limit)
        count = int(mask.sum())
        pct = round(count / len(df) * 100, 2)
        outlier_report[col] = {"count": count, "percentage": pct}
    return outlier_report


# Step 6 — Cardinality Check on Categoricals
def check_cardianility(df: pd.DataFrame) -> pd.DataFrame:
    report = {
        "Low Cardinality Prefer One Hot": [],
        "High Cardinality Prefer Target / Feature Encoding": [],
        "Flagged as Quasi-Constant! Consider dropping": []
    }
    categorical_col = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_col:
        # Edge case: If a column is completely empty, skip it to avoid errors
        if df[col].dropna().empty:
            continue
        dominant_value_percentage = df[col].value_counts(normalize=True).iloc[0]
        unique_count = df[col].nunique()
        if dominant_value_percentage > 0.95:
            report["Flagged as Quasi-Constant! Consider dropping"].append(col)
            continue
        if unique_count <= 30:
            report["Low Cardinality Prefer One Hot"].append(col)
        else:
            report["High Cardinality Prefer Target / Feature Encoding"].append(col)
    return report

# Step 7 — Target Column Analysis (Supervised only)
def target_column_analysis(df: pd.DataFrame, task_type: str, target_col: str, save_path: str = None):
    if task_type == "Clustering":
        return None
    if target_col not in df.columns:
        logger.error("Column is not found in the dataframe")
        return None
    y = df[target_col]
    if task_type == "Classification":
        counts = y.value_counts()
        percentage = y.value_counts(normalize=True) * 100
        logger.info(f"Target column: {target_col}")
        logger.info(f"Number of classes: {y.nunique()}")
        logger.info(f"Counts: {counts}")
        logger.info(f"Percentage: {percentage.round(2)}")
        # Plot
        plt.figure(figsize=(8, 5))
        sns.countplot(x=y)
        plt.title(f"Class Distribution — {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        # Imbalance Ratio
        imbalance_ratio = counts.max() / counts.min()
        logger.info(f"Imbalance ratio (majority/minority): {imbalance_ratio:.2f}")
        # ── Flag minority classes < 5% share ────────────────────────────
        minority_classes = percentage[percentage < 5]
        is_imbalanced = not minority_classes.empty
        if is_imbalanced:
            logger.warning("⚠️ Class imbalance detected — classes under 5% share:")
            logger.info(minority_classes.round(2))
        else:
            logger.info("Class distribution looks balanced")

        return {
            "class_counts": counts.to_dict(),
            "class_percentages": percentage.to_dict(),
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": is_imbalanced,
            "minority_classes": minority_classes.index.tolist()
        }
    elif task_type == "Regression":
        # ── Distribution plot ────────────────────────────────────────────
        plt.figure(figsize=(8, 5))
        sns.histplot(y, kde=True)
        plt.title(f"Target Distribution — {target_col}")
        plt.xlabel(target_col)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        # ── Skewness ─────────────────────────────────────────────────────
        skewness = y.skew()
        is_skewed = abs(skewness) > 1
        logger.info(f"Target column: {target_col}")
        logger.info(f"Skewness: {skewness:.3f}")
        if is_skewed:
            logger.warning("⚠️ Target is highly skewed — consider log transform")
        else:
            logger.info("Target distribution looks roughly normal")
        return {
            "skewness": skewness,
            "is_skewed": is_skewed,
            "mean": y.mean(),
            "median": y.median(),
            "std": y.std()
        }
    else:
        logger.error(f"Unknown task_type: {task_type} — skipping target analysis")
        return None

# Step 8 — Correlation Analysis (Comprehensive Numerical + Categorical)
def correlation_Analysis(df: pd.DataFrame, task_type: str = None, target_col: str = None, threshold: float = 0.85, save_path: str = None):
    numerical_col = df.select_dtypes(include=["number", "float", "int64"]).columns.tolist()
    categorical_col = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    
    multicollinear_pairs = []
    corr_matrix = None
    target_corr_ranked = {}

    # ── 1. Numerical Pearson & Multicollinearity ───────────────────────────
    if len(numerical_col) >= 2:
        corr_matrix = df[numerical_col].corr(method="pearson")
        # Multicollinearity
        for i in range(len(numerical_col)):
            for j in range(i+1, len(numerical_col)):
                col_a, col_b = numerical_col[i], numerical_col[j]
                val = corr_matrix.loc[col_a, col_b]
                if pd.notna(val) and abs(val) > threshold:
                    multicollinear_pairs.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "corr": round(float(val), 4)
                    })
        if multicollinear_pairs:
            logger.info(f"Total Multicollinear pairs: {len(multicollinear_pairs)}")
            for pair in multicollinear_pairs:
                logger.info(f"Correlation between {pair['col_a']} and {pair['col_b']} is {pair['corr']:.3f}")

        # Heatmap plot
        plt.figure(figsize=(max(8, len(numerical_col)), max(6, int(len(numerical_col)*0.8))))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Pearson Correlation Matrix (Numerical)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    # ── 2. Target Correlation / Association Ranking ────────────────────────
    if task_type in ("Regression", "Classification") and target_col is not None and target_col in df.columns:
        target_series = df[target_col].dropna()
        valid_idx = target_series.index

        # A. If target is numeric (Regression)
        if pd.api.types.is_numeric_dtype(df[target_col]):
            # Numerical features -> Pearson correlation
            for col in numerical_col:
                if col == target_col:
                    continue
                s = df.loc[valid_idx, col].dropna()
                common_idx = s.index.intersection(valid_idx)
                if len(common_idx) > 5:
                    r = df.loc[common_idx, col].corr(df.loc[common_idx, target_col])
                    if pd.notna(r):
                        target_corr_ranked[col] = round(float(r), 4)

            # Categorical features -> Correlation Ratio (Eta)
            for col in categorical_col:
                if col == target_col:
                    continue
                sub = df.loc[valid_idx, [col, target_col]].dropna()
                if sub[col].nunique() > 1 and len(sub) > 10:
                    try:
                        # Correlation ratio eta = sqrt(SS_between / SS_total)
                        groups = [group[target_col].values for _, group in sub.groupby(col) if len(group) > 0]
                        if len(groups) > 1:
                            grand_mean = sub[target_col].mean()
                            ss_total = ((sub[target_col] - grand_mean) ** 2).sum()
                            ss_between = sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in groups)
                            if ss_total > 0:
                                eta = np.sqrt(max(0.0, min(1.0, ss_between / ss_total)))
                                target_corr_ranked[col] = round(float(eta), 4)
                    except Exception as e:
                        logger.debug(f"Could not compute correlation ratio for {col}: {e}")

        # B. If target is categorical (Classification)
        else:
            # Categorical vs Categorical -> Cramér's V
            # Categorical vs Numerical -> Correlation Ratio (Eta)
            for col in numerical_col:
                sub = df.loc[valid_idx, [col, target_col]].dropna()
                if sub[target_col].nunique() > 1 and len(sub) > 10:
                    try:
                        groups = [group[col].values for _, group in sub.groupby(target_col) if len(group) > 0]
                        if len(groups) > 1:
                            grand_mean = sub[col].mean()
                            ss_total = ((sub[col] - grand_mean) ** 2).sum()
                            ss_between = sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in groups)
                            if ss_total > 0:
                                eta = np.sqrt(max(0.0, min(1.0, ss_between / ss_total)))
                                target_corr_ranked[col] = round(float(eta), 4)
                    except Exception as e:
                        logger.debug(f"Could not compute correlation ratio for {col}: {e}")

            for col in categorical_col:
                if col == target_col:
                    continue
                sub = df.loc[valid_idx, [col, target_col]].dropna()
                if sub[col].nunique() > 1 and len(sub) > 10:
                    try:
                        confusion_mat = pd.crosstab(sub[col], sub[target_col])
                        from scipy.stats import chi2_contingency
                        chi2 = chi2_contingency(confusion_mat)[0]
                        n = confusion_mat.sum().sum()
                        min_dim = min(confusion_mat.shape) - 1
                        if min_dim > 0 and n > 0:
                            cramers_v = np.sqrt(chi2 / (n * min_dim))
                            target_corr_ranked[col] = round(float(cramers_v), 4)
                    except Exception as e:
                        logger.debug(f"Could not compute Cramér's V for {col}: {e}")

        # Sort by absolute strength
        target_corr_ranked = dict(sorted(target_corr_ranked.items(), key=lambda item: abs(item[1]), reverse=True))
        logger.info(f"\nFeature association with target '{target_col}' (ranked by strength):")
        for k, v in list(target_corr_ranked.items())[:15]:
            logger.info(f" - {k}: {v}")

    return {
        "multicollinear_pairs": multicollinear_pairs,
        "target_corr_ranked": target_corr_ranked
    }


def _make_json_safe(obj):
    """Recursively convert numpy/pandas objects into JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Series):
        return _make_json_safe(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return _make_json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def save_eda_artifacts(df, columns_type, task_type, target_col, cleaning_report,
                        missing_report, high_missing_cols, highly_skewed, desc_stats, outlier_report,
                        cardinality_report, target_analysis_result, correlation_result,
                        output_dir="artifacts/eda"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_missing_values(df, save_path=out_dir / "missing_values.png")
    # ── Build summary ─────────────────────────────────────────────────
    eda_summary = {
        "column_types": columns_type,
        "task_type": task_type,
        "target_col": target_col,
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "cleaning_report": cleaning_report,
        "high_missing_cols": high_missing_cols,
        "highly_skewed_cols": highly_skewed,
        "missing_report": missing_report,
        "descriptive_stats": desc_stats,
        "outlier_report": outlier_report,
        "cardinality_report": cardinality_report,
        "target_analysis": target_analysis_result,
        "correlation": {
            "multicollinear_pairs": correlation_result["multicollinear_pairs"] if correlation_result else [],
            "target_corr_ranked": correlation_result["target_corr_ranked"] if correlation_result else None,
        } if correlation_result else None,
    }
    safe_summary = _make_json_safe(eda_summary)
    with open(out_dir / "eda_summary.json", "w") as f:
        json.dump(safe_summary, f, indent=2)

    print(f"Artifacts saved to {out_dir.resolve()}")
    return eda_summary


def run_eda(df: pd.DataFrame, task_type: str = None, target_col: str = None, cleaning_report: dict = None, output_dir: str = "artifacts/eda"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    basic_info(df)
    columns_type = get_column_types(df)
    missing_report, high_missing_cols = missing_value(df)
    desc_stats, highly_skewed = descriptive_statics(df)
    outlier_report = Outlier_detection(df)
    cardinality_report = check_cardianility(df)
    target_analysis_result = target_column_analysis(df, task_type, target_col,
                                save_path=Path(output_dir) / "target_distribution.png")
    correlation_result = correlation_Analysis(df, task_type, target_col,
                                save_path=Path(output_dir) / "correlation_heatmap.png")
    eda_summary = save_eda_artifacts(
        df, columns_type, task_type, target_col, cleaning_report, missing_report,
        high_missing_cols, highly_skewed, desc_stats, outlier_report,
        cardinality_report, target_analysis_result,
        correlation_result, output_dir=output_dir
    )
    return eda_summary