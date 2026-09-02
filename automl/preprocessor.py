import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from openai import OpenAI
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

from automl.logger import get_logger
import re
import json

logger = get_logger("Preprocessing")
def load_eda_summary(eda_summary_path: str = "../artifacts/eda/eda_summary.json") -> dict:
    path = Path(eda_summary_path)
    if not path.exists():
        raise FileNotFoundError(f"File '{eda_summary_path}' does not exist")
    with open(eda_summary_path) as f:
        eda_summary = json.load(f)
    logger.info(f"EDA summary loaded from {path.resolve()}")
    return  eda_summary

# Step 1 — Train-Test Split
def split_data(df: pd.DataFrame, target_col: str = None, task_type: str = None, test_size: float = 0.2, random_state: int = 42):
    df = df.copy()
    if task_type == "Clustering":
        x = df
        x_train, x_test = train_test_split(x, test_size=test_size, random_state=random_state)
        y_train, y_test = None, None
    else:
        if target_col is None or target_col not in df.columns:
            logger.info(f"Error: Could not find target column '{target_col}' for {task_type}.")
            return None, None, None, None
        x = df.drop(columns=[target_col])
        y = df[target_col]
        if task_type == "Regression":
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        else:
            class_counts = y.value_counts()
            if (class_counts < 2).any() or len(y) < 10:
                logger.info("Some classes have <2 samples. Splitting without stratification.")
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
            else:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
    logger.info("--- Train-Test Split Completed ---")
    logger.info(f"X_train shape: {x_train.shape} | X_test shape: {x_test.shape}")
    if y_train is not None:
        logger.info(f"y_train shape: {y_train.shape} | y_test shape: {y_test.shape}")
    else:
        logger.info("y_train: None | y_test: None (Clustering Task)")
    return x_train, x_test, y_train, y_test

# Step 2 — Drop Quasi-Constant and Useless Columns
def drop_useless_columns(x_train, x_test, eda_summary):
    dropeed_column = [];
    quasi_constant = eda_summary.get("cardinality_report", {}).get("Flagged as Quasi-Constant! Consider dropping", [])
    if len(quasi_constant) == 0:
        logger.info("No columns to drop.")
        return x_train, x_test, dropeed_column
    for col in quasi_constant:
        if col in x_train.columns:
            x_train = x_train.drop(columns=[col])
            x_test = x_test.drop(columns=[col])
            dropeed_column.append(col)
            logger.info(f"Dropping '{col}': Flagged as Quasi-Constant (almost zero variance).")
    for col in x_train.columns:
        if x_train[col].nunique() == len(x_train):
            logger.info(f"Dropping '{col}': Flagged as Unique Identifier (100% unique values).")
            dropeed_column.append(col)
    if len(dropeed_column) > 0:
        x_train = x_train.drop(columns=dropeed_column, errors="ignore")
        x_test = x_test.drop(columns=dropeed_column, errors="ignore")
        logger.info(f"\nSummary: Successfully dropped {len(dropeed_column)} useless columns.")
    else:
        logger.info("Summary: No quasi-constant or unique ID columns found. Nothing dropped.")
    return x_train, x_test, dropeed_column

# Step 3 — Missing Value Imputation
def impute_missing_values(x_train, x_test, eda_summary):
    x_train = x_train.copy()
    x_test = x_test.copy()
    high_missing_cols = eda_summary.get("high_missing_cols", [])
    missing_value = eda_summary.get("missing_report", [])
    missing_columns = [row["Column"] for row in eda_summary.get("missing_report", [])]
    highly_skewed_cols = eda_summary.get("highly_skewed_cols", [])
    col_types = eda_summary.get("column_types", {})
    numerical_continuous = col_types.get("numerical_continuous", [])
    numerical_discrete = col_types.get("numerical_discrete", [])
    categorical_cols = col_types.get("categorical", [])
    boolean_cols = col_types.get("boolean", [])
    datetime_cols = col_types.get("datetime", [])
    numerical_cols = numerical_continuous + numerical_discrete
    fitted_imputers = {}
    for col in missing_columns:
        if col not in x_train.columns:
            continue
        imputer = None
        if col in datetime_cols:
            logger.warning(f"WARNING: Datetime column '{col}' has nulls. Do not impute here. Handle in feature engineering.")
            continue
        elif col in numerical_cols:
            if col in high_missing_cols:
                logger.warning(f"WARNING: Numerical column '{col}' has >30% missing values. Results may be unreliable. Imputing with median.")
                imputer = SimpleImputer(strategy='median')
            else:
                if col in highly_skewed_cols:
                    imputer = SimpleImputer(strategy='median')
                else:
                    imputer = SimpleImputer(strategy='mean')
        elif col in categorical_cols:
            imputer = SimpleImputer(strategy='most_frequent')
        elif col in boolean_cols:
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            imputer = SimpleImputer(strategy='most_frequent')

        if imputer is not None:
            x_train[col] = imputer.fit_transform(x_train[[col]]).ravel()
            x_test[col] = imputer.transform(x_test[[col]]).ravel()
            fitted_imputers[col] = imputer
    return x_train, x_test, fitted_imputers

# Step 3.5 — Domain & Spec Feature Extraction (e.g., Engine Specs, Normalized Categories)
def extract_domain_features(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    x_train = x_train.copy()
    x_test = x_test.copy()
    extracted_cols = []

    # 1. Engine Specs Extraction
    if "engine" in x_train.columns:
        logger.info("Extracting structured specifications (HP, Displacement, Cylinders, Tier) from 'engine' column...")
        def _parse_engine(val):
            if pd.isna(val) or str(val).strip() in ["–", "", "not supported", "nan", "None"]:
                return np.nan, np.nan, np.nan
            val_str = str(val).upper()
            # Horsepower
            hp_m = re.search(r"(\d+\.?\d*)\s*HP", val_str)
            hp = float(hp_m.group(1)) if hp_m else np.nan
            # Displacement
            disp_m = re.search(r"(\d+\.?\d*)\s*(?:L|LITER)", val_str)
            disp = float(disp_m.group(1)) if disp_m else np.nan
            # Cylinders
            cyl_m = re.search(r"(\d+)\s*CYLINDER|V(\d+)|I(\d+)|H(\d+)|W(\d+)|(\d+)\s*CYL", val_str)
            cyl = None
            if cyl_m:
                for g in cyl_m.groups():
                    if g is not None:
                        cyl = float(g)
                        break
            return hp, disp, cyl

        train_specs = x_train["engine"].apply(_parse_engine).tolist()
        test_specs = x_test["engine"].apply(_parse_engine).tolist()

        train_spec_df = pd.DataFrame(train_specs, columns=["hp", "displacement", "cylinders"], index=x_train.index)
        test_spec_df = pd.DataFrame(test_specs, columns=["hp", "displacement", "cylinders"], index=x_test.index)

        # Impute missing parsed values with train median
        for c in ["hp", "displacement", "cylinders"]:
            med = train_spec_df[c].median()
            if pd.isna(med):
                med = 0.0
            train_spec_df[c] = train_spec_df[c].fillna(med)
            test_spec_df[c] = test_spec_df[c].fillna(med)

        # Engine Tier ordinal feature
        def _get_tier(row):
            hp = row["hp"]
            cyl = row["cylinders"]
            if hp > 380 or cyl >= 8:
                return 3  # High Performance
            elif hp < 200 or (cyl is not None and cyl <= 4):
                return 1  # Economy
            return 2      # Standard

        train_spec_df["engine_tier"] = train_spec_df.apply(_get_tier, axis=1)
        test_spec_df["engine_tier"] = test_spec_df.apply(_get_tier, axis=1)

        x_train = x_train.drop(columns=["engine"])
        x_test = x_test.drop(columns=["engine"])

        x_train = pd.concat([x_train, train_spec_df], axis=1)
        x_test = pd.concat([x_test, test_spec_df], axis=1)
        extracted_cols.extend(["hp", "displacement", "cylinders", "engine_tier"])
        logger.info(f"Engine specs successfully extracted: {['hp', 'displacement', 'cylinders', 'engine_tier']}")

    # 2. Rule-based Categorical Simplification (for columns with high variance/messy labels when AI grouping is skipped)
    if "accident" in x_train.columns:
        acc_map = {"none reported": 0, "at least 1 accident or damage reported": 1, "0": 0, "1": 1}
        x_train["accident"] = x_train["accident"].astype(str).str.lower().str.strip().map(acc_map).fillna(0).astype(int)
        x_test["accident"] = x_test["accident"].astype(str).str.lower().str.strip().map(acc_map).fillna(0).astype(int)

    if "clean_title" in x_train.columns:
        title_map = {"yes": 1, "1": 1, "true": 1, "no": 0, "0": 0, "false": 0}
        x_train["clean_title"] = x_train["clean_title"].astype(str).str.lower().str.strip().map(title_map).fillna(0).astype(int)
        x_test["clean_title"] = x_test["clean_title"].astype(str).str.lower().str.strip().map(title_map).fillna(0).astype(int)

    # Transmission simplification
    if "transmission" in x_train.columns and x_train["transmission"].dtype == "object":
        def _simplify_trans(trans):
            if pd.isna(trans) or str(trans).strip() in ["–", "2", "F", "", "nan", "None"]:
                return "Automatic"
            t = str(trans).lower().strip()
            if any(k in t for k in ["dual shift", "manual mode", "cmdshft", "dual-clutch", "pdk", "dct", "at/mt", "both"]):
                return "Both"
            if any(k in t for k in ["m/t", "manual", "6-speed manual", "7-speed manual", "8-speed manual"]):
                return "Manual"
            return "Automatic"
        x_train["transmission"] = x_train["transmission"].apply(_simplify_trans)
        x_test["transmission"] = x_test["transmission"].apply(_simplify_trans)

    # Color simplification
    for col in ["ext_col", "int_col"]:
        if col in x_train.columns and x_train[col].dtype == "object":
            def _simplify_color(color):
                if pd.isna(color) or str(color).strip() in ["–", "c / c", "Custom Color", "Metallic", "", "nan", "None"]:
                    return "Other"
                c = str(color).lower().strip(". ")
                if any(w in c for w in ["black", "ebony", "noir", "obsidian", "night", "nero", "onyx", "midnight", "raven", "blk", "beluga"]):
                    return "Black"
                if any(w in c for w in ["white", "bianco", "alpine", "snow", "ivory", "glacier", "pearl", "frost", "arctic", "linen", "cream"]):
                    return "White"
                if any(w in c for w in ["silver", "gray", "grey", "metallic", "graphite", "platinum", "charcoal", "slate", "steel", "titanium", "ash", "grigio", "pewter"]):
                    return "Silver/Gray"
                if any(w in c for w in ["blue", "blu", "navy", "aqua", "cyan", "nautical", "horizon", "sky"]):
                    return "Blue"
                if any(w in c for w in ["red", "rosso", "maroon", "burgundy", "crimson", "scarlet", "ruby", "cherry", "garnet", "magma", "rioja"]):
                    return "Red"
                if any(w in c for w in ["brown", "mocha", "coffee", "chocolate", "bronze", "copper", "saddle", "caramel", "cocoa", "walnut", "espresso"]):
                    return "Brown"
                if any(w in c for w in ["beige", "tan", "sand", "sandstone", "wheat", "champagne", "parchment", "camel", "almond", "oyster"]):
                    return "Beige/Tan"
                if any(w in c for w in ["green", "verde", "emerald", "forest", "moss", "jungle", "olive"]):
                    return "Green"
                if any(w in c for w in ["orange", "mango", "arancio", "sunset"]):
                    return "Orange"
                if any(w in c for w in ["yellow", "gold", "hellayella"]):
                    return "Yellow/Gold"
                return "Other"
            x_train[col] = x_train[col].apply(_simplify_color)
            x_test[col] = x_test[col].apply(_simplify_color)

    return x_train, x_test, extracted_cols

# Step 4 — Outlier Treatment (IQR Capping on Features AND Target)
def cap_outliers(x_train, x_test, y_train=None, y_test=None, eda_summary=None, task_type="Regression"):
    outlier_report = eda_summary.get("outlier_report", {}) if eda_summary else {}
    capping_bound = {}
    
    # Feature outlier capping
    for col, stats in outlier_report.items():
        if col in x_train.columns and stats.get("percentage", 0) > 0 and pd.api.types.is_numeric_dtype(x_train[col]):
            q1, q3 = x_train[col].quantile(0.25), x_train[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            affected_rows = int(((x_train[col] < lower) | (x_train[col] > upper)).sum())
            logger.info(f"Capped feature '{col}' - bounds: ({lower:.2f}, {upper:.2f}) | Affected train rows: {affected_rows}")
            capping_bound[col] = (float(lower), float(upper))
            x_train[col] = x_train[col].clip(lower=lower, upper=upper)
            if col in x_test.columns:
                x_test[col] = x_test[col].clip(lower=lower, upper=upper)

    # Target outlier treatment (Only clip extreme impossible outliers / typos, preserving valid high-end distributions)
    if task_type == "Regression" and y_train is not None:
        target_col = eda_summary.get("target_col", "target") if eda_summary else "target"
        y_min = float(y_train.min())
        # If target has extreme non-positive values when it's strictly positive, or extreme 99.9th percentile typos
        if (y_train > 0).all():
            y_upper = float(y_train.quantile(0.999))
            y_lower = float(y_train.quantile(0.001))
            capping_bound[f"TARGET_{target_col}"] = (y_lower, y_upper)
            y_train = y_train.clip(lower=y_lower, upper=y_upper)
            if y_test is not None:
                y_test = y_test.clip(lower=y_lower, upper=y_upper)

    return x_train, x_test, y_train, y_test, capping_bound

# AI Grouping
def ai_group_column(series: pd.Series, col_name: str, client: OpenAI, min_unique: int = 7, max_unique: int = 300) -> dict | None:
    """
    Sends unique values of a column to OpenAI API.
    Returns a mapping dict {original_value: group_label} or None if it fails.
    """
    unique_vals = series.dropna().unique().tolist()

    # Skip if too few unique values (already clean) or too many (model/engine)
    if len(unique_vals) < min_unique or len(unique_vals) > max_unique:
        logger.info(f"'{col_name}' skipped for AI grouping | unique count: {len(unique_vals)}")
        return None

    unique_vals_str = json.dumps(unique_vals[:150])  # cap to avoid token overflow

    prompt = f"""You are a data preprocessing expert.

Column name: "{col_name}"
Unique values: {unique_vals_str}

Group these values into meaningful semantic categories.
Rules:
- Create between 5 and 10 groups maximum
- Every value must be assigned to exactly one group
- Use clear, simple group names (e.g. "Black", "Automatic", "Gasoline")
- Unknown, missing, or ambiguous values → group as "Unknown"
- Return ONLY a valid JSON object mapping each original value to its group
- No explanation, no markdown, no extra text. Pure JSON only.

Example format:
{{"Jet Black": "Black", "Pearl White": "White", "–": "Unknown"}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",        # cheap + smart enough for grouping tasks
            messages=[
                {
                    "role": "system",
                    "content": "You are a data preprocessing expert. You only return valid JSON. No markdown, no explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,              # 0 = deterministic, no creativity needed here
            max_tokens=2000,
            response_format={"type": "json_object"}   # forces valid JSON output
        )

        raw = response.choices[0].message.content.strip()
        mapping = json.loads(raw)
        logger.info(f"'{col_name}' AI grouped | {len(unique_vals)} → {len(set(mapping.values()))} groups")
        return mapping
    except Exception as e:
        logger.info(f"AI grouping failed for '{col_name}': {e}")
        return None

def decide_columns_for_ai_grouping(X_train: pd.DataFrame, eda_summary: dict,
                                    min_unique: int = 7,
                                    max_unique: int = 300) -> list:
    """
    Automatically decides which columns benefit from AI semantic grouping.
    Reads directly from eda_summary — no manual column list needed.

    A column is selected if:
    - It is categorical (object/category dtype)
    - Its unique count is between min_unique and max_unique
    - It is in high cardinality bucket (already flagged by EDA)
    """
    # ── Get high cardinality cols flagged by EDA ───────────────────
    high_cardinality_cols = eda_summary["cardinality_report"].get(
        "High Cardinality Prefer Target / Feature Encoding", []
    )

    # ── Also consider low-cardinality cols that still have ────────
    # ── enough unique values to benefit from grouping     ────────
    low_cardinality_cols = eda_summary["cardinality_report"].get(
        "Low Cardinality Prefer One Hot", []
    )

    candidates = []

    for col in X_train.columns:
        # Must be categorical dtype
        if X_train[col].dtype not in ["object", "category", "string"] and \
            not pd.api.types.is_string_dtype(X_train[col]):
                continue

        unique_count = X_train[col].nunique()

        # Must be within groupable range
        if unique_count < min_unique or unique_count > max_unique:
            logger.info(
                f"'{col}' skipped for AI grouping | "
                f"unique count {unique_count} outside range [{min_unique}, {max_unique}]"
            )
            continue

        candidates.append(col)
        logger.info(
            f"'{col}' selected for AI grouping | "
            f"unique count: {unique_count}"
        )

    logger.info(
        f"AI grouping candidates: {candidates} | "
        f"total: {len(candidates)} columns"
    )
    return candidates

def apply_ai_grouping(X_train: pd.DataFrame, X_test: pd.DataFrame,
                      eda_summary: dict,
                      api_key: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fully automated AI grouping — no manual column list needed.
    Decides which columns to group by reading eda_summary directly.
    """
    # ── Automatically decide which columns need grouping ───────────
    columns_to_group = decide_columns_for_ai_grouping(X_train, eda_summary)

    if not columns_to_group:
        logger.info("No columns selected for AI grouping — skipping")
        return X_train, X_test, {}

    client = OpenAI(api_key=api_key)
    all_mappings = {}

    for col in columns_to_group:
        logger.info(f"Running AI grouping on: '{col}'")
        mapping = ai_group_column(X_train[col], col, client)

        if mapping is None:
            logger.info(f"'{col}' grouping returned None — keeping original values")
            continue

        # ── Validate mapping covers all unique values ──────────────
        original_vals = set(X_train[col].dropna().unique())
        mapped_vals   = set(mapping.keys())
        unmapped      = original_vals - mapped_vals
        if unmapped:
            logger.info(
                f"'{col}' has {len(unmapped)} unmapped values → "
                f"defaulting to 'Unknown': {list(unmapped)[:10]}"
            )

        # ── Apply to both train and test ───────────────────────────
        X_train[col] = X_train[col].map(mapping).fillna("Unknown")
        X_test[col]  = X_test[col].map(mapping).fillna("Unknown")
        all_mappings[col] = mapping

        logger.info(
            f"'{col}' grouped | "
            f"unique before: {len(original_vals)} → "
            f"unique after: {X_train[col].nunique()}"
        )

    logger.info(
        f"AI grouping complete | "
        f"columns processed: {list(all_mappings.keys())}"
    )
    return X_train, X_test, all_mappings

# Step 5 — Categorical Encoding
def target_encode_column(X_train: pd.DataFrame, X_test: pd.DataFrame,
                         col: str, y_train: pd.Series,
                         n_splits: int = 5,
                         smoothing: float = 1.0) -> tuple:
    """
    Out-of-fold target encoding — no leakage.
    Each row in X_train is encoded using target stats from OTHER folds only.
    X_test is encoded using full X_train stats.
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    global_mean  = float(y_train.mean())
    oof_encoded  = np.zeros(len(X_train))
    kf           = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ── Out-of-fold encoding for X_train ──────────────────────────
    for train_idx, val_idx in kf.split(X_train):
        fold_X_tr  = X_train[col].iloc[train_idx]
        fold_y_tr  = y_train.iloc[train_idx]
        fold_X_val = X_train[col].iloc[val_idx]

        stats    = fold_y_tr.groupby(fold_X_tr).agg(["count", "mean"])
        smoothed = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )
        oof_encoded[val_idx] = (
            fold_X_val.map(smoothed).fillna(global_mean).values
        )

    # ── Full-train mapping for X_test ─────────────────────────────
    # Must be computed BEFORE overwriting X_train[col] with numbers
    full_stats    = y_train.groupby(X_train[col]).agg(["count", "mean"])
    full_smoothed = (
        (full_stats["count"] * full_stats["mean"] + smoothing * global_mean)
        / (full_stats["count"] + smoothing)
    )

    # ── Apply ─────────────────────────────────────────────────────
    X_train[col] = oof_encoded
    X_test[col]  = X_test[col].map(full_smoothed).fillna(global_mean)

    encoding_map = full_smoothed.to_dict()
    encoding_map["__global_mean__"] = global_mean

    return X_train, X_test, encoding_map

def encode_categoricals(x_train: pd.DataFrame, x_test: pd.DataFrame,
                        eda_summary: dict, target_col: str,
                        y_train: pd.Series, task_type: str):
    x_train = x_train.copy()
    x_test = x_test.copy()
    fitted_encoders = {}
    datetime_cols = eda_summary.get("column_types", {}).get("datetime", [])
    boolean_cols  = eda_summary.get("column_types", {}).get("boolean", [])
    categorical_col = x_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for col in categorical_col:
        if col in datetime_cols:
            logger.info(f"'{col}' skipped — datetime, handled in feature extraction")
            continue
        if col in boolean_cols:
            logger.info(f"'{col}' skipped — boolean, already encoded")
            continue
        # Edge case: If a column is completely empty, skip it to avoid errors
        if x_train[col].dropna().empty:
            continue
        unique_count = x_train[col].nunique()
        logger.info(f"Encoding '{col}' | unique values: {unique_count}")
        if unique_count <= 30:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
            # Reset index before concat to avoid NaN misalignment
            x_train = x_train.reset_index(drop=True)
            x_test  = x_test.reset_index(drop=True)
            train_encoded = ohe.fit_transform(x_train[[col]])
            test_encoded = ohe.transform(x_test[[col]])
            feature_name = ohe.get_feature_names_out([col])
            train_encoded_df = pd.DataFrame(train_encoded, columns=feature_name, index=x_train.index)
            test_encoded_df = pd.DataFrame(test_encoded, columns=feature_name, index=x_test.index)
            x_train = pd.concat([x_train.drop(columns=[col]), train_encoded_df], axis=1)
            x_test = pd.concat([x_test.drop(columns=[col]), test_encoded_df], axis=1)
            fitted_encoders[col] = {"type": "ohe", "encoder": ohe}
            logger.info(f"  → OneHot | created {len(feature_name)} columns")
        else:
            if task_type == "Clustering":
                # frequency encoding
                freq_map = (x_train[col].value_counts(normalize=True)).to_dict()
                x_train[col] = x_train[col].map(freq_map).fillna(0)
                x_test[col] = x_test[col].map(freq_map).fillna(0)
                fitted_encoders[col] = {
                    "type": "frequency",
                    "map": freq_map
                }
                logger.info(f"  → Frequency Encoding | {unique_count} → numeric")
            else:
                # Target Encoding
                x_train, x_test, enc_map = target_encode_column(
                    x_train, x_test, col, y_train
                )
                fitted_encoders[col] = {
                    "type": "target",
                    "map": enc_map
                }
                logger.info(f"  → Target Encoding (OOF) | {unique_count} → numeric")
    logger.info(f"\nEncoding complete | columns encoded: {len(fitted_encoders)}")
    logger.info(f"  OHE columns    : {[c for c,v in fitted_encoders.items() if v['type']=='ohe']}")
    logger.info(f"  Target encoded : {[c for c,v in fitted_encoders.items() if v['type']=='target']}")
    logger.info(f"  Freq encoded   : {[c for c,v in fitted_encoders.items() if v['type']=='frequency']}")
    return x_train, x_test, fitted_encoders

# Step 6 — Datetime Feature Extraction
def extract_datetime_features(x_train, x_test, eda_summary):
    dataTimeCol = eda_summary.get("column_types", {}).get("datetime", [])
    x_train = x_train.copy()
    x_test = x_test.copy()
    dt_references = {}
    for col in dataTimeCol:
        if col not in x_train.columns:
            continue
        x_train[col] = pd.to_datetime(x_train[col])
        x_test[col] = pd.to_datetime(x_test[col])
        reference_data = x_train[col].min()
        dt_references[col] = str(reference_data)
        for df_ in [x_train, x_test]:
            df_[f'{col}_year'] = df_[col].dt.year
            df_[f'{col}_month'] = df_[col].dt.month
            df_[f'{col}_day'] = df_[col].dt.day
            df_[f'{col}_dayofweek'] = df_[col].dt.dayofweek
            df_[f'{col}_is_weekend'] = (df_[col].dt.dayofweek >= 5).astype(int)
            df_[f'{col}_days_since'] = (df_[col] - reference_data).dt.days
        x_train = x_train.drop(columns=[col])
        x_test = x_test.drop(columns=[col])
        logger.info(f"'{col}' expanded → 6 new features | reference: {reference_data}")
    return x_train, x_test, dt_references


# Step 7 — Feature Scaling
def scale_features(x_train, x_test, eda_summary):
    # 1. Safely identify column types
    highly_skewed_cols = eda_summary.get("highly_skewed_cols", [])
    column_types = eda_summary.get("column_types", {})
    all_numerical = column_types.get("numerical_continuous", []) + column_types.get("numerical_discrete", [])

    # Store original columns and index to reconstruct the DataFrame later
    train_cols = x_train.columns.tolist()

    # Filter columns to ensure we only process what is actually inside x_train
    highly_skewed_cols = [col for col in highly_skewed_cols if col in train_cols]
    normal_skew_cols = [col for col in all_numerical if col not in highly_skewed_cols and col in train_cols]

    # Identify the remainder (categorical/boolean columns that pass through untouched)
    passthrough_cols = [col for col in train_cols if col not in highly_skewed_cols and col not in normal_skew_cols]

    # 2. Determine the EXACT output order of the ColumnTransformer
    output_col_order = highly_skewed_cols + normal_skew_cols + passthrough_cols

    # 3. Create the Transformers
    skewed_pipeline = Pipeline(steps=[
        ('log1p', FunctionTransformer(np.log1p, validate=False)),
        ('minmax', MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('skewed', skewed_pipeline, highly_skewed_cols),
            ('normal', StandardScaler(), normal_skew_cols)
        ],
        remainder='passthrough'
    )

    # 4. Fit and Transform
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    # THE FIX: Force the output into a raw, nameless NumPy array to prevent Pandas from creating NaNs
    x_train_raw_array = np.asarray(x_train_transformed)
    x_test_raw_array = np.asarray(x_test_transformed)

    # 5. Manually rebuild the Pandas DataFrames
    x_train_scaled = pd.DataFrame(
        x_train_raw_array,
        columns=output_col_order,
        index=x_train.index
    )
    x_test_scaled = pd.DataFrame(
        x_test_raw_array,
        columns=output_col_order,
        index=x_test.index
    )

    # 6. Restore the exact original column order
    x_train_scaled = x_train_scaled[train_cols]
    x_test_scaled = x_test_scaled[train_cols]

    return x_train_scaled, x_test_scaled, preprocessor


# Step 8 — Handle Class Imbalance (Classification Only)
def handle_class_imbalance(x_train, y_train, eda_summary, task_type):
    if task_type != "Classification":
        logger.info(f"Skipping class imbalance — task is {task_type}")
        return x_train, y_train
    target_analysis = eda_summary.get("target_analysis", {})
    is_imbalanced = target_analysis.get("is_imbalanced", False)
    if not is_imbalanced:
        logger.info("Classes are balanced — no resampling needed")
        return x_train, y_train
    imbalance_ratio = target_analysis.get("imbalance_ratio", 1.0)
    logger.info(f"Class imbalance detected | ratio: {imbalance_ratio:.2f}")
    logger.info(f"Class distribution before resampling:\n{y_train.value_counts()}")
    if imbalance_ratio <= 10:
        # Use Smote
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=42)
        strategy = "SMOTE"
    else:
        from imblearn.combine import SMOTETomek
        sampler = SMOTETomek(random_state=42)
        strategy = "SMOTETomek"
    x_train_res, y_train_res = sampler.fit_resample(x_train, y_train)
    logger.info(f"Strategy used: {strategy}")
    logger.info(f"Class distribution after resampling:\n{y_train_res.value_counts()}")
    logger.info(f"Shape before: {x_train.shape} → after: {x_train_res.shape}")
    return x_train_res, y_train_res


# Step 9 — VIF Check (Multicollinearity Removal
def remove_high_vif_features(x_train, x_test, threshold=10.0):
    x_train = x_train.copy()
    x_test = x_test.copy()

    numerical_col = x_train.select_dtypes(include=["number", "int64", "float64", "float"]).columns.tolist()
    dropped_features = []
    x_train[numerical_col] = x_train[numerical_col].replace([float("inf"), float("-inf")], pd.NA)
    for col in numerical_col:
        if x_train[col].isna().any():
            median_val = x_train[col].median()
            x_train[col] = x_train[col].fillna(median_val)
            x_test[col] = x_test[col].replace([float("inf"), float("-inf")], pd.NA).fillna(median_val)

    while True:
        if len(numerical_col) == 0:
            break
        # 1. Add the constant for correct math
        x_calc = add_constant(x_train[numerical_col])
        vif_data = []

        # 2. Iterate through columns, skipping the 'const' column we just added
        for i in range(x_calc.shape[1]):
            col_name = x_calc.columns[i]
            if col_name == 'const':
                continue
            vif_score = variance_inflation_factor(x_calc.values, i)
            vif_data.append((col_name, vif_score))
        # Sort by highest VIF
        vif_data.sort(key=lambda x: x[1], reverse=True)
        max_col, max_vif = vif_data[0]

        # 3. Handle the NaN warning (Constant columns like clean_title)
        if pd.isna(max_vif):
            print(f"Dropping '{max_col}' | VIF Score: NaN (Zero variance / Constant column)")
            numerical_col.remove(max_col)
            dropped_features.append(max_col)
            x_train = x_train.drop(columns=[max_col])
            x_test = x_test.drop(columns=[max_col])
            continue # Restart the loop
        # 4. Standard threshold check with core feature protection
        protected = {"hp", "displacement", "cylinders", "engine_tier", "model_year", "milage", "mileage", "year"}
        if max_vif > threshold:
            if max_col in protected:
                # Do not drop primary domain specs; remove from calculation list and continue checking other cols
                numerical_col.remove(max_col)
                continue
            logger.info(f"Dropping '{max_col}' | VIF Score: {max_vif:.2f}")
            numerical_col.remove(max_col)
            dropped_features.append(max_col)

            x_train = x_train.drop(columns=[max_col])
            x_test = x_test.drop(columns=[max_col])
        else:
            break
    logger.info(f"Total features dropped due to multicollinearity: {len(dropped_features)}")
    return x_train, x_test, dropped_features

# Step 10 — Save Preprocessor Artifacts
def save_preprocessor_artifacts(
    imputers: dict,
    encoders: dict,
    scaler_pipeline,
    outlier_bounds: dict,
    dt_references: dict,
    dropped_columns: list,
    ai_mappings: dict,
    vif_dropped: list,
    task_type: str,
    imbalance_strategy: str,
    output_dir: str = "artifacts/preprocessor"
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Save fitted objects with joblib ───────────────────────────
    joblib.dump(imputers,        out / "imputers.joblib")
    joblib.dump(encoders,        out / "encoders.joblib")
    joblib.dump(scaler_pipeline, out / "scaler_pipeline.joblib")
    joblib.dump(outlier_bounds,  out / "outlier_bounds.joblib")
    joblib.dump(dt_references,   out / "datetime_references.joblib")
    joblib.dump(dropped_columns + vif_dropped, out / "dropped_columns.joblib")
    joblib.dump(ai_mappings,     out / "ai_mappings.joblib")

    logger.info(f"Saved all joblib artifacts to {out.resolve()}")

    # ── Build human-readable summary ──────────────────────────────
    def _encoder_summary(encoders):
        return {
            col: info["type"]
            for col, info in encoders.items()
        }

    def _imputer_summary(imputers):
        return {
            col: imp.strategy
            for col, imp in imputers.items()
        }

    def _bounds_summary(bounds):
        return {
            col: {"lower": round(float(lo), 4), "upper": round(float(hi), 4)}
            for col, (lo, hi) in bounds.items()
        }

    summary = {
        "imputation": _imputer_summary(imputers),
        "encoding":   _encoder_summary(encoders),
        "scaling": {
            "skewed_cols_log1p_minmax": [
                t[2] for t in scaler_pipeline.transformers
                if t[0] == "skewed"
            ] if hasattr(scaler_pipeline, "transformers") else "unknown",
            "normal_cols_standard": [
                t[2] for t in scaler_pipeline.transformers
                if t[0] == "normal"
            ] if hasattr(scaler_pipeline, "transformers") else "unknown",
        },
        "outlier_capping":        _bounds_summary(outlier_bounds),
        "ai_grouping_cols":       list(ai_mappings.keys()),
        "quasi_constant_dropped": dropped_columns,
        "vif_dropped":            vif_dropped,
        "datetime_references":    dt_references,
        "class_imbalance_strategy": imbalance_strategy,
        "task_type":              task_type,
    }

    summary_path = out / "preprocessor_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Saved preprocessor_summary.json to {summary_path.resolve()}")
    return summary


# Step 11 — Single Entry Point
def run_preprocessor(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    eda_summary: dict,
    api_key: str = None,
    output_dir: str = "artifacts/preprocessor"
):
    logger.info("=" * 60)
    logger.info("PREPROCESSOR STARTED")
    logger.info("=" * 60)

    # ── Step 1 — Split ────────────────────────────────────────────
    logger.info("STEP 1 — Train-Test Split")
    x_train, x_test, y_train, y_test = split_data(df, target_col, task_type)
    if x_train is None:
        logger.error("Split failed — aborting preprocessor")
        return None

    # ── Step 2 — Drop useless columns ────────────────────────────
    logger.info("STEP 2 — Dropping Useless Columns")
    x_train, x_test, dropped_cols = drop_useless_columns(x_train, x_test, eda_summary)

    # ── Step 3 — Impute missing values ────────────────────────────
    logger.info("STEP 3 — Imputing Missing Values")
    x_train, x_test, imputers = impute_missing_values(x_train, x_test, eda_summary)

    # ── Step 3.5 — Domain & Spec Feature Extraction ───────────────
    logger.info("STEP 3.5 — Extracting Domain & Spec Features")
    x_train, x_test, extracted_domain_cols = extract_domain_features(x_train, x_test)

    # ── AI Grouping (between Step 3 and Step 5) ───────────────────
    logger.info("AI GROUPING — Semantic Category Reduction")
    if api_key:
        x_train, x_test, ai_mappings = apply_ai_grouping(
            x_train, x_test, eda_summary, api_key
        )
    else:
        ai_mappings = {}
        logger.warning("No API key — skipping AI grouping")

    # ── Step 4 — Cap outliers ─────────────────────────────────────
    logger.info("STEP 4 — Capping Outliers (Features & Target)")
    x_train, x_test, y_train, y_test, outlier_bounds = cap_outliers(
        x_train, x_test, y_train, y_test, eda_summary, task_type=task_type
    )

    # ── Step 5 — Encode categoricals ──────────────────────────────
    logger.info("STEP 5 — Encoding Categorical Columns")
    x_train, x_test, encoders = encode_categoricals(
        x_train, x_test, eda_summary, target_col, y_train, task_type
    )

    # ── Step 6 — Datetime features ────────────────────────────────
    logger.info("STEP 6 — Extracting Datetime Features")
    x_train, x_test, dt_references = extract_datetime_features(
        x_train, x_test, eda_summary
    )

    # ── Step 7 — Scale features ───────────────────────────────────
    logger.info("STEP 7 — Scaling Features")
    x_train, x_test, scaler_pipeline = scale_features(x_train, x_test, eda_summary)

    # ── Step 8 — Handle class imbalance ──────────────────────────
    logger.info("STEP 8 — Handling Class Imbalance")
    x_train, y_train = handle_class_imbalance(x_train, y_train, eda_summary, task_type)
    imbalance_strategy = (
        "SMOTE" if task_type == "Classification" and
        eda_summary.get("target_analysis", {}).get("is_imbalanced") else "none"
    )

    # ── Step 9 — VIF check ────────────────────────────────────────
    logger.info("STEP 9 — VIF Multicollinearity Check")
    x_train, x_test, vif_dropped = remove_high_vif_features(x_train, x_test)

    # ── Step 10 — Save artifacts ──────────────────────────────────
    logger.info("STEP 10 — Saving Preprocessor Artifacts")
    preprocessor_summary = save_preprocessor_artifacts(
        imputers        = imputers,
        encoders        = encoders,
        scaler_pipeline = scaler_pipeline,
        outlier_bounds  = outlier_bounds,
        dt_references   = dt_references,
        dropped_columns = dropped_cols,
        ai_mappings     = ai_mappings,
        vif_dropped     = vif_dropped,
        task_type       = task_type,
        imbalance_strategy = imbalance_strategy,
        output_dir      = output_dir
    )

    logger.info("=" * 60)
    logger.info("PREPROCESSOR COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

    return {
        "X_train":              x_train,
        "X_test":               x_test,
        "y_train":              y_train,
        "y_test":               y_test,
        "preprocessor_summary": preprocessor_summary
    }