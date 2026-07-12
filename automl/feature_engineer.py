import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, f_classif, SelectFromModel, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from automl.logger import get_logger
logger = get_logger("feature_engineer")


# Step 1
def generate_polynomial_features(x_train, x_test, eda_summary, degree=2,max_cols=10):
    candidates = eda_summary.get("column_types", {}).get("numerical_continuous", [])
    candidates = [col for col in candidates if col in x_train.columns]
    # Check minimum requirements
    if len(candidates) < 2:
        logger.info(f"Skipping PolynomialFeatures: Found {len(candidates)} valid continuous columns (need at least 2).")
        return x_train, x_test, None
    variances = x_train[candidates].var()
    top_cols = variances.sort_values(ascending=False).head(max_cols).index.tolist()
    logger.info(f"Generating polynomial features (degree={degree}) for {len(top_cols)} high-variance continuous columns...")
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    # fit on x_train and transform on x_test
    x_train_poly_arrr = poly.fit_transform(x_train[top_cols])
    x_test_poly_arrr = poly.transform(x_test[top_cols])
    new_feature_names = poly.get_feature_names_out(top_cols)
    # Reconstructing the dataset
    x_train_poly_df = pd.DataFrame(x_train_poly_arrr, columns=new_feature_names, index=x_train.index)
    x_test_poly_df = pd.DataFrame(x_test_poly_arrr, columns=new_feature_names, index=x_test.index)
    # we need to drop the original columns from X_train/X_test before merging the new ones in.
    x_train_dropped = x_train.drop(columns=top_cols)
    x_test_dropped = x_test.drop(columns=top_cols)
    x_train_final = pd.concat([x_train_dropped, x_train_poly_df], axis=1)
    x_test_final = pd.concat([x_test_dropped, x_test_poly_df], axis=1)
    new_cols_count = len(new_feature_names) - len(top_cols) # subtract the original columns to get the net *new* count
    logger.info(f"Successfully created {new_cols_count} new polynomial/interaction features.")
    logger.info(f"Old X_train shape: {x_train.shape} | New X_train shape: {x_train_final.shape}")
    return x_train_final, x_test_final, poly


# Step 2 — AI-Based Derived Feature Generation
def generate_ai_derived_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    eda_summary: dict,
    task_type: str,
    target_col: str,
    api_key: str,
    max_features: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Uses OpenAI GPT to analyze the dataset context and derive new meaningful
    features by combining existing columns. AI decides:
        1. Whether feature derivation is even needed for this dataset
        2. Which columns to combine
        3. What mathematical operation to apply
        4. Why the new feature is meaningful (domain reasoning)

    Only derives features if AI determines it will help. Skips if dataset
    doesn't benefit. Caps at max_features to avoid noise.

    Args:
        X_train      : Training features (post-preprocessing, post-polynomial)
        X_test       : Test features
        eda_summary  : Full EDA summary dict (provides context to AI)
        task_type    : "Regression" | "Classification" | "Clustering"
        target_col   : Name of the target column (None for Clustering)
        api_key      : OpenAI API key
        max_features : Hard cap on number of new features AI can derive

    Returns:
        X_train      : DataFrame with new columns appended (if any)
        X_test       : DataFrame with same new columns appended
        derived_cols : List of new column names created (empty if AI skips)
    """
    import json
    from openai import OpenAI

    logger.info("STEP 2 — AI-Based Derived Feature Generation")

    if not api_key:
        logger.info("No OpenAI API key provided — skipping AI feature derivation")
        return X_train, X_test, []

    # ── Build context snapshot for AI ─────────────────────────────────────────
    # Give AI only what it needs: column names, types, stats, task context.
    # Sending full data would blow the token limit and leak row-level info.

    numerical_cols = (
        eda_summary.get("column_types", {}).get("numerical_continuous", []) +
        eda_summary.get("column_types", {}).get("numerical_discrete", [])
    )
    categorical_cols = eda_summary.get("column_types", {}).get("categorical", [])
    target_corr_raw = eda_summary.get("correlation", {}).get("target_corr_ranked", {})
    if hasattr(target_corr_raw, "to_dict"):  # it's a pandas Series
        target_corr_raw = target_corr_raw.to_dict()

    # Filter to only columns that actually exist in X_train right now
    # (some may have been dropped by preprocessing or polynomial step)
    available_numerical = [c for c in numerical_cols if c in X_train.columns]
    available_categorical = [c for c in categorical_cols if c in X_train.columns]
    all_available = X_train.columns.tolist()

    # Build a compact stats snapshot — mean, std, min, max for numericals
    stats_snapshot = {}
    for col in available_numerical[:20]:  # cap at 20 to avoid token overflow
        stats_snapshot[col] = {
            "mean":   round(float(X_train[col].mean()), 4),
            "std":    round(float(X_train[col].std()), 4),
            "min":    round(float(X_train[col].min()), 4),
            "max":    round(float(X_train[col].max()), 4),
        }

    # Sample top categories for categorical columns
    cat_snapshot = {}
    for col in available_categorical[:10]:
        top_vals = X_train[col].value_counts().head(5).index.tolist()
        cat_snapshot[col] = top_vals

    context = {
        "task_type": task_type,
        "target_column": target_col,
        "all_columns": all_available[:50],           # cap to avoid token overflow
        "numerical_columns": available_numerical[:20],
        "categorical_columns": available_categorical[:10],
        "numerical_stats": stats_snapshot,
        "categorical_top_values": cat_snapshot,
        "n_rows": eda_summary.get("n_rows"),
        "highly_skewed_cols": eda_summary.get("highly_skewed_cols", []),
        "multicollinear_pairs": eda_summary.get("correlation", {}).get("multicollinear_pairs", [])[:5],
        "target_corr_ranked": target_corr_raw
    }

    # ── Build the prompt ───────────────────────────────────────────────────────
    prompt = f"""You are a senior data scientist and domain expert reviewing a dataset for a machine learning pipeline.

DATASET CONTEXT:
{json.dumps(context, indent=2)}

YOUR TASK:
Analyze the dataset and decide whether deriving new features by combining existing columns would meaningfully improve model performance.

RULES:
1. Only suggest a feature if it has clear domain logic — not just mathematical combinations.
2. Only use columns that exist in "all_columns" list above.
3. Each feature must be derivable using basic math: +, -, *, /, log, ratio, difference, or a conditional.
4. Maximum {max_features} features total. Fewer is better — only suggest if genuinely useful.
5. If the dataset does not benefit from derived features (e.g. already clean tabular data with clear independent columns), return an empty list.
6. Never combine a column with itself.
7. The formula must be executable as a pandas expression using column names.
8. For classification/regression: prioritize features that likely correlate with the target.
9. For clustering: prioritize features that capture meaningful group differences.

RESPONSE FORMAT — return ONLY a valid JSON object, no markdown, no explanation:

{{
  "should_derive": true or false,
  "reason": "one sentence explaining your decision",
  "features": [
    {{
      "name": "new_column_name",
      "formula": "col_a / (col_b + 1e-8)",
      "columns_used": ["col_a", "col_b"],
      "domain_reason": "brief explanation of why this combination is meaningful"
    }}
  ]
}}

If should_derive is false, return an empty features list.
Column names in formula must exactly match names in "all_columns".
Use 1e-8 in denominators to avoid division by zero."""

    # ── Call OpenAI API ────────────────────────────────────────────────────────
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data science expert. "
                        "You return only valid JSON. No markdown, no explanation outside the JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,                                    # deterministic
            max_tokens=1000,
            response_format={"type": "json_object"}          # force valid JSON
        )

        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)

    except Exception as e:
        logger.info(f"OpenAI API call failed for feature derivation: {e}")
        return X_train, X_test, []

    # ── Parse AI response ──────────────────────────────────────────────────────
    should_derive = result.get("should_derive", False)
    reason = result.get("reason", "")
    features = result.get("features", [])

    logger.info(f"AI decision — should_derive: {should_derive} | reason: {reason}")

    if not should_derive or not features:
        logger.info("AI determined no derived features are needed for this dataset — skipping")
        return X_train, X_test, []

    # ── Apply each derived feature ─────────────────────────────────────────────
    X_train = X_train.copy()
    X_test = X_test.copy()
    derived_cols = []

    for feature_spec in features[:max_features]:            # hard cap enforcement
        name = feature_spec.get("name", "").strip()
        formula = feature_spec.get("formula", "").strip()
        cols_used = feature_spec.get("columns_used", [])
        domain_reason = feature_spec.get("domain_reason", "")

        # ── Validate before applying ───────────────────────────────────────────
        if not name or not formula:
            logger.info(f"Skipping feature — missing name or formula: {feature_spec}")
            continue

        # Check all required columns exist in X_train
        missing_cols = [c for c in cols_used if c not in X_train.columns]
        if missing_cols:
            logger.info(
                f"Skipping '{name}' — columns not found in X_train: {missing_cols}"
            )
            continue

        # Check name doesn't clash with existing column
        if name in X_train.columns:
            logger.info(f"Skipping '{name}' — column already exists")
            continue

        # ── Apply formula safely using eval ───────────────────────────────────
        # We use a restricted namespace: only the DataFrame columns + numpy.
        # This prevents arbitrary code execution from the AI response.
        try:
            namespace = {col: X_train[col] for col in X_train.columns}
            namespace["np"] = __import__("numpy")

            train_values = eval(formula, {"__builtins__": {}}, namespace)

            # Validate output is numeric and same length as X_train
            train_series = pd.to_numeric(pd.Series(train_values, index=X_train.index), errors="coerce")

            if train_series.isna().mean() > 0.5:
                logger.info(
                    f"Skipping '{name}' — formula produced >50% NaN values "
                    f"(formula may be invalid: {formula})"
                )
                continue
            # Apply same formula to X_test
            test_namespace = {col: X_test[col] for col in X_test.columns}
            test_namespace["np"] = __import__("numpy")
            test_values = eval(formula, {"__builtins__": {}}, test_namespace)
            test_series = pd.to_numeric(pd.Series(test_values, index=X_test.index), errors="coerce")

            # Fill any NaN in new feature with train median (safe for test too)
            fill_val = float(train_series.median())
            train_series = train_series.fillna(fill_val)
            test_series = test_series.fillna(fill_val)

            # Append to DataFrames
            X_train[name] = train_series
            X_test[name] = test_series
            derived_cols.append(name)
            logger.info(
                f"Created '{name}' | formula: {formula} | "
                f"reason: {domain_reason} | "
                f"train NaN after fill: {X_train[name].isna().sum()}"
            )
        except Exception as e:
            logger.info(f"Failed to apply formula for '{name}': {formula} | error: {e}")
            continue
    logger.info(
        f"AI feature derivation complete | "
        f"requested: {len(features)} | "
        f"successfully created: {len(derived_cols)} | "
        f"columns: {derived_cols}"
    )
    return X_train, X_test, derived_cols



# Step 3
def generate_aggregate_features(x_train, x_test, eda_summary, task_type):
    x_train = x_train.copy()
    x_test = x_test.copy()
    if task_type == "Clustering":
        logger.info("Skipping Aggregate Features: Task type is Clustering.")
        return x_train,x_test,[]
    categorial_col = eda_summary.get("column_types", {}).get("categorical", [])
    categorial_col = [col for col in categorial_col if col in x_train.columns]
    numerical_cont = eda_summary.get("column_types", {}).get("numerical_continuous", [])
    numerical_cont = [col for col in numerical_cont if col in x_train.columns]
    if not categorial_col or not numerical_cont:
        logger.info("Skipping Aggregate Features: Missing categorical or numerical columns.")
        return x_train,x_test,[]
    new_col_names = []
    for cat_col in categorial_col:
        if x_train[cat_col].nunique() > 30:
            continue
        for num_col in numerical_cont:
            logger.info(f"Creating aggregate features for: {cat_col} -> {num_col}")
            group_stats = x_train.groupby(cat_col)[num_col].agg(['mean', 'std', 'min', 'max', 'count'])
            global_mean = x_train[num_col].mean()
            global_std = x_train[num_col].std()
            for stat in ['mean', 'std', 'min', 'max', 'count']:
                new_col_name = f"{cat_col}_{num_col}_{stat}"
                stat_lookup = group_stats[stat].to_dict()
                x_train[new_col_name] = x_train[cat_col].map(stat_lookup)
                x_test[new_col_name] = x_test[cat_col].map(stat_lookup)
                fill_value = global_std if stat == 'std' else global_mean
                if stat == 'count': fill_value = 0
                x_train[new_col_name] = x_train[new_col_name].fillna(fill_value)
                x_test[new_col_name] = x_test[new_col_name].fillna(fill_value)
                new_col_names.append(new_col_name)
    if len(new_col_names) > 50:
        logger.info(f"Capping aggregate features: Dropping {len(new_col_names) - 50} excess columns.")
        columns_to_drop = new_col_names[50:]
        x_train = x_train.drop(columns=columns_to_drop)
        x_test = x_test.drop(columns=columns_to_drop)
        new_col_names = new_col_names[:50]
    logger.info(f"Successfully created {len(new_col_names)} aggregate features.")
    return x_train, x_test, new_col_names



# Step 4 — select_features_variance
def select_features_variance(x_train, x_test, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(x_train)
    # get_support() returns an array like: [True, False, False, True]
    kept_mask = selector.get_support()
    removed_cols = x_train.columns[~kept_mask].tolist()
    kept_cols = x_train.columns[kept_mask].tolist()
    # Log removed columns and their variance scores
    logger.info("Dropped columns and their variances:")
    for idx, col in enumerate(x_train.columns):
        if not kept_mask[idx]:
            # selector.variances_ holds the variance for every original feature
            variance_score = selector.variances_[idx]
            logger.info(f" - {col}: {variance_score:.6f}")
    # transform return numpy array
    x_train_arr = selector.transform(x_train)
    x_test_arr = selector.transform(x_test)
    x_train_df = pd.DataFrame(x_train_arr, columns=kept_cols, index=x_train.index)
    x_test_df = pd.DataFrame(x_test_arr, columns=kept_cols, index=x_test.index)
    logger.info(f"Removed col: {removed_cols}")
    logger.info(f"\nOriginal X_train shape: {x_train.shape}")
    logger.info(f"New X_train shape: {x_train_df.shape}")
    return x_train_df, x_test_df, removed_cols


# Step 5 — select_features_univariate
def select_features_univariate(x_train, x_test, y_train, task_type, k="all"):
    if task_type == "Clustering":
        logger.info("Skipping Univariate Selection: Task type is Clustering (no target variable).")
        return x_train, x_test, list(x_train.columns), []
    if task_type == "Regression":
        score_function = f_regression
    elif task_type == "Classification":
        score_function = f_classif
    else:
        logger.info(f"Error: Unknown task type '{task_type}'. Skipping selection.")
        return x_train, x_test, list(x_train.columns), []
    selector = SelectKBest(score_function, k=k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(x_train, y_train)
    # (If a feature has perfectly 0 variance, sklearn sets its score to NaN. We fill those with 0).
    scores = np.nan_to_num(selector.scores_, nan=0.0)
    pvalues = np.nan_to_num(selector.pvalues_, nan=1.0)
    stats_df = pd.DataFrame({
        'Feature': x_train.columns,
        'Score': scores,
        'P_Value': pvalues
    })
    bottom_20 = stats_df.sort_values(by='Score', ascending=True).head(20)
    logger.info("\n--- Bottom 20 Weakest Features ---")
    logger.info(bottom_20.to_string(index=False))
    insignificant = stats_df[stats_df['P_Value'] > 0.05]
    if not insignificant.empty:
        logger.info(f"\nFlagged {len(insignificant)} features with p-value > 0.05 (Statistically Insignificant).")
    selected_cols = selector.get_feature_names_out(x_train.columns)
    x_train_selected = pd.DataFrame(selector.transform(x_train), columns=selected_cols, index=x_train.index)
    x_test_selected = pd.DataFrame(selector.transform(x_test), columns=selected_cols, index=x_test.index)
    dropped_count = len(x_train.columns) - len(selected_cols)
    if dropped_count > 0:
        logger.info(f"\nSummary: Successfully dropped {dropped_count} lowest-scoring features.")
    else:
        logger.info("\nSummary: Evaluated all features, none dropped (k='all').")
    scores_list = [
        {"Feature": col, "Score": round(float(s), 4), "P_Value": round(float(p), 4)}
        for col, s, p in zip(x_train.columns, scores, pvalues)
    ]
    return x_train_selected, x_test_selected, selected_cols, scores_list


# Step 6 - select_features_model_based
def select_features_model_based(x_train, x_test, y_train, task_type, threshold='median'):
    if task_type == "Clustering":
        return x_train, x_test, list(x_train.columns)

    # FIX 1: ExtraTrees (Plural)
    if task_type == "Regression":
        estimator = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif task_type == "Classification":
        estimator = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        logger.info(f"Error: Unknown task type '{task_type}'. Skipping selection.")
        return x_train, x_test, list(x_train.columns)

    selector = SelectFromModel(estimator=estimator, threshold=threshold)
    selector.fit(x_train, y_train)

    importance = selector.estimator_.feature_importances_
    kept_mask = selector.get_support()

    importance_df = pd.DataFrame({
        'Feature': x_train.columns,
        'Importance': importance,
        'Kept': kept_mask
    }).sort_values(by='Importance', ascending=False)

    # FIX 2: Matched 'Kept' (Capital K)
    kept_cols = importance_df[importance_df['Kept']]['Feature'].tolist()
    dropped_cols = importance_df[~importance_df['Kept']]['Feature'].tolist()

    logger.info("\n--- Top 10 Most Important Features (ExtraTrees) ---")
    logger.info(importance_df.head(10).to_string(index=False))
    logger.info(f"\nSummary: Kept {len(kept_cols)} features, Dropped {len(dropped_cols)} features.")

    # FIX 3: Matched x_train and x_test (lowercase x)
    x_train_selected = pd.DataFrame(
        selector.transform(x_train),
        columns=kept_cols,
        index=x_train.index
    )
    x_test_selected = pd.DataFrame(
        selector.transform(x_test),
        columns=kept_cols,
        index=x_test.index
    )
    return x_train_selected, x_test_selected, kept_cols


# Step 7 — select_features_rfe
def select_features_rfe(x_train, x_test, y_train, task_type, n_features_to_select=None):
    if task_type == "Clustering":
        logger.info("Skipping RFE Selection: Task type is Clustering (no target variable).")
        return x_train, x_test, list(x_train.columns)
    if x_train.shape[1] > 50:
        logger.warning(
            f"Skipping RFE: Too many features ({x_train.shape[1]}) — "
            f"model-based selection should have reduced this below 50 first."
        )
        return x_train, x_test, list(x_train.columns)
    if task_type == "Regression":
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        metric = 'r2'
    elif task_type == "Classification":
        estimator = LogisticRegression(max_iter=200, random_state=42)
        metric = 'f1_weighted'
    else:
        logger.info(f"Error: Unknown task type '{task_type}'. Skipping selection.")
        return x_train, x_test, list(x_train.columns)
    selector = RFECV(estimator=estimator, step=1, cv=5, scoring=metric)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(x_train, y_train)
    kept_mask = selector.get_support()
    kept_cols = x_train.columns[kept_mask].tolist()
    eliminated_cols = x_train.columns[~kept_mask].tolist()
    logger.info(f"\nOptimal number of features found by CV: {selector.n_features_}")
    logger.info(f"Eliminated {len(eliminated_cols)} features:")
    # Print the dropped features so the user/logs can see exactly what RFE thought was useless
    if eliminated_cols:
        logger.info(", ".join(eliminated_cols))
    else:
        logger.info("None")
    x_train_selected = pd.DataFrame(
        selector.transform(x_train),
        columns=kept_cols,
        index=x_train.index
    )
    x_test_selected = pd.DataFrame(
        selector.transform(x_test),
        columns=kept_cols,
        index=x_test.index
    )
    return x_train_selected, x_test_selected, kept_cols


# Step 8 — save_feature_engineering_artifacts
def save_feature_engineering_artifacts(
    poly_object,
    ai_derived_cols,
    aggregate_cols,
    variance_dropped,
    univariate_scores,
    model_based_kept,
    model_based_dropped,
    rfe_selected,
    final_feature_list,
    output_dir="artifacts/feature_engineer"
) -> dict:
    """
    Saves all fitted feature engineering objects and produces
    feature_engineering_summary.json consumed by model_trainer.

    Args:
        poly_object         : Fitted PolynomialFeatures object (or None if skipped)
        ai_derived_cols     : List of column names created by AI derivation
        aggregate_cols      : List of column names created by aggregate step
        variance_dropped    : List of columns removed by VarianceThreshold
        univariate_scores   : DataFrame-as-dict with Feature/Score/P_Value per column
        model_based_kept    : List of columns kept by ExtraTrees model-based selection
        model_based_dropped : List of columns dropped by ExtraTrees model-based selection
        rfe_selected        : List of columns surviving RFECV (final selected set)
        final_feature_list  : Exact column list in X_train passed to model_trainer
        output_dir          : Folder to save artifacts into

    Returns:
        summary dict (same content as feature_engineering_summary.json)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save fitted objects needed at inference time ───────────────────────────
    # poly_object is needed to apply the same polynomial expansion to new data
    if poly_object is not None:
        joblib.dump(poly_object, out_dir / "poly_features.joblib")
        logger.info(f"Saved poly_features.joblib")
    else:
        logger.info("No polynomial object to save (step was skipped)")

    # feature_selection_mask is the final column list — inference applies this
    # to drop everything that didn't survive the full selection funnel
    joblib.dump(final_feature_list, out_dir / "feature_selection_mask.joblib")
    logger.info(f"Saved feature_selection_mask.joblib | {len(final_feature_list)} features")

    # ── Build summary dict ─────────────────────────────────────────────────────
    summary = {
        "ai_derived_features":      ai_derived_cols,
        "aggregate_features":       aggregate_cols,
        "variance_dropped":         variance_dropped,
        "univariate_scores":        univariate_scores,   # list of {Feature, Score, P_Value}
        "model_based_kept":         model_based_kept,
        "model_based_dropped":      model_based_dropped,
        "rfe_selected":             rfe_selected,
        "final_feature_list":       final_feature_list,
        "final_feature_count":      len(final_feature_list),
    }

    # ── Write JSON ─────────────────────────────────────────────────────────────
    with open(out_dir / "feature_engineering_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved feature_engineering_summary.json to {out_dir.resolve()}")

    return summary


# Step 9 — run_feature_engineering (single entry point)
def run_feature_engineering(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    target_col: str,
    eda_summary: dict,
    api_key: str = None,
    output_dir: str = "artifacts/feature_engineer"
) -> dict:
    """
    Single entry point for the full feature engineering pipeline.
    Calls all steps in order and returns final engineered datasets.

    Args:
        X_train     : Preprocessed training features
        X_test      : Preprocessed test features
        y_train     : Training target (None for Clustering)
        y_test      : Test target (None for Clustering) — passed through unchanged
        task_type   : "Regression" | "Classification" | "Clustering"
        target_col  : Name of target column (None for Clustering)
        eda_summary : Full EDA summary dict
        api_key     : OpenAI API key (None skips AI step)
        output_dir  : Where to save artifacts

    Returns:
        {
            "X_train": df,
            "X_test":  df,
            "y_train": series,
            "y_test":  series,
            "feature_engineering_summary": dict
        }
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING STARTED")
    logger.info("=" * 60)
    logger.info(f"Input shape — X_train: {X_train.shape} | X_test: {X_test.shape}")

    # ── Step 1 — Polynomial Features ──────────────────────────────────────────
    logger.info("STEP 1 — Polynomial Feature Generation")
    X_train, X_test, poly_object = generate_polynomial_features(
        X_train, X_test, eda_summary
    )
    logger.info(f"After polynomial — X_train: {X_train.shape}")

    # ── Step 2 — AI Derived Features ──────────────────────────────────────────
    logger.info("STEP 2 — AI-Based Feature Derivation")
    X_train, X_test, ai_derived_cols = generate_ai_derived_features(
        X_train, X_test, eda_summary,
        task_type=task_type,
        target_col=target_col,
        api_key=api_key
    )
    logger.info(f"After AI derivation — X_train: {X_train.shape}")

    # ── Step 3 — Aggregate Features ───────────────────────────────────────────
    logger.info("STEP 3 — Aggregate Feature Generation")
    X_train, X_test, aggregate_cols = generate_aggregate_features(
        X_train, X_test, eda_summary, task_type
    )
    logger.info(f"After aggregate — X_train: {X_train.shape}")

    # ── Step 4 — Variance Selection ───────────────────────────────────────────
    logger.info("STEP 4 — Variance Threshold Selection")
    X_train, X_test, variance_dropped = select_features_variance(X_train, X_test)
    logger.info(f"After variance filter — X_train: {X_train.shape}")

    # ── Step 5 — Univariate Selection ─────────────────────────────────────────
    logger.info("STEP 5 — Univariate Feature Selection")
    X_train, X_test, univariate_selected, univariate_scores = select_features_univariate(
        X_train, X_test, y_train, task_type
    )
    logger.info(f"After univariate — X_train: {X_train.shape}")

    # ── Step 6 — Model-Based Selection ────────────────────────────────────────
    logger.info("STEP 6 — Model-Based Feature Selection")
    X_train, X_test, model_based_kept = select_features_model_based(
        X_train, X_test, y_train, task_type
    )
    # Derive dropped list by comparing kept to what entered Step 6
    model_based_dropped = [
        col for col in univariate_selected if col not in model_based_kept
    ]
    logger.info(f"After model-based — X_train: {X_train.shape}")

    # ── Step 7 — RFE Selection ────────────────────────────────────────────────
    logger.info("STEP 7 — Recursive Feature Elimination (RFECV)")

    # Minimum feature guard — RFE is meaningless with <5 features
    if X_train.shape[1] < 5:
        logger.warning(
            f"Only {X_train.shape[1]} features remaining — "
            f"skipping RFE to avoid over-reduction"
        )
        rfe_selected = list(X_train.columns)
    else:
        X_train, X_test, rfe_selected = select_features_rfe(
            X_train, X_test, y_train, task_type
        )
    logger.info(f"After RFE — X_train: {X_train.shape}")

    # ── Final feature list ─────────────────────────────────────────────────────
    final_feature_list = list(X_train.columns)

    # ── Feature funnel summary log ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING FUNNEL SUMMARY")
    logger.info(f"  AI derived features created  : {len(ai_derived_cols)}")
    logger.info(f"  Aggregate features created   : {len(aggregate_cols)}")
    logger.info(f"  Dropped by variance filter   : {len(variance_dropped)}")
    logger.info(f"  Dropped by model-based       : {len(model_based_dropped)}")
    logger.info(f"  Final feature count          : {len(final_feature_list)}")
    logger.info("=" * 60)

    # ── Step 8 — Save Artifacts ───────────────────────────────────────────────
    logger.info("STEP 8 — Saving Feature Engineering Artifacts")
    summary = save_feature_engineering_artifacts(
        poly_object=poly_object,
        ai_derived_cols=ai_derived_cols,
        aggregate_cols=aggregate_cols,
        variance_dropped=variance_dropped,
        univariate_scores=univariate_scores,
        model_based_kept=model_based_kept,
        model_based_dropped=model_based_dropped,
        rfe_selected=rfe_selected,
        final_feature_list=final_feature_list,
        output_dir=output_dir
    )

    logger.info("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")

    return {
        "X_train":                       X_train,
        "X_test":                        X_test,
        "y_train":                       y_train,
        "y_test":                        y_test,
        "feature_engineering_summary":   summary
    }