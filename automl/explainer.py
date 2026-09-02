import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from automl.logger import get_logger

logger = get_logger("Model Explainer")


try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning(
        "CRITICAL WARNING: 'shap' library is not installed. "
        "SHAP explanations will be unavailable."
    )

# Sentinel object/constants for clean short-circuiting in run_explainer
SKIP_SHAP_SENTINEL = "SKIP_SHAP"
CLUSTERING_PATH_SENTINEL = "CLUSTERING_PATH"
def _validate_explainer_inputs(final_best_model, X_train, X_test, task_type, feature_names) -> None:
    if not HAS_SHAP:
        logger.warning("SHAP library is missing. Skipping SHAP computation.")
        return SKIP_SHAP_SENTINEL
        # 2. Model Structure Validation
    if not isinstance(final_best_model, dict) or not final_best_model:
        raise RuntimeError("final_best_model must be a non-empty dictionary.")
    fitted_model = final_best_model.get("fitted_model")
    if fitted_model is None:
        raise RuntimeError(
            "final_best_model dictionary missing valid 'fitted_model' instance."
        )
    # 3. Model SHAP Support Check
    if not final_best_model.get("supports_shap", True):
        logger.warning(
            "Model does not support SHAP. Explainability will be skipped."
        )
        return SKIP_SHAP_SENTINEL
    # 4. Special Task Handling (Clustering)
    if task_type.lower() == "clustering":
        logger.info(
            "Clustering task detected — SHAP explanations will be replaced "
            "with cluster profile analysis."
        )
        return CLUSTERING_PATH_SENTINEL
    # 5. Data Structure & Match Checks
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_train and X_test must be valid, non-None DataFrames.")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test must have identical columns.")
    # 6. Feature Names Alignment & Fallbacks
    if not feature_names:
        logger.warning("feature_names list is empty. Falling back to X_train.columns.")
        feature_names.clear()
        feature_names.extend(X_train.columns.tolist())
    elif len(feature_names) != X_train.shape[1]:
        logger.error(
            f"Feature name count mismatch ({len(feature_names)}) vs data "
            f"shape ({X_train.shape[1]}). Feature engineering summary may be "
            "out of sync. Falling back to X_train.columns."
        )
        feature_names.clear()
        feature_names.extend(X_train.columns.tolist())
    return None


def _select_shap_explainer(model, model_key, model_type, X_train, task_type):
    # Rule 0: Unwrap TransformedTargetRegressor if present
    if hasattr(model, "regressor_"):
        model = model.regressor_

    # Rule 4: Clustering models (no SHAP)
    if task_type == "Clustering":
        return None, "cluster_profile"

    # Helper function for KernelExplainer edge cases
    def _get_predict_func():
        # Edge case: sklearn's GBM requires predict_proba for classification in KernelExplainer
        if task_type == "Classification" and "GradientBoosting" in type(model).__name__:
            if hasattr(model, "predict_proba"):
                return model.predict_proba
        return model.predict

    try:
        # Rule 1: Tree-based models
        tree_models = ["random_forest", "extra_trees", "gradient_boosting", "xgboost", "lightgbm"]
        if model_type == "ensemble" and model_key in tree_models:

            # Edge case: XGBoost (ensure probability outputs for classification instead of log-odds)
            if model_key == "xgboost" or "XGB" in type(model).__name__:
                return shap.TreeExplainer(model, model_output="raw"), "tree"

            # Edge case: LightGBM (requires interventional perturbation on certain setups)
            elif model_key == "lightgbm" or "LGBM" in type(model).__name__:
                try:
                    return shap.TreeExplainer(model), "tree"
                except Exception:
                    return shap.TreeExplainer(model, feature_perturbation="interventional"), "tree"

            # Standard tree models
            else:
                return shap.TreeExplainer(model), "tree"

        # Rule 2: Linear models
        linear_models = ["ridge", "lasso", "elastic_net", "logistic_regression", "ridge_classifier"]
        if model_type == "linear" and model_key in linear_models:
            masker = shap.maskers.Independent(X_train)
            return shap.LinearExplainer(model, masker=masker), "linear"

        # Rule 3: KNN and other non-linear models
        if model_type == "neighbor":
            background_data = shap.sample(X_train, 100)
            return shap.KernelExplainer(_get_predict_func(), background_data), "kernel"

        # Rule 5: Unknown model type (fallback)
        logger.warning(
            f"Unknown model type '{model_type}' (key: {model_key}) — falling back to KernelExplainer with 50 background samples.")
        background_data = shap.sample(X_train, 50)
        return shap.KernelExplainer(_get_predict_func(), background_data), "kernel"

    except Exception as e:
        # Global fallback: Catch pipeline failures or unsupported custom transformers
        logger.warning(
            f"Failed to initialize primary SHAP explainer for {model_key} ({str(e)}). "
            f"Falling back to KernelExplainer with 50 background samples."
        )
        background_data = shap.sample(X_train, 50)
    return shap.KernelExplainer(_get_predict_func(), background_data), "kernel"


def _compute_shap_values(explainer, explainer_type, X_train, X_test, task_type, model_key):
    """
    Computes global and local SHAP values while enforcing memory guards and
    handling multi-class/binary shapes appropriately.
    """

    # -----------------------------------------
    # 1. Memory Guards and Sampling
    # -----------------------------------------
    # Training Set (Global Explanations)
    if explainer_type == "tree" and len(X_train) > 5000:
        logger.info(f"TreeExplainer memory guard: Sampling 2000 / {len(X_train)} rows for global SHAP.")
        X_train_used = shap.sample(X_train, 2000)
    elif explainer_type == "kernel":
        logger.info("KernelExplainer: Sampling 200 rows maximum for global SHAP.")
        X_train_used = shap.sample(X_train, min(200, len(X_train)))
    else:
        X_train_used = X_train

    # Test Set (Local Explanations)
    if explainer_type == "kernel":
        logger.info("KernelExplainer: Sampling 100 rows maximum for local SHAP.")
        X_test_used = shap.sample(X_test, min(100, len(X_test)))
    else:
        X_test_used = X_test

    # -----------------------------------------
    # 2. Compute SHAP Values
    # -----------------------------------------
    logger.info(f"Computing global SHAP values using {explainer_type} explainer...")
    raw_shap_train = explainer.shap_values(X_train_used)

    logger.info(f"Computing local SHAP values using {explainer_type} explainer...")
    raw_shap_test = explainer.shap_values(X_test_used)

    if raw_shap_train is None or raw_shap_test is None:
        raise ValueError(
            "Explainer returned None for SHAP values. The model may be degenerate, "
            "or the explainer initialization failed silently."
        )

    # -----------------------------------------
    # 3. Handle SHAP Object Types (LightGBM/SHAP newer versions)
    # -----------------------------------------
    def _extract_array(shap_result):
        if hasattr(shap_result, "values"):
            return shap_result.values
        return shap_result

    shap_train_extracted = _extract_array(raw_shap_train)
    shap_test_extracted = _extract_array(raw_shap_test)

    # -----------------------------------------
    # 4. Handle Classification/Regression Shapes
    # -----------------------------------------
    shap_values_all_classes = None

    def _process_shape(shap_extracted, is_train=False):
        nonlocal shap_values_all_classes

        if isinstance(shap_extracted, list):
            if len(shap_extracted) > 2:
                # Multiclass (>2 classes)
                if is_train:
                    shap_values_all_classes = shap_extracted
                # Average the absolute values across all classes to get a 2D magnitude array
                return np.mean(np.abs(shap_extracted), axis=0)

            elif len(shap_extracted) == 2:
                # Binary Classification: take the positive class (index 1)
                return shap_extracted[1]

            else:
                return shap_extracted[0]

        # For Regression, or binary classification that directly returned a 2D array
        return shap_extracted

    shap_train_final = _process_shape(shap_train_extracted, is_train=True)
    shap_test_final = _process_shape(shap_test_extracted, is_train=False)

    # -----------------------------------------
    # 5. Validation
    # -----------------------------------------
    if shap_train_final.shape[1] != X_train_used.shape[1]:
        err_msg = (
            f"SHAP shape mismatch! SHAP values have {shap_train_final.shape[1]} features, "
            f"but X_train has {X_train_used.shape[1]}. The explainer is misconfigured."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    # -----------------------------------------
    # 6. Base Value Retrieval
    # -----------------------------------------
    base_value = getattr(explainer, "expected_value", None)

    # Try to unpack base value if it's wrapped in a single-element list unexpectedly
    if isinstance(base_value, (np.ndarray, list)) and len(base_value) == 1:
        base_value = base_value[0]

    return {
        "shap_values_train": shap_train_final,
        "shap_values_test": shap_test_final,
        "base_value": base_value,
        "explainer_type": explainer_type,
        "X_train_used": X_train_used,
        "X_test_used": X_test_used,
        "shap_values_all_classes": shap_values_all_classes,
        "n_train_samples_used": len(X_train_used),
        "n_test_samples_used": len(X_test_used),
    }


def compute_global_feature_importance(shap_values_train, feature_names, X_train_used):
    """
    Computes global feature importance metrics from SHAP values, generates a ranked list,
    and identifies potentially redundant feature pairs based on SHAP value correlations.
    """
    total_features = len(feature_names)

    # ---------------------------------------------------------
    # 1. Mean Absolute SHAP Importance
    # ---------------------------------------------------------
    importance = np.mean(np.abs(shap_values_train), axis=0)

    # ---------------------------------------------------------
    # 2. Build Ranked Feature Importance List
    # ---------------------------------------------------------
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Add rank (1-based)
    importance_df["rank"] = importance_df.index + 1

    # Calculate percentages safely to avoid division by zero
    total_importance = importance_df["importance"].sum()
    if total_importance > 0:
        importance_df["importance_pct"] = (importance_df["importance"] / total_importance) * 100
    else:
        importance_df["importance_pct"] = 0.0

    importance_df["cumulative_pct"] = importance_df["importance_pct"].cumsum()

    # ---------------------------------------------------------
    # 3. Top-N Features that Explain 80% of Model Behavior
    # ---------------------------------------------------------
    if total_importance > 0:
        n_features_80pct = int((importance_df["cumulative_pct"] <= 80).sum()) + 1
        n_features_80pct = min(n_features_80pct, total_features)  # cap at max features
    else:
        n_features_80pct = total_features

    logger.info(f"Top {n_features_80pct} features explain 80% of model decisions.")

    # ---------------------------------------------------------
    # 4. Feature Interaction Signal (Redundancy Check)
    # ---------------------------------------------------------
    redundant_pairs = []

    # Convert to DataFrame to easily compute correlation matrix with feature names
    shap_df = pd.DataFrame(shap_values_train, columns=feature_names)

    # Compute absolute correlation matrix
    corr_matrix = shap_df.corr().abs()

    # Extract upper triangle to avoid duplicate pairs and self-correlation
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find highly correlated pairs
    high_corr_threshold = 0.8
    for col in upper_tri.columns:
        highly_correlated = upper_tri.index[upper_tri[col] > high_corr_threshold].tolist()
        for row in highly_correlated:
            correlation_value = upper_tri.loc[row, col]
            pair = {"feature_1": row, "feature_2": col, "correlation": correlation_value}
            redundant_pairs.append(pair)
            logger.warning(
                f"Highly correlated SHAP signal detected between '{row}' and '{col}' "
                f"(r={correlation_value:.2f}). These features explain similar variance "
                f"and may be redundant."
            )

    # ---------------------------------------------------------
    # 5. Format Output
    # ---------------------------------------------------------
    feature_importance_list = importance_df.to_dict(orient="records")
    top_10_features = importance_df["feature"].head(10).tolist()

    return {
        "feature_importance": feature_importance_list,
        "top_10_features": top_10_features,
        "n_features_80pct": n_features_80pct,
        "total_features": total_features,
        "redundant_pairs": redundant_pairs
    }


def compute_local_explanation(shap_values_test, X_test_used, feature_names, base_value, row_index, y_pred):
    """
    Generates a human-readable explanation and detailed feature contributions
    for a single prediction based on local SHAP values.
    """

    # ---------------------------------------------------------
    # 1. Edge Case: Bounds Check
    # ---------------------------------------------------------
    if row_index < 0 or row_index >= len(X_test_used):
        raise IndexError(
            f"row_index {row_index} is out of bounds. "
            f"Valid range is 0 to {len(X_test_used) - 1}."
        )

    # ---------------------------------------------------------
    # 2. Extract Row Data
    # ---------------------------------------------------------
    row_shap = shap_values_test[row_index]
    row_X = X_test_used.iloc[row_index]
    predicted = float(y_pred[row_index])

    # ---------------------------------------------------------
    # 3. Build Contribution List
    # ---------------------------------------------------------
    contributions = []
    for i, feature in enumerate(feature_names):
        feat_val = row_X[feature]
        shap_val = float(row_shap[i])

        # Handle NaN feature values safely (e.g., from certain encodings/imputations)
        safe_feat_val = float(feat_val) if pd.notna(feat_val) else float('nan')

        contributions.append({
            "feature": feature,
            "feature_value": safe_feat_val,
            "shap_value": shap_val,
            "direction": "increases" if shap_val > 0 else "decreases",
            "abs_impact": abs(shap_val)
        })

    # Sort by absolute impact descending and assign ranks
    contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
    for rank, c in enumerate(contributions, 1):
        c["rank"] = rank

    # ---------------------------------------------------------
    # 4. Verification Check (SHAP sum vs Prediction)
    # ---------------------------------------------------------
    shap_sum = float(np.sum(row_shap))
    predicted_from_shap = float(base_value) + shap_sum
    discrepancy = abs(predicted - predicted_from_shap)

    # Dynamic threshold: 0.01% of prediction or minimum 1.0 (for small target scales)
    threshold = max(1.0, abs(predicted) * 0.0001)

    is_log_odds = False
    if discrepancy > threshold:
        logger.warning(
            f"SHAP values don't sum to prediction for row {row_index}. "
            f"Prediction: {predicted:.4f}, SHAP sum + base: {predicted_from_shap:.4f}. "
            f"Discrepancy: {discrepancy:.4f}"
        )
        # If there's a massive discrepancy, it typically means SHAP is in log-odds
        # (e.g., TreeExplainer on classification) while y_pred is a probability or class label.
        is_log_odds = True

    # ---------------------------------------------------------
    # 5. Human-Readable Summary
    # ---------------------------------------------------------
    top3 = contributions[:3]

    # Format the predicted value string
    summary_lines = [f"Predicted value: {predicted:.4g}. The 3 strongest drivers were:"]

    for i, c in enumerate(top3):
        arrow = "↑" if c["direction"] == "increases" else "↓"
        verb = "increased" if c["direction"] == "increases" else "decreased"

        # Add "prediction" noun only on the first item for readability
        noun_phrase = " prediction" if i == 0 else ""
        impact_str = f"{c['abs_impact']:.4g}"

        punctuation = "," if i < len(top3) - 1 else ""
        summary_lines.append(f"  '{c['feature']}' ({arrow} {verb}{noun_phrase} by {impact_str}){punctuation}")

    if is_log_odds:
        summary_lines.append(
            "\n*Note: Contributions are in log-odds units, which may not directly sum up "
            "to the final output probability/class.*"
        )

    human_summary = "\n".join(summary_lines)

    # ---------------------------------------------------------
    # 6. Return Output
    # ---------------------------------------------------------
    return {
        "row_index": row_index,
        "predicted_value": predicted,
        "base_value": float(base_value),
        "contributions": contributions,
        "top_3_drivers": top3,
        "human_summary": human_summary,
        "shap_sum_check": discrepancy
    }


def _plot_shap_summary(shap_values_train, X_train_used, feature_names, output_dir):
    """
    Generates and saves a SHAP beeswarm summary plot to visualize global feature importance
    and directionality.
    """
    # ---------------------------------------------------------
    # 1. Handle Multiclass SHAP Values
    # ---------------------------------------------------------
    if isinstance(shap_values_train, list):
        logger.info(
            "Multiclass SHAP values detected. Generating SHAP summary plot for Class 0 only. "
            "For full class coverage, consider generating separate plots per class."
        )
        shap_values_to_plot = shap_values_train[0]
    else:
        shap_values_to_plot = shap_values_train

    # ---------------------------------------------------------
    # 2. Generate and Save Plot
    # ---------------------------------------------------------
    plot_path = os.path.join(output_dir, "shap_summary_plot.png")

    try:
        # Create figure
        plt.figure(figsize=(12, 8))

        # Generate summary plot
        # show=False is MANDATORY - otherwise it blocks the execution pipeline indefinitely
        shap.summary_plot(
            shap_values_to_plot,
            X_train_used,
            feature_names=feature_names,
            show=False,
            max_display=20
        )

        plt.title("SHAP Feature Importance — Global Summary")
        plt.tight_layout()

        # Save figure
        # bbox_inches='tight' is MANDATORY to prevent feature names from being clipped
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')

    except Exception as e:
        logger.error(f"Failed to generate SHAP summary plot: {str(e)}")
        return None

    finally:
        # Always close the plot to free memory and avoid "figure already closed" backend errors
        plt.close('all')

    return plot_path


def _plot_shap_bar(global_importance: dict, output_dir: str) -> str:
    """Generates a clean horizontal bar chart of mean absolute SHAP values."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Extract top 15 features
    feat_imps = global_importance.get("feature_importance", [])
    sorted_feats = sorted(feat_imps, key=lambda x: x["importance"], reverse=True)[:15]
    features = [f["feature"] for f in sorted_feats]
    importances = [f["importance"] for f in sorted_feats]

    if not sorted_feats:
        return None

    # features = [f[0] for f in sorted_feats]
    # importances = [f[1] for f in sorted_feats]

    # Sanitize and truncate feature names for matplotlib
    clean_features = []
    for f in features:
        sanitized = f.replace("^", "²")
        if len(sanitized) > 30:
            sanitized = sanitized[:27] + "..."
        clean_features.append(sanitized)

    # Reverse so the most important is at the top of the horizontal bar chart
    clean_features.reverse()
    importances.reverse()

    plt.figure(figsize=(10, 8))
    # Color bars by importance magnitude (darker = more important)
    colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(importances)))

    bars = plt.barh(clean_features, importances, color=colors)
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output)")
    plt.title("Top Global Feature Importances")

    # Add value labels on bars
    # BUG FIX: 'padding' is not a valid matplotlib Text parameter — use a small x-offset instead
    for bar in bars:
        width = bar.get_width()
        plt.text(width * 1.01, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
                 ha='left', va='center')

    plt.tight_layout()
    path = os.path.join(plots_dir, "shap_bar_importance.png")
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()  # Invariant 1: Ensure plots don't hang

    return path


def _plot_shap_waterfall(local_explanation: dict, row_index: int, output_dir: str, suffix: str) -> str:
    """Generates a waterfall plot for one specific prediction."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    contributions = local_explanation.get("contributions", [])
    if not contributions:
        return None

    explanation = shap.Explanation(
        values=np.array([c["shap_value"] for c in contributions]),
        base_values=local_explanation["base_value"],
        data=np.array([c["feature_value"] for c in contributions]),
        feature_names=[c["feature"] for c in contributions]
    )

    plt.figure(figsize=(12, 8))
    # Invariant 1: show=False
    shap.waterfall_plot(explanation, show=False, max_display=15)

    pred_val = local_explanation.get('predicted_value', 0.0)
    plt.title(f"SHAP Waterfall — Row {row_index} | Predicted: {pred_val:.2f}")
    plt.tight_layout()

    path = os.path.join(plots_dir, f"shap_waterfall_{suffix}.png")
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()

    return path


def _plot_shap_dependence(shap_values_train, X_train_used, global_importance: dict, feature_names: list,
                          output_dir: str) -> dict:
    """Generates SHAP dependence plots for the top 3 most important features."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # BUG FIX: feature_importance is a list[dict], not a plain dict — .items() would crash
    feat_imps = global_importance.get("feature_importance", [])
    top_features = sorted(feat_imps, key=lambda x: x["importance"], reverse=True)[:3]

    paths = {}
    for i, feat_dict in enumerate(top_features):
        feature_name = feat_dict["feature"]
        if feature_name not in feature_names:
            continue

        feature_idx = feature_names.index(feature_name)

        # Sanitize filename
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_',
                           feature_name.replace(" ", "_").replace("^", "sq").replace("/", "_div_"))

        plt.figure(figsize=(10, 6))

        try:
            # Invariant 1: show=False
            shap.dependence_plot(
                feature_idx,
                shap_values_train,
                X_train_used,
                feature_names=feature_names,
                interaction_index="auto",
                show=False,
                alpha=0.5
            )
        except Exception:
            # Fallback for sparse data or when auto fails
            plt.clf()
            shap.dependence_plot(
                feature_idx,
                shap_values_train,
                X_train_used,
                feature_names=feature_names,
                interaction_index=None,
                show=False,
                alpha=0.5
            )

        plt.tight_layout()
        path = os.path.join(plots_dir, f"shap_dependence_{safe_name}.png")
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()

        paths[f"dependence_feature_{i + 1}"] = path

    return paths


def _compute_cluster_profiles(labels: np.ndarray, X_train: pd.DataFrame, feature_names: list) -> dict:
    """Computes cluster profiles as an explainability substitute for clustering tasks."""
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    unique_labels = np.unique(labels)
    valid_clusters = [lbl for lbl in unique_labels if lbl != -1]

    overall_mean = X_train.mean()
    overall_std = X_train.std().replace(0, 1e-9)  # Prevent division by zero

    profiles = []
    summaries = []
    cluster_means = {}

    for cluster_id in valid_clusters:
        mask = (labels == cluster_id)
        X_cluster = X_train[mask]
        size = int(mask.sum())

        cluster_mean = X_cluster.mean()
        cluster_means[cluster_id] = cluster_mean.values

        deviations = (cluster_mean - overall_mean) / overall_std

        defining_features = []
        for feat in feature_names:
            dev = deviations[feat]
            if abs(dev) > 1.0:
                defining_features.append({
                    "feature": feat,
                    "deviation": float(dev),
                    "direction": "above" if dev > 0 else "below"
                })

        # Sort by magnitude of deviation
        defining_features.sort(key=lambda x: abs(x["deviation"]), reverse=True)

        profiles.append({
            "cluster_id": int(cluster_id),
            "size": size,
            "pct_of_data": float(size / len(X_train)) * 100,
            "defining_features": defining_features,
            "feature_means": cluster_mean.to_dict()
        })

        # Human Summary
        if defining_features:
            top_high = [f["feature"] for f in defining_features if f["direction"] == "above"][:2]
            top_low = [f["feature"] for f in defining_features if f["direction"] == "below"][:2]
            desc = f"Cluster {cluster_id} ({size} samples, {size / len(X_train) * 100:.1f}% of data): characterized by "
            if top_high:
                desc += f"high {', '.join(top_high)}"
            if top_high and top_low:
                desc += " and "
            if top_low:
                desc += f"low {', '.join(top_low)}"
        else:
            desc = f"Cluster {cluster_id} ({size} samples): no strongly defining features."

        summaries.append(desc)

    # Compute between-cluster separation (average pairwise distance)
    separation_score = 0.0
    if len(valid_clusters) > 1:
        distances = []
        for i in range(len(valid_clusters)):
            for j in range(i + 1, len(valid_clusters)):
                dist = np.linalg.norm(cluster_means[valid_clusters[i]] - cluster_means[valid_clusters[j]])
                distances.append(dist)
        separation_score = float(np.mean(distances))

    return {
        "n_clusters": len(valid_clusters),
        "cluster_profiles": profiles,
        "separation_score": separation_score,
        "human_summaries": summaries
    }


def save_explainer_artifacts(global_importance: dict, local_explanations: list,
                             plot_paths: dict, shap_data: dict, task_type: str,
                             explainer_type: str, model_key: str, output_dir: str) -> dict:
    """Saves all explainability artifacts (SHAP values arrays, importance JSON, summary)."""

    # 1. Save SHAP Values
    shap_values_path = os.path.join(output_dir, "shap_values_test.npy")
    np.save(shap_values_path, shap_data.get("shap_values_test", np.array([])))

    file_size_mb = os.path.getsize(shap_values_path) / (1024 * 1024)
    print(f"Saved SHAP values array. Size: {file_size_mb:.2f} MB")

    # 2. Save Global Importance
    global_importance_path = os.path.join(output_dir, "global_importance.json")
    with open(global_importance_path, 'w') as f:
        json.dump(global_importance, f, indent=4)

    # 3. Save Local Explanations (handle potential bloat)
    local_explanations_path = os.path.join(output_dir, "local_explanations.json")
    local_json_str = json.dumps(local_explanations)

    if len(local_json_str.encode('utf-8')) > 2 * 1024 * 1024:
        print("local_explanations exceeds 2MB, truncating to first 100 rows.")
        local_explanations = local_explanations[:100]

    with open(local_explanations_path, 'w') as f:
        json.dump(local_explanations, f, indent=4)

    # 4. Save Explainer Summary
    # BUG FIX: feature_importance is a list[dict] not a plain dict — .items() would crash
    feat_imps = global_importance.get("feature_importance", [])
    sorted_feats = sorted(feat_imps, key=lambda x: x["importance"], reverse=True)

    summary = {
        "task_type": task_type,
        "explainer_type": explainer_type,
        "model_key": model_key,
        "n_features_total": len(feat_imps),
        # BUG FIX: correct key is "n_features_80pct", not "features_for_80_pct_importance"
        "n_features_80pct": global_importance.get("n_features_80pct", 0),
        "top_10_features": [f["feature"] for f in sorted_feats[:10]],
        # BUG FIX: correct key is "shap_base_value" not "base_value" on global_importance
        # base_value lives in shap_data, so we use the first local explanation's base_value
        "shap_base_value": shap_data.get("base_value", 0.0),
        "plots_generated": list(plot_paths.keys()),
        "explanation_timestamp": datetime.utcnow().isoformat()
    }

    summary_path = os.path.join(output_dir, "explainer_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    return {
        "summary_path": summary_path,
        "global_importance_path": global_importance_path,
        "local_explanations_path": local_explanations_path,
        "shap_values_path": shap_values_path,
        "all_plot_paths": plot_paths
    }


def run_explainer(final_best_model, X_train, X_test, y_train, y_test,
                  task_type: str, feature_names: list, evaluator_result: dict,
                  output_dir: str) -> dict:
    """
    Single entry point for model explainability. Chains all explainer steps
    and saves global/local explanations and SHAP plots to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # BRANCH 1: CLUSTERING (Skip SHAP, use profiles)
    # ---------------------------------------------------------
    if task_type == "Clustering":
        fitted_model = final_best_model.get("fitted_model", final_best_model) if isinstance(final_best_model, dict) else final_best_model
        labels = final_best_model.get("labels") if isinstance(final_best_model, dict) else None
        if labels is None:
            if hasattr(fitted_model, 'labels_'):
                labels = fitted_model.labels_
            elif hasattr(fitted_model, 'predict'):
                labels = fitted_model.predict(X_train)
            else:
                labels = np.zeros(len(X_train))

        cluster_profiles = _compute_cluster_profiles(labels, X_train, feature_names)

        # Save minimal artifacts for clustering
        summary_path = os.path.join(output_dir, "explainer_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "task_type": task_type,
                "timestamp": datetime.utcnow().isoformat(),
                "n_clusters": cluster_profiles.get("n_clusters", 0)
            }, f, indent=4)

        return {
            "cluster_profiles": cluster_profiles,
            "skipped": False,
            "artifact_paths": {"summary_path": summary_path}
        }

    # ---------------------------------------------------------
    # BRANCH 2: SUPERVISED LEARNING (Regression/Classification)
    # ---------------------------------------------------------

    # 1. Validate inputs
    is_valid = _validate_explainer_inputs(final_best_model, X_train, X_test, task_type, feature_names)
    if is_valid in (SKIP_SHAP_SENTINEL, CLUSTERING_PATH_SENTINEL):
        return {"skipped": True}

    # 2. Select and compute SHAP
    explainer, explainer_type = _select_shap_explainer(
        model=final_best_model["fitted_model"],
        model_key=final_best_model["model_key"],
        model_type=final_best_model["model_type"],
        X_train=X_train,
        task_type=task_type
    )
    shap_data = _compute_shap_values(
        explainer=explainer,
        explainer_type=explainer_type,
        X_train=X_train,
        X_test=X_test,
        task_type=task_type,
        model_key=final_best_model["model_key"]
    )

    # 3. Global Importance
    global_importance = compute_global_feature_importance(
        shap_values_train=shap_data["shap_values_train"],
        feature_names=feature_names,
        X_train_used=shap_data["X_train_used"]
    )

    # 4. Generate Global Plots
    plot_paths = {}

    # Assuming _plot_shap_summary was defined in Step 6
    plot_paths["shap_summary"] = _plot_shap_summary(
        shap_data["shap_values_train"],
        shap_data["X_train_used"],
        feature_names,
        output_dir
    )
    plot_paths["shap_bar"] = _plot_shap_bar(global_importance, output_dir)

    # 5. Local Explanations & Waterfalls (Representative Rows)
    # BUG FIX: final_best_model is a dict — .predict() lives on fitted_model inside it
    preds = final_best_model["fitted_model"].predict(X_test)

    # Flatten just in case predict() returns a 2D column vector (n_samples, 1)
    if isinstance(preds, np.ndarray) and preds.ndim > 1:
        preds = preds.flatten()

    local_explanations = []
    if len(preds) > 0:
        # Define indices to plot to get a good spread of the prediction distribution
        target_indices = {
            "row_0": 0,
            "max_pred": int(np.argmax(preds)),
            "min_pred": int(np.argmin(preds)),
            "median_pred": int(np.argsort(preds)[len(preds) // 2])
        }

        # Deduplicate indices in case dataset is tiny (e.g., max == min)
        seen_indices = set()

        for suffix, idx in target_indices.items():
            if idx < len(X_test) and idx not in seen_indices:
                seen_indices.add(idx)

                # Compute and store
                local_exp = compute_local_explanation(
                    shap_values_test=shap_data["shap_values_test"],
                    X_test_used=shap_data["X_test_used"],
                    feature_names=feature_names,
                    base_value=shap_data["base_value"],
                    row_index=idx,
                    y_pred=preds
                )
                local_explanations.append(local_exp)

                # Generate waterfall
                waterfall_path = _plot_shap_waterfall(local_exp, idx, output_dir, suffix)
                if waterfall_path:
                    plot_paths[f"waterfall_{suffix}"] = waterfall_path

    # 6. Generate Dependence Plots (Top 3 Features)
    dependence_paths = _plot_shap_dependence(
        shap_data["shap_values_train"],
        shap_data["X_train_used"],  # Always use the background sample for dependence
        global_importance,
        feature_names,
        output_dir
    )
    plot_paths.update(dependence_paths)

    # 7. Save all artifacts to disk
    artifact_paths = save_explainer_artifacts(
        global_importance=global_importance,
        local_explanations=local_explanations,
        plot_paths=plot_paths,
        shap_data=shap_data,
        task_type=task_type,
        explainer_type=explainer_type,
        model_key=evaluator_result.get("model_key", "unknown_model"),
        output_dir=output_dir
    )

    # Load back the summary so it can be returned directly in the orchestrator dict
    with open(artifact_paths["summary_path"], 'r') as f:
        explainer_summary = json.load(f)

    return {
        "global_importance": global_importance,
        "local_explanations": local_explanations,
        "shap_data": shap_data,
        "plot_paths": plot_paths,
        "artifact_paths": artifact_paths,
        "explainer_summary": explainer_summary,
        "cluster_profiles": None,
        "skipped": False
    }