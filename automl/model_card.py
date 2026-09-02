import os
import json
import logging
import platform
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn

# Optional imports for software version tracking
try:
    import xgboost
    HAS_XGBOOST = True
except (ImportError, AttributeError):
    xgboost = None
    HAS_XGBOOST = False

try:
    import lightgbm
    HAS_LIGHTGBM = True
except (ImportError, AttributeError):
    lightgbm = None
    HAS_LIGHTGBM = False

try:
    import optuna
    HAS_OPTUNA = True
except (ImportError, AttributeError):
    optuna = None
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except (ImportError, AttributeError):
    shap = None
    HAS_SHAP = False

from automl.logger import get_logger

logger = get_logger("Model_Card")


def _make_json_serializable(obj):
    """
    Recursively converts numpy and pandas objects, custom types, and estimators
    into pure Python JSON-serializable primitives (int, float, bool, str, list, dict, None).
    Strips raw fitted estimators and functions.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.intc, np.intp, np.int8, np.int16, np.int32,
                          np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return _make_json_serializable(obj.tolist())
    elif isinstance(obj, (pd.Series, pd.Index)):
        return _make_json_serializable(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return _make_json_serializable(obj.to_dict(orient="records"))
    elif isinstance(obj, Path):
        return str(obj)
    elif pd.isna(obj) if 'pd' in globals() else obj != obj:
        return None
    elif hasattr(obj, "fit") or hasattr(obj, "predict") or callable(obj):
        # Strip fitted estimator objects or functions
        return str(type(obj).__name__)
    else:
        return obj


def _validate_model_card_inputs(
    eda_summary: dict = None,
    preprocessor_summary: dict = None,
    feature_engineering_summary: dict = None,
    trainer_summary: dict = None,
    hpo_summary: dict = None,
    evaluator_result: dict = None,
    explainer_result: dict = None,
    final_best_model: dict = None,
    task_type: str = "Regression"
) -> dict:
    """
    Step 1 — Validates all pipeline input summaries and computes completeness scores.
    Never raises exceptions, ensuring downstream assembly always proceeds gracefully.
    """
    inputs_to_check = {
        "eda_summary": {
            "data": eda_summary,
            "expected_keys": ["task_type", "target_col", "n_rows", "column_types"]
        },
        "preprocessor_summary": {
            "data": preprocessor_summary,
            "expected_keys": ["encoding", "scaling", "imputation"]
        },
        "feature_engineering": {
            "data": feature_engineering_summary,
            "expected_keys": ["final_feature_list", "final_feature_count"]
        },
        "trainer_summary": {
            "data": trainer_summary,
            "expected_keys": ["leaderboard", "best_model"]
        },
        "hpo_summary": {
            "data": hpo_summary,
            "expected_keys": ["leaderboard", "best_model", "n_trials_per_model"]
        },
        "evaluator_result": {
            "data": evaluator_result,
            "expected_keys": ["metrics", "plot_paths"]
        },
        "explainer_result": {
            "data": explainer_result,
            "expected_keys": ["global_importance", "plot_paths"]
        },
        "final_best_model": {
            "data": final_best_model,
            "expected_keys": ["model_key", "model_name", "fitted_model"]
        }
    }

    report = {}
    missing_sections = []
    total_score = 0.0
    active_sections = 0

    for section_name, info in inputs_to_check.items():
        data = info["data"]
        expected_keys = info["expected_keys"]

        if section_name == "feature_engineering" and (data is None or (isinstance(data, dict) and len(data) == 0)):
            report[section_name] = "skipped / not applicable"
            continue

        active_sections += 1
        if data is None:
            status = "missing"
            missing_sections.append(section_name)
        elif not isinstance(data, dict) or len(data) == 0:
            status = "empty"
            missing_sections.append(section_name)
        else:
            present_keys = [k for k in expected_keys if k in data and data[k] is not None]
            if len(present_keys) == len(expected_keys):
                status = "complete"
                total_score += 1.0
            elif len(present_keys) > 0:
                status = "partial"
                total_score += 0.5
            else:
                status = "empty"
                missing_sections.append(section_name)

        report[section_name] = status

    overall_completeness_pct = round((total_score / max(1, active_sections)) * 100.0, 1)
    report["overall_completeness_pct"] = overall_completeness_pct
    report["missing_sections"] = missing_sections

    logger.info("=" * 60)
    logger.info("Model Card Completeness:")
    for section_name in inputs_to_check.keys():
        st = report.get(section_name, "missing").upper()
        logger.info(f"  {section_name:<25}: {st}")
    logger.info(f"  Overall Completeness      : {overall_completeness_pct}%")
    logger.info("=" * 60)

    return report


def _build_dataset_section(
    eda_summary: dict = None,
    preprocessor_summary: dict = None,
    feature_engineering_summary: dict = None,
    task_type: str = "Regression",
    target_col: str = None,
    dataset_path: str = "dataset.csv"
) -> dict:
    """
    Step 2 — Builds the comprehensive dataset description section.
    """
    eda = eda_summary or {}
    prep = preprocessor_summary or {}
    fe = feature_engineering_summary or {}

    dataset_str = str(dataset_path) if dataset_path else "dataset.csv"
    dataset_name = Path(dataset_str).name

    n_rows_orig = int(eda.get("n_rows", 0)) if eda.get("n_rows") is not None else 0
    n_cols_orig = int(eda.get("n_columns", 0)) if eda.get("n_columns") is not None else 0

    n_rows_train = int(round(n_rows_orig * 0.8)) if n_rows_orig > 0 else 0
    n_rows_test = n_rows_orig - n_rows_train

    # Column Types Breakdown
    col_types = eda.get("column_types", {})
    if not isinstance(col_types, dict):
        col_types = {}

    column_types_dict = {
        "numerical_continuous": list(col_types.get("numerical_continuous", []) or []),
        "numerical_discrete": list(col_types.get("numerical_discrete", []) or []),
        "categorical": list(col_types.get("categorical", []) or []),
        "boolean": list(col_types.get("boolean", []) or []),
        "datetime": list(col_types.get("datetime", []) or [])
    }

    # Target Analysis
    target_analysis_data = eda.get("target_analysis", {})
    if not isinstance(target_analysis_data, dict):
        target_analysis_data = {}

    target_analysis_dict = {
        "mean": float(target_analysis_data.get("mean")) if target_analysis_data.get("mean") is not None else None,
        "median": float(target_analysis_data.get("median")) if target_analysis_data.get("median") is not None else None,
        "std": float(target_analysis_data.get("std")) if target_analysis_data.get("std") is not None else None,
        "skewness": float(target_analysis_data.get("skewness")) if target_analysis_data.get("skewness") is not None else None,
        "is_skewed": bool(target_analysis_data.get("is_skewed", False))
    }

    # Missing Data
    missing_report = eda.get("missing_report", []) or []
    columns_with_missing = []
    max_missing_pct = 0.0

    if isinstance(missing_report, list):
        for item in missing_report:
            if isinstance(item, dict):
                col = item.get("Column") or item.get("column")
                pct = item.get("Missing_Percentage") or item.get("missing_percentage") or item.get("pct", 0.0)
                if col:
                    columns_with_missing.append(str(col))
                try:
                    if float(pct) > max_missing_pct:
                        max_missing_pct = float(pct)
                except (ValueError, TypeError):
                    pass
    elif isinstance(missing_report, dict):
        columns_with_missing = list(missing_report.keys())

    imputation_strategy = prep.get("imputation", {})
    if not isinstance(imputation_strategy, dict):
        imputation_strategy = {}

    missing_data_dict = {
        "columns_with_missing": columns_with_missing,
        "max_missing_pct": round(float(max_missing_pct), 2),
        "imputation_strategy": imputation_strategy
    }

    # Preprocessing Applied
    encoding_applied = prep.get("encoding", {}) if isinstance(prep.get("encoding"), dict) else {}
    scaling_applied = prep.get("scaling", {}) if isinstance(prep.get("scaling"), dict) else {}
    outlier_capping = prep.get("outliers", prep.get("outlier_bounds", {}))
    if not isinstance(outlier_capping, dict):
        outlier_capping = {}

    preprocessing_applied_dict = {
        "encoding": encoding_applied,
        "scaling": scaling_applied,
        "outlier_capping": outlier_capping
    }

    # Feature Engineering / Preprocessing Feature Count
    final_feature_list = list(fe.get("final_feature_list", []) or prep.get("feature_names", []) or [])
    final_feature_count = int(fe.get("final_feature_count", len(final_feature_list)))
    if final_feature_count == 0:
        final_feature_count = int(prep.get("final_feature_count", prep.get("n_features", n_cols_orig)))
    orig_feature_count = int(fe.get("original_feature_count", n_cols_orig))

    ai_derived = fe.get("ai_derived_cols", fe.get("ai_derived_features", [])) or []
    poly_feats = fe.get("poly_features", fe.get("polynomial_features", [])) or []
    features_created = list(set(list(ai_derived) + list(poly_feats)))

    var_dropped = fe.get("variance_dropped", []) or []
    model_dropped = fe.get("model_based_dropped", []) or []
    features_dropped = list(set(list(var_dropped) + list(model_dropped)))

    feature_engineering_dict = {
        "original_feature_count": orig_feature_count,
        "final_feature_count": final_feature_count,
        "features_created": features_created,
        "features_dropped": features_dropped,
        "final_feature_list": final_feature_list
    }

    return {
        "dataset_path": dataset_str,
        "dataset_name": dataset_name,
        "task_type": task_type,
        "target_column": target_col,
        "n_rows_original": n_rows_orig,
        "n_columns_original": n_cols_orig,
        "n_rows_train": n_rows_train,
        "n_rows_test": n_rows_test,
        "column_types": column_types_dict,
        "target_analysis": target_analysis_dict,
        "missing_data": missing_data_dict,
        "preprocessing_applied": preprocessing_applied_dict,
        "feature_engineering": feature_engineering_dict
    }


def _build_model_section(
    final_best_model: dict = None,
    trainer_summary: dict = None,
    hpo_summary: dict = None,
    task_type: str = "Regression"
) -> dict:
    """
    Step 3 — Builds the model description section including selection process,
    hyperparameters, training details, and leaderboards. Strips fitted objects.
    """
    best = final_best_model or {}
    trainer = trainer_summary or {}
    hpo = hpo_summary or {}

    model_key = str(best.get("model_key", "unknown"))
    model_name = str(best.get("model_name", "Unknown Model"))
    model_type = str(best.get("model_type", "ensemble"))
    model_source = str(best.get("source", "baseline")).lower()

    # Selection process
    trainer_lb = trainer.get("leaderboard", []) or []
    hpo_lb = hpo.get("leaderboard", []) or []

    n_models_trained = int(trainer.get("total_models_attempted", len(trainer_lb)))
    n_models_hpo_tuned = int(hpo.get("n_models_tuned", len(hpo_lb)))

    task_lower = task_type.lower()
    if task_lower == "regression":
        selection_criterion = "lowest RMSE (cross-validated)"
    elif task_lower == "classification":
        selection_criterion = "highest Weighted F1 / Accuracy (cross-validated)"
    elif task_lower == "clustering":
        selection_criterion = "highest Silhouette Score"
    else:
        selection_criterion = "primary evaluation metric"

    # Baseline & HPO ranks
    baseline_rank = None
    training_time_s = 0.0
    for idx, entry in enumerate(trainer_lb):
        if isinstance(entry, dict) and entry.get("model_key") == model_key:
            baseline_rank = entry.get("rank", idx + 1)
            training_time_s = float(entry.get("training_time_s", entry.get("train_time", 0.0)) or 0.0)
            break

    hpo_rank = None
    final_refit_time_s = 0.0
    hpo_improved = best.get("hpo_improved")
    for idx, entry in enumerate(hpo_lb):
        if isinstance(entry, dict) and entry.get("model_key") == model_key:
            hpo_rank = entry.get("rank", idx + 1)
            final_refit_time_s = float(entry.get("refit_time_s", entry.get("tuning_time_seconds", 0.0)) or 0.0)
            if hpo_improved is None:
                hpo_improved = entry.get("hpo_improved")
            break

    # Hyperparameters
    raw_params = best.get("best_params") or best.get("params") or {}
    if not raw_params and model_source == "baseline":
        final_params = {"defaults": "sklearn default parameters"}
    else:
        final_params = _make_json_serializable(raw_params) if isinstance(raw_params, dict) else {}

    hpo_n_trials = int(hpo.get("n_trials_per_model", best.get("n_trials", 0)) or 0)
    hpo_duration_s = float(best.get("tuning_time_seconds", hpo.get("total_tuning_time_seconds", 0.0)) or 0.0)

    # Clean leaderboards (stripping fitted models)
    clean_baseline_lb = []
    for item in trainer_lb:
        if isinstance(item, dict):
            c = dict(item)
            c.pop("fitted_model", None)
            c.pop("model_obj", None)
            clean_baseline_lb.append(_make_json_serializable(c))

    clean_hpo_lb = []
    for item in hpo_lb:
        if isinstance(item, dict):
            c = dict(item)
            c.pop("fitted_model", None)
            c.pop("model_obj", None)
            clean_hpo_lb.append(_make_json_serializable(c))

    cv_strategy = "StratifiedKFold(n_splits=5, shuffle=True)" if task_type == "Classification" else "KFold(n_splits=5, shuffle=True)"

    return {
        "model_key": model_key,
        "model_name": model_name,
        "model_type": model_type,
        "model_source": model_source,
        "selection_process": {
            "n_models_trained": n_models_trained,
            "n_models_hpo_tuned": n_models_hpo_tuned,
            "selection_criterion": selection_criterion,
            "baseline_rank": baseline_rank,
            "hpo_rank": hpo_rank,
            "hpo_improved": hpo_improved
        },
        "hyperparameters": {
            "final_params": final_params,
            "hpo_n_trials": hpo_n_trials,
            "hpo_duration_s": round(hpo_duration_s, 2),
            "hpo_optimizer": "Optuna TPE Sampler"
        },
        "training_details": {
            "cv_folds": 5,
            "cv_strategy": cv_strategy,
            "training_time_s": round(training_time_s, 3),
            "final_refit_time_s": round(final_refit_time_s, 3)
        },
        "baseline_leaderboard": clean_baseline_lb,
        "hpo_leaderboard": clean_hpo_lb
    }


def _build_performance_section(
    evaluator_result: dict = None,
    task_type: str = "Regression",
    target_col: str = None
) -> dict:
    """
    Step 4 — Builds the complete model performance section for Regression,
    Classification, or Clustering tasks.
    """
    ev = evaluator_result or {}
    metrics = ev.get("metrics", {}) or {}
    plot_paths = ev.get("plot_paths", {}) or {}
    pred_intervals = ev.get("prediction_intervals", {}) or {}

    task = task_type.capitalize() if task_type else "Regression"

    if task == "Regression":
        rmse = float(metrics.get("rmse", 0.0) or 0.0)
        mae = float(metrics.get("mae", 0.0) or 0.0)
        r2 = float(metrics.get("r2", 0.0) or 0.0)
        adj_r2 = float(metrics.get("adjusted_r2", metrics.get("adj_r2", r2)) or 0.0)
        mape = float(metrics.get("mape", 0.0) or 0.0)

        # Interpretations
        if r2 > 0.9:
            r2_interp = "Excellent fit — model explains >90% of variance."
        elif r2 > 0.7:
            r2_interp = "Good fit — model explains >70% of variance."
        elif r2 > 0.5:
            r2_interp = "Moderate fit — explains >50% of variance."
        else:
            r2_interp = "Poor fit — explains less than 50% of variance. Consider more features."

        target_name = target_col or "target"
        overall_interp = (
            f"The model explains {r2 * 100:.1f}% of variance in {target_name}. "
            f"Predictions have an average error of {rmse:,.2f} (RMSE) and {mae:,.2f} (MAE)."
        )

        # Residuals
        residual_stats = metrics.get("residual_analysis", {}) or {}
        res_mean = float(residual_stats.get("mean", 0.0) or 0.0)
        res_std = float(residual_stats.get("std", rmse) or 0.0)
        res_skew = float(residual_stats.get("skew", 0.0) or 0.0)

        if abs(res_mean) < 1.0:
            res_interp = "Near-zero mean indicates no systematic bias in predictions."
        else:
            res_interp = f"Mean residual of {res_mean:.2f} suggests slight systematic bias."

        # Overfitting Assessment
        overfitting_gap = float(metrics.get("overfitting_gap", 0.0) or 0.0)
        if overfitting_gap > 0.15:
            of_verdict = "Significant overfitting"
        elif overfitting_gap > 0.10:
            of_verdict = "Possible overfitting"
        else:
            of_verdict = "No overfitting detected"

        return {
            "primary_metrics": {
                "rmse": {"value": round(rmse, 4), "unit": target_name, "interpretation": f"Root Mean Squared Error in {target_name} units"},
                "mae": {"value": round(mae, 4), "unit": target_name, "interpretation": f"Mean Absolute Error in {target_name} units"},
                "r2": {"value": round(r2, 4), "unit": None, "interpretation": r2_interp},
                "adj_r2": {"value": round(adj_r2, 4), "unit": None, "interpretation": f"Adjusted R² accounting for {metrics.get('n_features', 'all')} features"},
                "mape": {"value": round(mape, 2), "unit": "%", "interpretation": f"Mean Absolute Percentage Error is {mape:.1f}%"}
            },
            "interpretation": overall_interp,
            "residual_analysis": {
                "mean": round(res_mean, 4),
                "std": round(res_std, 4),
                "skew": round(res_skew, 4),
                "interpretation": res_interp
            },
            "overfitting_assessment": {
                "gap": round(overfitting_gap, 4),
                "verdict": of_verdict,
                "threshold_used": 0.10
            },
            "prediction_intervals": _make_json_serializable(pred_intervals),
            "evaluation_warnings": metrics.get("warnings", []) or []
        }

    elif task == "Classification":
        acc = float(metrics.get("accuracy", 0.0) or 0.0)
        f1_w = float(metrics.get("f1_weighted", metrics.get("f1_score", 0.0)) or 0.0)
        auc_roc = float(metrics.get("auc_roc")) if metrics.get("auc_roc") is not None else None
        mcc = float(metrics.get("mcc", 0.0) or 0.0)
        precision = float(metrics.get("precision_weighted", metrics.get("precision", 0.0)) or 0.0)
        recall = float(metrics.get("recall_weighted", metrics.get("recall", 0.0)) or 0.0)

        conf_mat = metrics.get("confusion_matrix", [])
        per_class = metrics.get("classification_report", {})
        classes = metrics.get("classes", []) or []

        overall_interp = (
            f"The model achieves an overall accuracy of {acc * 100:.1f}% and a weighted F1-score of {f1_w:.3f}. "
            f"Matthews Correlation Coefficient (MCC) is {mcc:.3f}."
        )

        return {
            "primary_metrics": {
                "accuracy": {"value": round(acc, 4), "interpretation": f"Correctly classified {acc * 100:.1f}% of test instances."},
                "f1_weighted": {"value": round(f1_w, 4), "interpretation": f"Weighted harmonic mean of precision and recall is {f1_w:.3f}."},
                "auc_roc": {"value": round(auc_roc, 4) if auc_roc is not None else None, "interpretation": f"Area Under ROC Curve is {auc_roc:.3f}" if auc_roc is not None else "N/A"},
                "mcc": {"value": round(mcc, 4), "interpretation": f"Matthews Correlation Coefficient is {mcc:.3f}."},
                "precision": {"value": round(precision, 4)},
                "recall": {"value": round(recall, 4)}
            },
            "confusion_matrix": _make_json_serializable(conf_mat),
            "per_class_report": _make_json_serializable(per_class),
            "n_classes": len(classes) if classes else len(per_class),
            "class_labels": [str(c) for c in classes],
            "interpretation": overall_interp,
            "evaluation_warnings": metrics.get("warnings", []) or []
        }

    else:  # Clustering
        silhouette = float(metrics.get("silhouette_score", metrics.get("silhouette", 0.0)) or 0.0)
        db_score = float(metrics.get("davies_bouldin_score", metrics.get("davies_bouldin", 0.0)) or 0.0)
        ch_score = float(metrics.get("calinski_harabasz_score", metrics.get("calinski_harabasz", 0.0)) or 0.0)

        n_clusters = int(metrics.get("n_clusters", 0))
        noise_pts = int(metrics.get("noise_points", 0))
        cluster_sizes = metrics.get("cluster_sizes", {}) or {}

        if silhouette > 0.5:
            sil_interp = "Strong, well-separated cluster structure."
        elif silhouette > 0.25:
            sil_interp = "Moderate cluster separation."
        else:
            sil_interp = "Weak or overlapping clusters."

        overall_interp = (
            f"Clustering partitioned data into {n_clusters} clusters with a Silhouette score of {silhouette:.3f}. "
            f"Davies-Bouldin index: {db_score:.3f}."
        )

        return {
            "primary_metrics": {
                "silhouette": {"value": round(silhouette, 4), "range": "[-1, 1]", "interpretation": sil_interp},
                "davies_bouldin": {"value": round(db_score, 4), "interpretation": "Lower values indicate better separation"},
                "calinski_harabasz": {"value": round(ch_score, 2), "interpretation": "Higher values indicate denser, more separated clusters"}
            },
            "n_clusters": n_clusters,
            "noise_points": noise_pts,
            "cluster_sizes": _make_json_serializable(cluster_sizes),
            "interpretation": overall_interp,
            "evaluation_warnings": metrics.get("warnings", []) or []
        }


def _build_fairness_section(evaluator_result: dict = None, task_type: str = "Regression") -> dict:
    """
    Step 5 — Builds the fairness and subgroup slice disparity section.
    """
    ev = evaluator_result or {}
    fairness = ev.get("fairness", {}) or {}

    if not fairness or fairness.get("skipped", False):
        return {
            "fairness_assessed": False,
            "has_disparity": False,
            "overall_verdict": "Fairness analysis skipped — high cardinality or not applicable",
            "slices_analyzed": 0,
            "slices_flagged": 0,
            "disparity_details": [],
            "disparity_warnings": fairness.get("warnings", ["Slice analysis skipped or not requested."]),
            "recommendation": "Fairness analysis was skipped. If evaluating in high-stakes domains, run subgroup slice evaluations."
        }

    has_disparity = bool(fairness.get("has_disparity", False))
    overall_verdict = (
        "Disparities detected — review recommended" if has_disparity else "No significant disparities detected"
    )
    slices_analyzed = int(fairness.get("slices_analyzed", fairness.get("total_slices", 0)))
    slices_flagged = int(fairness.get("slices_flagged", 0))

    disparity_details = []
    raw_details = fairness.get("disparity_details", fairness.get("slices_details", [])) or []

    for item in raw_details:
        if isinstance(item, dict):
            diff_pct = float(item.get("difference_pct", 0.0) or 0.0)
            if diff_pct < 10.0:
                concern = "low"
            elif diff_pct < 25.0:
                concern = "medium"
            else:
                concern = "high"

            disparity_details.append({
                "feature": str(item.get("feature", "unknown")),
                "group_value": str(item.get("group_value", item.get("group", "unknown"))),
                "group_size": int(item.get("group_size", item.get("size", 0))),
                "metric_value": round(float(item.get("metric_value", 0.0)), 4),
                "overall_metric": round(float(item.get("overall_metric", 0.0)), 4),
                "difference": round(float(item.get("difference", 0.0)), 4),
                "difference_pct": round(diff_pct, 2),
                "concern_level": concern
            })

    if not has_disparity:
        recommendation = "Model performs consistently across all tested groups."
    else:
        recommendation = (
            "Consider collecting more data for underperforming groups or applying group-aware training techniques."
        )

    return {
        "fairness_assessed": True,
        "has_disparity": has_disparity,
        "overall_verdict": overall_verdict,
        "slices_analyzed": slices_analyzed,
        "slices_flagged": slices_flagged,
        "disparity_details": disparity_details,
        "disparity_warnings": fairness.get("warnings", []) or [],
        "recommendation": recommendation
    }


def _build_explainability_section(explainer_result: dict = None, task_type: str = "Regression") -> dict:
    """
    Step 6 — Builds the explainability section with global feature rankings, SHAP metadata,
    and automatic narrative key findings.
    """
    exp = explainer_result or {}

    if not exp or exp.get("skipped", False):
        return {
            "explainability_available": False,
            "explainer_type": "none",
            "global_feature_importance": [],
            "key_findings": ["SHAP explanation skipped — model does not support this explainer type or SHAP was not installed."],
            "n_features_80pct": 0,
            "shap_base_value": None,
            "plot_paths": {
                "shap_summary": None,
                "shap_bar": None,
                "shap_waterfall": None,
                "shap_dependence": None
            },
            "cluster_profiles": exp.get("cluster_profiles") if isinstance(exp, dict) else None
        }

    explainer_type = str(exp.get("explainer_type", "tree"))
    raw_importance = exp.get("global_importance", {}) or {}
    feature_list = raw_importance.get("feature_importance", []) or []

    # Format top 10 features
    global_feature_importance = []
    for idx, f in enumerate(feature_list[:10]):
        if isinstance(f, dict):
            global_feature_importance.append({
                "rank": idx + 1,
                "feature": str(f.get("feature", f"feature_{idx}")),
                "importance_pct": round(float(f.get("importance_pct", 0.0)), 2),
                "cumulative_pct": round(float(f.get("cumulative_pct", 0.0)), 2)
            })

    n_features_80pct = int(raw_importance.get("n_features_80pct", len(global_feature_importance)))
    shap_data = exp.get("shap_data", {}) or {}
    shap_base_val = shap_data.get("base_value")
    if shap_base_val is not None:
        try:
            shap_base_val = round(float(shap_base_val), 4)
        except (ValueError, TypeError):
            shap_base_val = None

    # Key Findings Generation
    findings = []
    if global_feature_importance:
        top_f = global_feature_importance[0]
        findings.append(
            f"'{top_f['feature']}' is the strongest predictor, accounting for {top_f['importance_pct']:.1f}% of model decisions."
        )
        findings.append(
            f"Top {n_features_80pct} features explain 80% of all model decisions."
        )
        if len(feature_list) >= 3:
            bottom_3 = [f.get("feature") for f in feature_list[-3:] if isinstance(f, dict)]
            findings.append(
                f"Lowest impact features: {bottom_3}. Consider removing in future iterations to improve inference latency."
            )
    else:
        findings.append("Global feature importance details are summarized in the explainer artifacts.")

    # Plot paths
    plots = exp.get("plot_paths", {}) or {}
    plot_paths = {
        "shap_summary": plots.get("shap_summary"),
        "shap_bar": plots.get("shap_bar"),
        "shap_waterfall": plots.get("waterfall_row_0", plots.get("shap_waterfall")),
        "shap_dependence": plots.get("dependence_0", plots.get("shap_dependence"))
    }

    return {
        "explainability_available": True,
        "explainer_type": explainer_type,
        "global_feature_importance": global_feature_importance,
        "key_findings": findings,
        "n_features_80pct": n_features_80pct,
        "shap_base_value": shap_base_val,
        "plot_paths": plot_paths,
        "cluster_profiles": _make_json_serializable(exp.get("cluster_profiles"))
    }


def _build_limitations_section(
    evaluator_result: dict = None,
    eda_summary: dict = None,
    task_type: str = "Regression",
    final_best_model: dict = None,
    target_col: str = None
) -> dict:
    """
    Step 7 — Generates the limitations, risks, and failure modes section.
    """
    ev = evaluator_result or {}
    eda = eda_summary or {}
    limitations = []

    # 1. Data coverage limitation
    n_rows = int(eda.get("n_rows", 0)) if eda.get("n_rows") is not None else 0
    if n_rows < 1000 and n_rows > 0:
        limitations.append({
            "type": "data_size",
            "severity": "high",
            "description": f"Model trained on only {n_rows} samples. Predictions may be unreliable for rare feature combinations not well represented in training data."
        })
    elif n_rows < 5000 and n_rows > 0:
        limitations.append({
            "type": "data_size",
            "severity": "medium",
            "description": f"Training set of {n_rows} samples is moderate. Generalization performance may improve significantly with larger training datasets."
        })

    # 2. Skewed target limitation (Regression)
    if task_type == "Regression":
        target_analysis = eda.get("target_analysis", {})
        skewness = target_analysis.get("skewness") if isinstance(target_analysis, dict) else None
        if skewness is not None and abs(float(skewness)) > 2.0:
            target_name = target_col or "target"
            limitations.append({
                "type": "target_distribution",
                "severity": "medium",
                "description": f"Target '{target_name}' is highly skewed (skewness={float(skewness):.2f}). The model may underperform on extreme tail values."
            })

    # 3. Overfitting warning
    gap = 0.0
    if isinstance(ev.get("metrics"), dict):
        gap = float(ev["metrics"].get("overfitting_gap", 0.0) or 0.0)

    if gap > 0.15:
        limitations.append({
            "type": "overfitting",
            "severity": "high",
            "description": f"Overfitting detected (train-test gap: {gap:.3f}). Model may not generalize robustly to unseen distribution shifts."
        })
    elif gap > 0.10:
        limitations.append({
            "type": "overfitting",
            "severity": "medium",
            "description": f"Mild generalization gap observed ({gap:.3f}). Regularization should be monitored."
        })

    # 4. Missing data at inference
    missing = eda.get("missing_report", []) or []
    if missing:
        if isinstance(missing, list):
            cols = [m.get("Column", m.get("column", "col")) for m in missing if isinstance(m, dict)]
        else:
            cols = list(missing.keys())
        limitations.append({
            "type": "missing_data_handling",
            "severity": "low",
            "description": f"Columns {cols} contained missing values during training. At inference time, automated imputation will be applied, introducing slight uncertainty."
        })

    # 5. Fairness disparity
    fairness = ev.get("fairness", {})
    if isinstance(fairness, dict) and fairness.get("has_disparity"):
        limitations.append({
            "type": "fairness",
            "severity": "medium",
            "description": "Performance disparities detected across subgroup slices. Review fairness audit before deploying in high-stakes domains."
        })

    # 6. Temporal limitation
    limitations.append({
        "type": "temporal",
        "severity": "medium",
        "description": "Model was trained on historical data and assumes stationary distribution. Re-training is recommended periodically to combat covariate drift."
    })

    # 7. Out-of-distribution
    limitations.append({
        "type": "out_of_distribution",
        "severity": "medium",
        "description": "Predictions for feature values outside the observed training bounds represent extrapolations and should be treated with lower confidence."
    })

    # 8. Clustering specific
    if task_type == "Clustering":
        limitations.append({
            "type": "cluster_instability",
            "severity": "low",
            "description": "Cluster assignments may shift with new data points. Clusters are not guaranteed to be strictly invariant across different random initializations."
        })

    # Tally severities
    n_high = sum(1 for l in limitations if l["severity"] == "high")
    n_med = sum(1 for l in limitations if l["severity"] == "medium")
    n_low = sum(1 for l in limitations if l["severity"] == "low")

    if n_high > 0:
        overall_risk = "high"
        deployment_recommendation = "Not recommended for high-stakes deployment without improvements"
    elif n_med > 0:
        overall_risk = "medium"
        deployment_recommendation = "Requires review before deployment"
    else:
        overall_risk = "low"
        deployment_recommendation = "Ready for production with monitoring"

    return {
        "limitations": limitations,
        "n_high_severity": n_high,
        "n_medium_severity": n_med,
        "n_low_severity": n_low,
        "overall_risk_level": overall_risk,
        "deployment_recommendation": deployment_recommendation
    }


def _build_recommendations_section(
    evaluator_result: dict = None,
    explainer_result: dict = None,
    eda_summary: dict = None,
    trainer_summary: dict = None,
    final_best_model: dict = None,
    task_type: str = "Regression"
) -> dict:
    """
    Step 8 — Generates prioritized and actionable model improvement recommendations.
    """
    ev = evaluator_result or {}
    exp = explainer_result or {}
    eda = eda_summary or {}
    trainer = trainer_summary or {}
    best = final_best_model or {}

    recommendations = []

    # 1. Feature recommendations from SHAP
    raw_importance = exp.get("global_importance", {}) if isinstance(exp, dict) else {}
    feature_list = raw_importance.get("feature_importance", []) or []
    if len(feature_list) > 6:
        bottom_features = [f.get("feature") for f in feature_list[-3:] if isinstance(f, dict)]
        recommendations.append({
            "category": "feature_engineering",
            "priority": "low",
            "action": f"Consider removing low-importance features: {bottom_features}. Simplifying the feature set will reduce inference latency and noise."
        })

    # 2. Data recommendations
    n_rows = int(eda.get("n_rows", 0)) if eda.get("n_rows") is not None else 0
    if n_rows < 5000 and n_rows > 0:
        recommendations.append({
            "category": "data_collection",
            "priority": "high",
            "action": f"Collect more training data. Current dataset has {n_rows} rows; tree-based models typically exhibit significant gains with >10,000 samples."
        })

    # 3. HPO recommendations
    trainer_lb = trainer.get("leaderboard", []) or []
    hpo_worsened = [e for e in trainer_lb if isinstance(e, dict) and e.get("hpo_improved") is False]
    if hpo_worsened:
        recommendations.append({
            "category": "hyperparameter_optimization",
            "priority": "medium",
            "action": "HPO worsened some models. Consider increasing n_trials to ≥100 and refining search spaces for learning rate and estimators."
        })

    # 4. Alternative models to try
    model_key = str(best.get("model_key", "")).lower()
    if model_key in ("ridge", "lasso", "elastic_net", "linear_regression", "logistic_regression"):
        recommendations.append({
            "category": "model_selection",
            "priority": "high",
            "action": "Linear model won the leaderboard. Consider engineering higher-order polynomial interactions or exploring CatBoost."
        })

    # 5. Monitoring recommendations (always present)
    recommendations.append({
        "category": "production_monitoring",
        "priority": "high",
        "action": "Implement data drift monitoring on input features and configure automated alerts if prediction residuals exceed baseline thresholds by >20%."
    })

    # 6. Retraining schedule (always present)
    recommendations.append({
        "category": "retraining",
        "priority": "medium",
        "action": "Schedule periodic monthly retraining or triggered retraining upon detecting covariate shift. Version-control all artifacts."
    })

    n_high = sum(1 for r in recommendations if r["priority"] == "high")
    n_med = sum(1 for r in recommendations if r["priority"] == "medium")
    n_low = sum(1 for r in recommendations if r["priority"] == "low")

    top_3_actions = [r["action"] for r in recommendations[:3]]

    return {
        "recommendations": recommendations,
        "n_high_priority": n_high,
        "n_medium_priority": n_med,
        "n_low_priority": n_low,
        "top_3_actions": top_3_actions
    }


def _build_pipeline_provenance_section(
    eda_summary: dict = None,
    preprocessor_summary: dict = None,
    feature_engineering_summary: dict = None,
    trainer_summary: dict = None,
    hpo_summary: dict = None,
    final_best_model: dict = None,
    save_paths: dict = None
) -> dict:
    """
    Step 9 — Documents the complete lineage and software reproducibility specifications.
    """
    eda = eda_summary or {}
    prep = preprocessor_summary or {}
    fe = feature_engineering_summary or {}
    trainer = trainer_summary or {}
    hpo = hpo_summary or {}
    best = final_best_model or {}

    # Stage 1: EDA
    eda_keys = [
        f"Task type detected: {eda.get('task_type', 'Unknown')}",
        f"Total original rows: {eda.get('n_rows', 'Unknown')}",
        f"Columns: {eda.get('n_columns', 'Unknown')}"
    ]

    # Stage 2: Preprocessor
    prep_enc = list(prep.get("encoding", {}).keys()) if isinstance(prep.get("encoding"), dict) else []
    prep_scale = list(prep.get("scaling", {}).keys()) if isinstance(prep.get("scaling"), dict) else []
    prep_keys = [
        f"Encoding applied to: {prep_enc[:3]}",
        f"Scaling applied to: {prep_scale[:3]}",
        f"Imputation strategy: {list(prep.get('imputation', {}).keys())[:3]}"
    ]

    # Stage 3: Feature Engineering
    fe_orig = fe.get("original_feature_count", "N")
    fe_final = fe.get("final_feature_count", len(fe.get("final_feature_list", [])))
    fe_keys = [
        f"Started with {fe_orig} features",
        f"Final selected: {fe_final} features after selection funnel"
    ]

    # Stage 4: Model Trainer
    tr_lb = trainer.get("leaderboard", []) or []
    best_tr = trainer.get("best_model", {}) or {}
    tr_keys = [
        f"{len(tr_lb)} models evaluated across cross-validation folds",
        f"Top baseline model: {best_tr.get('model_name', 'Unknown')}"
    ]

    # Stage 5: HPO
    hpo_lb = hpo.get("leaderboard", []) or []
    hpo_keys = [
        f"{hpo.get('n_models_tuned', len(hpo_lb))} models tuned via Optuna TPE",
        f"Trials per model: {hpo.get('n_trials_per_model', 50)}",
        f"Total tuning duration: {hpo.get('total_tuning_time_seconds', 0.0):.1f}s"
    ]

    pipeline_stages = [
        {
            "stage": "EDA",
            "artifact": "artifacts/eda/eda_summary.json",
            "key_outputs": eda_keys
        },
        {
            "stage": "Preprocessing",
            "artifact": "artifacts/preprocessor/preprocessor_summary.json",
            "key_outputs": prep_keys
        },
        {
            "stage": "Feature Engineering",
            "artifact": "artifacts/feature_engineering/feature_engineering_summary.json",
            "key_outputs": fe_keys
        },
        {
            "stage": "Model Training",
            "artifact": "artifacts/model_trainer/trainer_summary.json",
            "key_outputs": tr_keys
        },
        {
            "stage": "HPO",
            "artifact": "artifacts/hpo/hpo_summary.json",
            "key_outputs": hpo_keys
        }
    ]

    # Software versions
    software_versions = {
        "python": platform.python_version(),
        "sklearn": getattr(sklearn, "__version__", "unknown"),
        "xgboost": getattr(xgboost, "__version__", "not installed") if HAS_XGBOOST else "not installed",
        "lightgbm": getattr(lightgbm, "__version__", "not installed") if HAS_LIGHTGBM else "not installed",
        "optuna": getattr(optuna, "__version__", "not installed") if HAS_OPTUNA else "not installed",
        "shap": getattr(shap, "__version__", "not installed") if HAS_SHAP else "not installed",
        "pandas": getattr(pd, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown")
    }

    model_file_path = str(best.get("model_path", "artifacts/hpo/models/final_best_model.joblib"))

    return {
        "pipeline_stages": pipeline_stages,
        "software_versions": software_versions,
        "reproducibility": {
            "random_seed": 42,
            "cv_folds": 5,
            "train_test_split": "80/20",
            "shuffle": True
        },
        "artifact_directory": "artifacts/",
        "model_file": model_file_path
    }


def assemble_model_card(
    dataset_section: dict,
    model_section: dict,
    performance_section: dict,
    fairness_section: dict,
    explainability_section: dict,
    limitations_section: dict,
    recommendations_section: dict,
    provenance_section: dict,
    task_type: str = "Regression",
    completeness_report: dict = None,
    target_col: str = None
) -> dict:
    """
    Step 10 — Assembles all section components into the unified root Model Card dictionary.
    """
    model_name = model_section.get("model_name", "Best Model")
    target_name = target_col or dataset_section.get("target_column") or "target"
    task = task_type.capitalize() if task_type else "Regression"

    # Executive Summary One-Liner
    if task == "Regression":
        primary = performance_section.get("primary_metrics", {})
        rmse_val = primary.get("rmse", {}).get("value", 0.0)
        r2_val = primary.get("r2", {}).get("value", 0.0)
        n_test = dataset_section.get("n_rows_test", 0)
        one_line = (
            f"{model_name} achieves an RMSE of {rmse_val:,.2f} with R²={r2_val:.3f} "
            f"on {target_name} across a held-out test set of {n_test} samples."
        )
    elif task == "Classification":
        primary = performance_section.get("primary_metrics", {})
        acc_val = primary.get("accuracy", {}).get("value", 0.0)
        f1_val = primary.get("f1_weighted", {}).get("value", 0.0)
        n_classes = performance_section.get("n_classes", 2)
        n_test = dataset_section.get("n_rows_test", 0)
        one_line = (
            f"{model_name} achieves {acc_val * 100:.1f}% accuracy and weighted F1={f1_val:.3f} "
            f"across {n_classes} classes on {n_test} test samples."
        )
    else:  # Clustering
        primary = performance_section.get("primary_metrics", {})
        sil_val = primary.get("silhouette", {}).get("value", 0.0)
        n_clusters = performance_section.get("n_clusters", 0)
        one_line = (
            f"{model_name} identified {n_clusters} clusters with a silhouette score of {sil_val:.3f}."
        )

    verdict = limitations_section.get("deployment_recommendation", "Review before deployment")
    top_3_actions = recommendations_section.get("top_3_actions", [])

    model_card = {
        "model_card_version": "1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task_type": task,
        "completeness": completeness_report or {},
        "executive_summary": {
            "model_name": model_name,
            "one_line": one_line,
            "verdict": verdict,
            "top_3_actions": top_3_actions
        },
        "dataset": dataset_section,
        "model": model_section,
        "performance": performance_section,
        "fairness": fairness_section,
        "explainability": explainability_section,
        "limitations": limitations_section,
        "recommendations": recommendations_section,
        "provenance": provenance_section
    }

    return model_card


def save_model_card(model_card: dict, output_dir: str = "artifacts/model_card") -> dict:
    """
    Step 11 — Serializes and saves the model card in both JSON and Markdown formats.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_path = out_path / "model_card.json"
    md_path = out_path / "model_card.md"

    # 1. Save JSON
    clean_card = _make_json_serializable(model_card)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_card, f, indent=4)

    json_size_kb = round(os.path.getsize(json_path) / 1024.0, 2)
    if json_size_kb > 10240.0:  # > 10MB
        logger.warning(f"model_card.json exceeds 10MB ({json_size_kb:.1f} KB). Check for raw data arrays.")

    # 2. Save Markdown
    md_content = _generate_model_card_markdown(clean_card)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    md_size_kb = round(os.path.getsize(md_path) / 1024.0, 2)

    logger.info(f"Model Card JSON saved to: {json_path.resolve()} ({json_size_kb} KB)")
    logger.info(f"Model Card Markdown saved to: {md_path.resolve()} ({md_size_kb} KB)")

    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "json_size_kb": json_size_kb,
        "md_size_kb": md_size_kb
    }


def _generate_model_card_markdown(card: dict) -> str:
    """
    Renders the Model Card dictionary into clean, publication-ready GitHub Flavored Markdown.
    """
    exec_summary = card.get("executive_summary", {})
    dataset = card.get("dataset", {})
    model = card.get("model", {})
    perf = card.get("performance", {})
    fair = card.get("fairness", {})
    exp = card.get("explainability", {})
    lims = card.get("limitations", {})
    recs = card.get("recommendations", {})
    prov = card.get("provenance", {})

    lines = []
    lines.append(f"# Model Card — {exec_summary.get('model_name', 'AutoML Model')}")
    lines.append(f"**Generated:** {card.get('generated_at', '')}  ")
    lines.append(f"**Task:** {card.get('task_type', '')} | **Target:** {dataset.get('target_column', 'N/A')}\n")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append(f"{exec_summary.get('one_line', 'No summary available.')}\n")
    lines.append(f"**Deployment Verdict:** `{exec_summary.get('verdict', 'N/A')}`\n")

    # Dataset
    lines.append("## Dataset")
    lines.append(f"- **Original Shape:** {dataset.get('n_rows_original', 0):,} rows × {dataset.get('n_columns_original', 0)} columns")
    lines.append(f"- **Train / Test Split:** {dataset.get('n_rows_train', 0):,} train | {dataset.get('n_rows_test', 0):,} test")
    fe_data = dataset.get("feature_engineering", {})
    lines.append(f"- **Final Engineered Features:** {fe_data.get('final_feature_count', 0)}")
    missing_cols = dataset.get("missing_data", {}).get("columns_with_missing", [])
    if missing_cols:
        lines.append(f"- **Imputed Columns:** {', '.join(missing_cols[:5])}")
    lines.append("")

    # Model
    lines.append("## Model Architecture & Selection")
    lines.append(f"- **Algorithm:** {model.get('model_name', 'Unknown')} ({model.get('model_type', 'N/A')})")
    lines.append(f"- **Selection Source:** {model.get('model_source', '').upper()}")
    hparams = model.get("hyperparameters", {}).get("final_params", {})
    lines.append(f"- **Hyperparameters:** `{json.dumps(hparams)}`\n")

    # Performance
    lines.append("## Performance Evaluation")
    lines.append("| Metric | Value | Interpretation |")
    lines.append("| :--- | :--- | :--- |")
    for metric_name, details in perf.get("primary_metrics", {}).items():
        if isinstance(details, dict):
            val = details.get("value")
            interp = details.get("interpretation", "")
            lines.append(f"| **{metric_name.upper()}** | {val} | {interp} |")
    lines.append(f"\n*{perf.get('interpretation', '')}*\n")

    # Explainability
    lines.append("## Explainability & Key Drivers")
    top_features = exp.get("global_feature_importance", [])
    if top_features:
        for f in top_features[:5]:
            lines.append(f"1. **{f.get('feature')}** — {f.get('importance_pct', 0.0)}% impact (cumulative: {f.get('cumulative_pct', 0.0)}%)")
    for finding in exp.get("key_findings", []):
        lines.append(f"- {finding}")
    lines.append("")

    # Fairness
    lines.append("## Fairness & Subgroup Performance")
    lines.append(f"- **Assessment Status:** {fair.get('overall_verdict', 'N/A')}")
    lines.append(f"- **Recommendation:** {fair.get('recommendation', 'N/A')}\n")

    # Limitations
    lines.append("## Limitations & Known Risks")
    for lim in lims.get("limitations", []):
        sev_icon = "⚠️" if lim.get("severity") in ("high", "medium") else "ℹ️"
        lines.append(f"- {sev_icon} **[{lim.get('severity', '').upper()}]** {lim.get('description', '')}")
    lines.append("")

    # Recommendations
    lines.append("## Actionable Recommendations")
    for rec in recs.get("recommendations", []):
        lines.append(f"- **[{rec.get('priority', '').upper()}]** {rec.get('action', '')}")
    lines.append("")

    # Provenance
    lines.append("## Lineage & Provenance")
    repro = prov.get("reproducibility", {})
    lines.append(f"Reproducible with random seed `{repro.get('random_seed', 42)}`, CV folds `{repro.get('cv_folds', 5)}`.")
    lines.append(f"- **Model Artifact:** `{prov.get('model_file', 'N/A')}`")
    lines.append(f"- **Artifacts Lineage Directory:** `{prov.get('artifact_directory', 'artifacts/')}`\n")

    return "\n".join(lines)


def run_model_card(
    eda_summary: dict = None,
    preprocessor_summary: dict = None,
    feature_engineering_summary: dict = None,
    trainer_summary: dict = None,
    hpo_summary: dict = None,
    evaluator_result: dict = None,
    explainer_result: dict = None,
    final_best_model: dict = None,
    task_type: str = "Regression",
    target_col: str = None,
    dataset_path: str = "dataset.csv",
    output_dir: str = "artifacts/model_card"
) -> dict:
    """
    Single Entry Point — Called by pipeline.py.
    Chains all Model Card assembly and serialization steps.
    """
    logger.info("=" * 60)
    logger.info(f"MODEL CARD GENERATION STARTED | task: {task_type}")
    logger.info("=" * 60)

    # Step 1: Validate Inputs & Completeness Report
    completeness_report = _validate_model_card_inputs(
        eda_summary=eda_summary,
        preprocessor_summary=preprocessor_summary,
        feature_engineering_summary=feature_engineering_summary,
        trainer_summary=trainer_summary,
        hpo_summary=hpo_summary,
        evaluator_result=evaluator_result,
        explainer_result=explainer_result,
        final_best_model=final_best_model,
        task_type=task_type
    )

    # Step 2: Dataset Section
    dataset_section = _build_dataset_section(
        eda_summary=eda_summary,
        preprocessor_summary=preprocessor_summary,
        feature_engineering_summary=feature_engineering_summary,
        task_type=task_type,
        target_col=target_col,
        dataset_path=dataset_path
    )

    # Step 3: Model Section
    model_section = _build_model_section(
        final_best_model=final_best_model,
        trainer_summary=trainer_summary,
        hpo_summary=hpo_summary,
        task_type=task_type
    )

    # Step 4: Performance Section
    performance_section = _build_performance_section(
        evaluator_result=evaluator_result,
        task_type=task_type,
        target_col=target_col
    )

    # Step 5: Fairness Section
    fairness_section = _build_fairness_section(
        evaluator_result=evaluator_result,
        task_type=task_type
    )

    # Step 6: Explainability Section
    explainability_section = _build_explainability_section(
        explainer_result=explainer_result,
        task_type=task_type
    )

    # Step 7: Limitations Section
    limitations_section = _build_limitations_section(
        evaluator_result=evaluator_result,
        eda_summary=eda_summary,
        task_type=task_type,
        final_best_model=final_best_model,
        target_col=target_col
    )

    # Step 8: Recommendations Section
    recommendations_section = _build_recommendations_section(
        evaluator_result=evaluator_result,
        explainer_result=explainer_result,
        eda_summary=eda_summary,
        trainer_summary=trainer_summary,
        final_best_model=final_best_model,
        task_type=task_type
    )

    # Step 9: Lineage & Provenance Section
    provenance_section = _build_pipeline_provenance_section(
        eda_summary=eda_summary,
        preprocessor_summary=preprocessor_summary,
        feature_engineering_summary=feature_engineering_summary,
        trainer_summary=trainer_summary,
        hpo_summary=hpo_summary,
        final_best_model=final_best_model
    )

    # Step 10: Assemble Full Model Card
    assembled_card = assemble_model_card(
        dataset_section=dataset_section,
        model_section=model_section,
        performance_section=performance_section,
        fairness_section=fairness_section,
        explainability_section=explainability_section,
        limitations_section=limitations_section,
        recommendations_section=recommendations_section,
        provenance_section=provenance_section,
        task_type=task_type,
        completeness_report=completeness_report,
        target_col=target_col
    )

    # Step 11: Save Artifacts (JSON & Markdown)
    save_paths = save_model_card(assembled_card, output_dir=output_dir)

    logger.info("=" * 60)
    logger.info("MODEL CARD GENERATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

    return {
        "model_card": assembled_card,
        "save_paths": save_paths,
        "completeness_report": completeness_report
    }
