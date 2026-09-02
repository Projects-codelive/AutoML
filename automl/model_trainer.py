import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import skew
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from automl.cv_utils import get_cv_strategy
from automl.logger import get_logger
from automl.model_registry import get_model_instance, get_all_model_keys

logger = get_logger("model_training")
def _validate_inputs(x_train, x_test, y_train, y_test, task_type, registry_result)->None:
    if not isinstance(x_train, pd.DataFrame) or not isinstance(x_test, pd.DataFrame):
        raise TypeError("x_train and x_test must be DataFrame and not None.")
    if x_train.shape[1] != x_test.shape[1]:
        raise TypeError(f"Feature count mismatch: X_train has {x_train.shape[1]} columns, X_test has {x_test.shape[1]}.")
    if x_train.columns.tolist() != x_test.columns.tolist():
        raise ValueError("X_train and X_test must have the exact same column names in the exact same order.")
    if task_type != "Clustering":
        if not isinstance(y_train, pd.Series) or not isinstance(y_test, pd.Series):
            raise ValueError("For supervised tasks, y_train and y_test must be pandas Series and not None.")
        if len(y_train) != len(x_train):
            raise ValueError(f"y_train length ({len(y_train)}) does not match X_train length ({len(x_train)}).")
        if len(y_test) != len(x_test):
            raise ValueError(f"y_test length ({len(y_test)}) does not match X_test length ({len(x_test)}).")
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("X_train and X_test cannot be empty.")
    valid_tasks = {"Classification", "Regression", "Clustering"}
    if task_type not in valid_tasks:
        raise ValueError(f"task_type must be one of {valid_tasks}, got '{task_type}'.")
    models_dict = registry_result.get("models", {})
    if not isinstance(models_dict, dict) or len(models_dict) == 0:
        raise ValueError("registry_result['models'] must be a non-empty dictionary.")
    nan_count = x_train.isna().sum().sum()
    if nan_count > 0:
        if nan_count > 0:
            # Find which models don't handle NaNs
            affected_models = [
                model_name for model_name, config in models_dict.items()
                if config.get("handles_nan", False) is False
            ]

            if affected_models:
                # Find exact columns with NaNs
                nan_cols = x_train.columns[x_train.isna().any()].tolist()
                logger.warning(
                    f"NaN check: Found {nan_count} NaN cells in x_train across columns: {nan_cols}. "
                    f"The following models do not support NaNs and will be affected: {affected_models}"
                )

            # 9. Check for infinite values (applied only to numeric columns to prevent TypeErrors)
        x_train_num = x_train.select_dtypes(include=[np.number])
        x_test_num = x_test.select_dtypes(include=[np.number])

        if np.isinf(x_train_num).values.any() or np.isinf(x_test_num).values.any():
            msg = "Infinite values (np.inf) detected in x_train or X_test. This will corrupt the models."
            logger.error(msg)
            raise ValueError(msg)

        # 10. Classification: y_train has at least 2 unique classes
        if task_type == "Classification":
            if y_train.nunique() < 2:
                raise ValueError("Classification task requires at least 2 unique classes in y_train.")

        # Edge case 1: Duplicate column names in x_train
        if x_train.columns.duplicated().any():
            msg = "x_train contains duplicate column names, which causes silent failures in sklearn."
            logger.error(msg)
            raise ValueError(msg)

        # Edge case 2: Regression with NaN targets
        if task_type == "Regression" and y_train.isna().any():
            msg = "y_train contains NaN values. Targets cannot contain NaNs for regression tasks."
            logger.error(msg)
            raise ValueError(msg)



def _train_single_model(model_key, model_def, X_train, X_test, y_train, y_test, task_type, cv, scoring_metric) -> dict:
    # Initialize the standardized result dictionary
    result = {
        "model_key": model_key,
        "model_name": model_def.get("name", model_key),
        "model_type": model_def.get("type", "unknown"),
        "status": "success",
        "fail_reason": None,
        "cv_scores": None,
        "cv_mean": None,
        "cv_std": None,
        "y_pred": None,
        "y_proba": None,
        "labels": None,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "n_clusters_found": None,
        "fitted_model": None,
        "training_time_s": 0.0,
        "supports_shap": model_def.get("supports_shap", False),
        "handles_nan": model_def.get("handles_nan", False),
    }

    start_time = time.time()

    try:
        # 1. Get fresh model instance
        model = get_model_instance(task_type, model_key, params=None)

        # Apply log-target transformation for positive skewed regression (e.g. price, income)
        use_log_target = False
        if task_type == "Regression" and y_train is not None:
            if (y_train > 0).all() and skew(y_train) > 0.75:
                use_log_target = True
                model = TransformedTargetRegressor(
                    regressor=model,
                    func=np.log1p,
                    inverse_func=np.expm1
                )

        # ---------------------------------------------------------
        # SUPERVISED BRANCH (Classification & Regression)
        # ---------------------------------------------------------
        if task_type in ["Classification", "Regression"]:

            # 3a. Cross-validation scoring
            try:
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring=scoring_metric, n_jobs=1
                )
                result["cv_scores"] = scores.tolist()
                result["cv_mean"] = float(np.mean(scores))
                result["cv_std"] = float(np.std(scores))
            except Exception as e:
                logger.error(f"[{model_key}] Cross-validation failed: {str(e)}")
                result["status"] = "failed"
                result["fail_reason"] = f"CV failure: {str(e)}"
                return result

            # 3b. Final fit on full X_train
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X_train, y_train)

                # Check if we caught any convergence warnings (common in Logistic/Lasso)
                for warn in w:
                    if issubclass(warn.category, ConvergenceWarning):
                        logger.warning(f"[{model_key}] Convergence warning during fit: {warn.message}")

            result["fitted_model"] = model

            # 3c. Predictions on X_test
            pred_start = time.time()
            y_pred = model.predict(X_test)
            pred_end = time.time()

            # Edge case: log prediction time for lazy learners like KNN
            if result["model_type"].lower() == "neighbor":
                logger.info(
                    f"[{model_key}] KNN prediction on {len(X_test)} samples took {pred_end - pred_start:.2f} seconds.")

            # Edge case: Check for garbage outputs in Regression
            if task_type == "Regression":
                if pd.isna(y_pred).any() or np.isinf(y_pred).any():
                    raise ValueError("Model produced NaN or Infinite predictions.")

            # Edge case: Check for degenerate outputs in Classification
            if task_type == "Classification":
                if len(np.unique(y_pred)) == 1:
                    logger.warning(
                        f"[{model_key}] Degenerate model: Predicted only a single class for all samples in test set.")

            result["y_pred"] = y_pred

            # Probability predictions for Classification
            if task_type == "Classification":
                if model_def.get("supports_predict_proba", False):
                    try:
                        result["y_proba"] = model.predict_proba(X_test)
                    except Exception as e:
                        logger.warning(
                            f"[{model_key}] predict_proba failed despite 'supports_predict_proba=True': {str(e)}")
                else:
                    logger.info(f"[{model_key}] does not support predict_proba. AUC/LogLoss cannot be computed.")

        # ---------------------------------------------------------
        # CLUSTERING BRANCH
        # ---------------------------------------------------------
        elif task_type == "Clustering":

            # 4a. Fit and Predict
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)

                if model_def.get("supports_predict", False):
                    model.fit(X_train)
                    labels = model.predict(X_train)
                else:
                    labels = model.fit_predict(X_train)

                for warn in w:
                    if issubclass(warn.category, ConvergenceWarning):
                        logger.warning(f"[{model_key}] Convergence warning during clustering: {warn.message}")

            result["fitted_model"] = model
            result["labels"] = labels

            # 4b. Filter noise for metrics
            n_clusters_found = len(set(labels) - {-1})
            result["n_clusters_found"] = n_clusters_found

            is_noise_model = model_def.get("returns_noise_label", False)
            if is_noise_model:
                valid_mask = labels != -1
                if not valid_mask.any():
                    logger.warning(f"[{model_key}] DBSCAN classified all points as noise (-1).")
            else:
                valid_mask = np.ones(len(labels), dtype=bool)

            valid_labels = labels[valid_mask]

            # Handle pandas vs numpy masking
            if isinstance(X_train, pd.DataFrame):
                valid_X = X_train.iloc[valid_mask]
            else:
                valid_X = X_train[valid_mask]

            # 4c. Calculate Silhouette
            if n_clusters_found < 2:
                result["silhouette"] = -1.0
            else:
                try:
                    result["silhouette"] = silhouette_score(valid_X, valid_labels)
                except Exception as e:
                    logger.warning(f"[{model_key}] Silhouette score calculation failed: {str(e)}")
                    result["silhouette"] = -1.0

            # 4d. Calculate Davies-Bouldin and Calinski-Harabasz (No noise filtering required)
            # Both metrics require >= 2 clusters to avoid division by zero / ValueError
            unique_total = len(set(labels))
            if unique_total >= 2:
                try:
                    result["davies_bouldin"] = davies_bouldin_score(X_train, labels)
                    result["calinski_harabasz"] = calinski_harabasz_score(X_train, labels)
                except Exception as e:
                    logger.warning(f"[{model_key}] DB/CH scoring failed: {str(e)}")

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    except MemoryError:
        error_msg = f"MemoryError during training. Dataset shape: X_train={X_train.shape}."
        logger.error(f"[{model_key}] {error_msg}")
        result["status"] = "failed"
        result["fail_reason"] = error_msg

    except Exception as e:
        error_msg = f"Unhandled exception during training/prediction: {str(e)}"
        logger.error(f"[{model_key}] {error_msg}")
        result["status"] = "failed"
        result["fail_reason"] = error_msg

    finally:
        # 5. Record wall-clock end time
        result["training_time_s"] = time.time() - start_time

    # 6. Return standard schema
    return result


def run_parallel_training(X_train, X_test, y_train, y_test, task_type, registry_result, n_jobs=-1):
    models = registry_result["models"]
    scoring = registry_result["scoring_metric"]
    scoring_str = scoring[0]
    model_key = get_all_model_keys(task_type)
    cv = get_cv_strategy(task_type, y_train)
    if task_type == "Clustering":
        n_rows = len(X_train)
        filtered_keys = []
        for key in model_key:
            max_rows = models[key].get("max_rows_recommended", float("inf"))
            if n_rows > max_rows:
                logger.warning(
                    f"Skipping '{key}': dataset has {n_rows} rows but model "
                    f"recommends max {max_rows}. Would be too slow/memory-intensive."
                )
            else:
                filtered_keys.append(key)
        model_key = filtered_keys
    if not model_key:
        raise RuntimeError("No models available to train after applying row limits.")
    logger.info(f"Starting parallel training | models: {model_key} | n_jobs: {n_jobs}")
    try:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_train_single_model)(
                model_key=key,
                model_def=models[key],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                task_type=task_type,
                cv=cv,
                scoring_metric=scoring_str,
            )
            for key in model_key
        )
    except Exception as parallel_err:
        logger.error(
            f"joblib Parallel execution crashed: {str(parallel_err)}. "
            "Falling back to sequential execution loop."
        )
        results = []
        for key in model_key:
            res = _train_single_model(
                model_key=key,
                model_def=models[key],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                task_type=task_type,
                cv=cv,
                scoring_metric=scoring_str,
            )
            results.append(res)
    successful_models = sum(1 for res in results if res.get("status") == "success")
    total_models = len(results)
    failed_models = total_models - successful_models
    if failed_models > 0:
        logger.info(f"{successful_models}/{total_models} models trained ({failed_models} failed).")
    else:
        logger.info(f"{successful_models}/{total_models} models trained successfully.")
        # Critical failure: entire pipeline fails if no models succeeded
    if successful_models == 0:
        logger.critical("All models failed during training. Pipeline cannot continue.")
        raise RuntimeError("0 models trained successfully. Check error logs for failure reasons.")
    return results


def build_leaderboard(results: list[dict], task_type: str) -> list[dict]:
    # 1. Filter to successful models only
    successful_models = [res for res in results if res.get("status") == "success"]
    failed_models = [res for res in results if res.get("status") != "success"]

    if failed_models:
        logger.info(f"Excluded {len(failed_models)} failed models from the leaderboard.")

    if not successful_models:
        logger.warning("No successful models to build a leaderboard with.")
        return []

    # 2. Sorting Logic
    if task_type in ["Classification", "Regression"]:
        # Primary sort: cv_mean (descending -> higher is better)
        # Secondary sort: cv_std (ascending -> lower is better)
        # We achieve this by sorting ascending on a tuple of (-cv_mean, cv_std)
        successful_models.sort(
            key=lambda x: (
                -x.get("cv_mean", float('-inf')),
                x.get("cv_std", float('inf'))
            )
        )

        # Add actual RMSE for regression to make the leaderboard human-readable
        if task_type == "Regression":
            for model in successful_models:
                cv_mean = model.get("cv_mean")
                if cv_mean is not None:
                    model["cv_rmse"] = -cv_mean
    elif task_type == "Clustering":
        def cluster_sort_key(model_result):
            sil = model_result.get("silhouette")
            ch = model_result.get("calinski_harabasz")
            sil = -1.0 if sil is None else float(sil)
            ch = 0.0 if ch is None else float(ch)

            # Special constraint: if silhouette == -1.0, push to bottom
            is_degenerate = 1 if sil == -1.0 else 0
            return (is_degenerate, -sil, -ch)
        successful_models.sort(key=cluster_sort_key)
    else:
        raise ValueError(f"Unknown task_type for leaderboard: {task_type}")
    # 3. Add Rank (1-indexed)
    for index, model in enumerate(successful_models):
        model["rank"] = index + 1
    return successful_models


def save_trained_models(results: list[dict], task_type: str, output_dir: str) -> dict:
    # 1. Setup Directories
    base_dir = Path(output_dir)
    models_dir = base_dir / "models"
    preds_dir = base_dir / "predictions"
    clust_dir = base_dir / "clustering"

    # Always create models directory
    models_dir.mkdir(parents=True, exist_ok=True)

    if task_type in ["Classification", "Regression"]:
        preds_dir.mkdir(parents=True, exist_ok=True)
    elif task_type == "Clustering":
        clust_dir.mkdir(parents=True, exist_ok=True)

    # Initialize return dictionary
    output = {
        "saved_models": {},
        "saved_predictions": {},
        "save_failures": []
    }

    # 2. Iterate and Save
    for res in results:
        # Skip failed models or those missing a fitted_model object
        if res.get("status") != "success" or res.get("fitted_model") is None:
            continue

        model_key = res["model_key"]

        try:
            # --- Save the Model ---
            model_path = models_dir / f"{model_key}.joblib"
            joblib.dump(res["fitted_model"], model_path, compress=3)

            # Log file size in MB
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"[{model_key}] Saved model to disk: {size_mb:.2f} MB")

            output["saved_models"][model_key] = str(model_path)

            # --- Save Artifacts based on Task Type ---
            if task_type in ["Classification", "Regression"]:
                # y_pred
                if res.get("y_pred") is not None:
                    pred_path = preds_dir / f"{model_key}_y_pred.npy"
                    np.save(pred_path, res["y_pred"])
                    output["saved_predictions"][model_key] = str(pred_path)

                # y_proba (Classification only)
                if res.get("y_proba") is not None:
                    proba_path = preds_dir / f"{model_key}_y_proba.npy"
                    np.save(proba_path, res["y_proba"])
            elif task_type == "Clustering":
                # Labels
                if res.get("labels") is not None:
                    labels_path = clust_dir / f"{model_key}_labels.npy"
                    np.save(labels_path, res["labels"])
                    # Map labels under saved_predictions for standardized tracking
                    output["saved_predictions"][model_key] = str(labels_path)
                # Metrics JSON
                metrics_path = clust_dir / f"{model_key}_metrics.json"
                metrics_data = {
                    "silhouette": res.get("silhouette"),
                    "davies_bouldin": res.get("davies_bouldin"),
                    "calinski_harabasz": res.get("calinski_harabasz"),
                    "n_clusters_found": res.get("n_clusters_found")
                }
                with open(metrics_path, "w") as f:
                    json.dump(metrics_data, f, indent=4)
        except Exception as e:
            logger.error(f"[{model_key}] Failed to save model or artifacts: {str(e)}")
            output["save_failures"].append(model_key)
    return output


def _make_json_serializable(obj):
    """Recursively converts numpy data types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [_make_json_serializable(v) for v in obj]
    return obj


def save_trainer_summary(leaderboard: list[dict], failed_models: list[dict], task_type: str, save_paths: dict,
                         output_dir: str) -> dict:
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Leaderboard entries with model paths
    # We strip out heavy non-summary objects like fitted_model or predictions if they are still there
    clean_leaderboard = []
    saved_models_dict = save_paths.get("saved_models", {})

    for entry in leaderboard:
        clean_entry = {
            "rank": entry.get("rank"),
            "model_key": entry.get("model_key"),
            "model_name": entry.get("model_name"),
            "model_type": entry.get("model_type"),
            "training_time_s": entry.get("training_time_s"),
            "supports_shap": entry.get("supports_shap"),
            "model_path": saved_models_dict.get(entry.get("model_key"))
        }

        # Add task-specific metrics
        if task_type in ["Classification", "Regression"]:
            clean_entry["cv_mean"] = entry.get("cv_mean")
            clean_entry["cv_std"] = entry.get("cv_std")
            if task_type == "Regression" and "cv_rmse" in entry:
                clean_entry["cv_rmse"] = entry.get("cv_rmse")
        elif task_type == "Clustering":
            clean_entry["silhouette"] = entry.get("silhouette")
            clean_entry["calinski_harabasz"] = entry.get("calinski_harabasz")
            clean_entry["davies_bouldin"] = entry.get("davies_bouldin")
            clean_entry["n_clusters_found"] = entry.get("n_clusters_found")

        clean_leaderboard.append(clean_entry)

    # 2. Identify Best and Worst Models
    best_model = None
    worst_model = None

    if clean_leaderboard:
        best = clean_leaderboard[0]
        worst = clean_leaderboard[-1]

        best_model = {
            "model_key": best.get("model_key"),
            "model_name": best.get("model_name"),
            "model_path": best.get("model_path")
        }
        worst_model = {
            "model_key": worst.get("model_key")
        }

        # Assign primary metric representation
        if task_type == "Regression":
            best_model["cv_rmse"] = best.get("cv_rmse")
            worst_model["cv_rmse"] = worst.get("cv_rmse")
            primary_metric_name = "neg_root_mean_squared_error"
        elif task_type == "Classification":
            best_model["cv_mean"] = best.get("cv_mean")
            worst_model["cv_mean"] = worst.get("cv_mean")
            primary_metric_name = "f1_weighted"  # Or fallback to standard setup string
        else:
            best_model["silhouette"] = best.get("silhouette")
            worst_model["silhouette"] = worst.get("silhouette")
            primary_metric_name = "silhouette"
    else:
        primary_metric_name = "unknown"

    # 3. Format failed models briefly
    clean_failed = [
        {
            "model_key": f.get("model_key"),
            "fail_reason": f.get("fail_reason")
        }
        for f in failed_models
    ]

    # 4. Construct Final Summary Dictionary
    summary = {
        "task_type": task_type,
        "training_timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_models_attempted": len(leaderboard) + len(failed_models),
        "total_models_succeeded": len(leaderboard),
        "total_models_failed": len(failed_models),
        "cv_folds": 5 if task_type != "Clustering" else None,
        "cv_strategy": "KFold/StratifiedKFold" if task_type != "Clustering" else None,
        "primary_metric": primary_metric_name,
        "leaderboard": clean_leaderboard,
        "failed_models": clean_failed,
        "best_model": best_model,
        "worst_model": worst_model
    }
    # 5. Sanitize types for JSON
    summary_serializable = _make_json_serializable(summary)
    # 6. Write to disk
    summary_path = base_dir / "trainer_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(summary_serializable, f, indent=4)
        logger.info(f"Trainer summary saved successfully to {summary_path}")
    except Exception as e:
        logger.error(f"Failed to write trainer_summary.json: {str(e)}")
        # Do not raise, we still want to return the summary dict to the pipeline
    return summary_serializable


def get_best_model_info(leaderboard: list[dict]) -> dict:
    # 1. Check for empty leaderboard
    if not leaderboard:
        raise ValueError("The leaderboard is empty. No successful models are available to retrieve.")
    best_model = leaderboard[0]
    # 2. Sanity check: Ensure the top model didn't fail
    # We use .get("status", "success") so it safely defaults to "success"
    # if the status key was already cleaned up during Step 7
    if best_model.get("status", "success") != "success":
        model_key = best_model.get("model_key", "unknown")
        raise ValueError(f"The top-ranked model '{model_key}' is marked as failed. The leaderboard is corrupted.")
    # 3. Return the rank-1 model's full dictionary
    return best_model


def load_trained_model(model_key: str, task_type: str, output_dir: str) -> object:
    # 1. Construct path
    model_path = Path(output_dir) / "models" / f"{model_key}.joblib"

    # 2. Check existence
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found for '{model_key}' at '{model_path}'. "
            f"Has trainer.py been run for this task?"
        )
    # 3. Log file size for traceability
    try:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"[{model_key}] Loading model from disk ({size_mb:.2f} MB)...")
    except OSError as e:
        logger.warning(f"[{model_key}] Could not read file size: {e}")
    # 4. Load the model with corruption handling
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        error_msg = (
            f"Failed to load the model file for '{model_key}' at '{model_path}'. "
            f"The file may be corrupted from an interrupted training run. "
            f"Please delete the file and retrain. Original error: {str(e)}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def run_trainer(X_train, X_test, y_train, y_test, task_type: str, registry_result: dict, output_dir: str,
                n_jobs: int = -1) -> dict:
    logger.info("=" * 60)
    logger.info("MODEL TRAINER STARTED")
    logger.info("=" * 60)

    # 1. Validate Inputs
    _validate_inputs(X_train, X_test, y_train, y_test, task_type, registry_result)
    logger.info(f"STEP 1 — Input validation passed | X_train: {X_train.shape} | task: {task_type}")

    # 2. Train Models in Parallel
    results = run_parallel_training(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task_type=task_type,
        registry_result=registry_result,
        n_jobs=n_jobs
    )

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    logger.info(f"STEP 2 — Training complete | {len(successful)}/{len(results)} models succeeded")

    # 3. Post-Processing & Persistence
    leaderboard = build_leaderboard(successful, task_type)
    save_paths = save_trained_models(results, task_type, output_dir)
    summary = save_trainer_summary(leaderboard, failed, task_type, save_paths, output_dir)
    best = get_best_model_info(leaderboard)

    # Extract the primary metric for logging (handles Classification, Regression, and Clustering)
    metric_val = best.get("cv_rmse") or best.get("cv_mean") or best.get("silhouette")
    logger.info(f"STEP 3 — Best model: {best['model_name']} | rank-1 metric: {metric_val}")

    logger.info("=" * 60)
    logger.info("MODEL TRAINER COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

    # 4. Return Final Orchestration Dictionary
    return {
        "leaderboard": leaderboard,
        "best_model": best,
        "failed_models": failed,
        "save_paths": save_paths,
        "trainer_summary": summary,
        "all_results": results
    }