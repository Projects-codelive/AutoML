import json
import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.stats import skew
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score

from automl.cv_utils import get_cv_strategy
from automl.logger import get_logger
from automl.model_registry import get_search_space, get_model_instance, get_scoring_metric

logger = get_logger("Hyperparameter Tuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — _validate_hpo_inputs
# ─────────────────────────────────────────────────────────────────────────────
def _validate_hpo_inputs(
    trainer_result, X_train, X_test, y_train, y_test,
    task_type, n_trials, top_n_models, timeout_per_model, registry_result
) -> None:
    """
    Validates all HPO inputs before any Optuna study is created.
    registry_result is passed as a dedicated parameter — it is NOT inside trainer_result.
    Raises on hard failures. Logs warnings on soft issues.
    """
    # 1. Leaderboard must be non-empty
    leader = trainer_result.get("leaderboard", [])
    if not leader:
        raise RuntimeError(
            "Cannot run HPO: trainer leaderboard is empty. Check trainer logs."
        )

    # 2. DataFrames
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_train and X_test must be pandas DataFrames and not None.")
    if X_train.shape[1] != X_test.shape[1]:
        raise TypeError(
            f"Feature count mismatch: X_train has {X_train.shape[1]} columns, "
            f"X_test has {X_test.shape[1]}."
        )
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise ValueError(
            "X_train and X_test must have the exact same column names in the exact same order."
        )

    # 3. n_trials
    if not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer >= 1.")
    if n_trials < 5:
        logger.warning(
            f"n_trials={n_trials} is fewer than 5. "
            "This gives unreliable HPO results. Consider increasing n_trials."
        )

    # 4. top_n_models — clamp if necessary (clamping happens in _select_models_for_hpo)
    if not isinstance(top_n_models, int) or top_n_models < 1:
        raise ValueError("top_n_models must be a positive integer >= 1.")
    leaderboard_size = len(leader)
    if top_n_models > leaderboard_size:
        logger.warning(
            f"top_n_models ({top_n_models}) exceeds leaderboard size ({leaderboard_size}). "
            f"Will be clamped to {leaderboard_size} during model selection."
        )

    # 5. timeout_per_model
    if not isinstance(timeout_per_model, int) or timeout_per_model <= 0:
        raise ValueError("timeout_per_model must be a positive integer.")
    if timeout_per_model < 30:
        logger.warning(
            "timeout_per_model < 30s may cause all trials to be pruned before completing."
        )

    # 6. Supervised task targets
    if task_type in ["Classification", "Regression"]:
        if not isinstance(y_train, pd.Series) or not isinstance(y_test, pd.Series):
            raise TypeError(
                "For supervised tasks, y_train and y_test must be pandas Series."
            )
        if len(y_train) != len(X_train):
            raise ValueError(
                f"Length mismatch: y_train ({len(y_train)}) != X_train ({len(X_train)})."
            )
        if len(y_test) != len(X_test):
            raise ValueError(
                f"Length mismatch: y_test ({len(y_test)}) != X_test ({len(X_test)})."
            )

    # 7. Search spaces — use registry_result parameter directly
    search_spaces = registry_result.get("search_spaces", {})
    if not search_spaces:
        raise RuntimeError(
            "Cannot run HPO: registry_result['search_spaces'] is empty. "
            "HPO has no search space to explore."
        )

    logger.info("HPO input validation passed.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — _select_models_for_hpo
# ─────────────────────────────────────────────────────────────────────────────
def _select_models_for_hpo(leaderboard: list[dict], top_n_models: int, task_type: str) -> list[dict]:
    """
    Selects which models from the leaderboard get HPO.
    Returns list[dict] — the candidate model dicts, NOT just their keys.
    Empty list means HPO is skipped entirely (caller handles gracefully).
    """
    # Take top-N — Python slicing never raises on out-of-bounds
    top_models = leaderboard[:top_n_models]

    hpo_candidates = []
    for model in top_models:
        model_key = model["model_key"]

        # Skip models with no search space
        search_space = get_search_space(task_type, model_key)
        if not search_space:
            logger.info(
                f"Skipping HPO for '{model_key}': no search space defined (uses constructor defaults)."
            )
            continue

        # Skip degenerate clustering baselines
        if task_type == "Clustering" and model.get("silhouette") == -1.0:
            logger.info(
                f"Skipping HPO for '{model_key}': degenerate baseline clustering "
                f"(silhouette == -1.0). No point optimizing."
            )
            continue

        hpo_candidates.append(model)

    if not hpo_candidates:
        logger.warning(
            "No models passed the HPO selection filter. "
            "HPO skipped — pipeline will use baseline best model."
        )
        return []

    selected_keys = [m["model_key"] for m in hpo_candidates]
    logger.info(f"HPO will run on: {selected_keys}")
    return hpo_candidates   # list[dict] — NOT list[str]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — _build_optuna_objective
# ─────────────────────────────────────────────────────────────────────────────
def _build_optuna_objective(model_key, task_type, X_train, y_train, cv, registry_result):
    """
    Returns an Optuna objective function (closure) for one specific model.
    Captures everything needed so the function signature is just objective(trial) -> float.
    n_jobs=1 inside cross_val_score is mandatory — Optuna owns the process.
    """
    search_space  = get_search_space(task_type, model_key)
    scoring_tuple = registry_result["scoring_metric"]
    scoring_str   = scoring_tuple[0]
    scorer_type   = scoring_tuple[2]   # "sklearn" or "custom"
    direction     = scoring_tuple[1]   # "minimize" or "maximize"
    model_def = registry_result["models"].get(model_key, {})
    def objective(trial) -> float:
        # 1. Build params dict from trial suggestions
        params = {}
        for param_def in search_space:
            name   = param_def["name"]
            p_type = param_def["type"]

            if p_type == "float":
                log = param_def.get("log", False)
                params[name] = trial.suggest_float(
                    name, param_def["low"], param_def["high"], log=log
                )
            elif p_type == "int":
                params[name] = trial.suggest_int(
                    name, param_def["low"], param_def["high"]
                )
            elif p_type == "categorical":
                params[name] = trial.suggest_categorical(
                    name, param_def["choices"]
                )

        # 2. Constraint: LogisticRegression penalty ↔ solver
        if model_key == "logistic_regression":
            penalty = params.get("penalty", "l2")
            if penalty in ("l1", "elasticnet") and params.get("solver") != "saga":
                params["solver"] = "saga"
            if penalty == "elasticnet" and "l1_ratio" not in params:
                params["l1_ratio"] = 0.5

        # 3. Constraint: AgglomerativeClustering ward linkage → euclidean only
        if model_key == "agglomerative" and params.get("linkage") == "ward":
            if "affinity" in params:
                params["affinity"] = "euclidean"

        # 4. Suppress boosting model verbosity inside trials
        if model_key == "xgboost":
            params["verbosity"] = 0
        if model_key == "lightgbm":
            params["verbose"] = -1
            params["verbosity"] = -1
            params["min_gain_to_split"] = 0.0

        # 5. Get fresh model instance with trial params
        model = get_model_instance(task_type, model_key, params)
        if task_type == "Regression" and y_train is not None:
            if (y_train > 0).all() and skew(y_train) > 0.75:
                model = TransformedTargetRegressor(
                    regressor=model,
                    func=np.log1p,
                    inverse_func=np.expm1
                )

        # 6. Score — branch on scorer_type
        try:
            if scorer_type == "sklearn":
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring=scoring_str, n_jobs=1,
                    error_score="raise"
                )
                score = float(np.mean(scores))

                # Guard against NaN scores from extreme hyperparams
                if np.isnan(score):
                    worst = float("inf") if direction == "minimize" else float("-inf")
                    logger.warning(
                        f"[{model_key}] Trial produced NaN score. "
                        f"Returning worst-case value {worst}."
                    )
                    return worst

            elif scorer_type == "custom":
                # Clustering — no cross-val, fit on full X_train
                if model_def.get("supports_predict"):
                    model.fit(X_train)
                    labels = model.predict(X_train)
                else:
                    labels = model.fit_predict(X_train)

                if model_def.get("returns_noise_label"):
                    valid_mask = labels != -1
                    if not valid_mask.any() or len(set(labels[valid_mask])) < 2:
                        return -1.0   # worst silhouette — prune this trial
                    score = float(silhouette_score(X_train[valid_mask], labels[valid_mask]))
                else:
                    n_unique = len(set(labels))
                    if n_unique < 2:
                        return -1.0
                    score = float(silhouette_score(X_train, labels))
            else:
                raise ValueError(f"Unknown scorer_type: '{scorer_type}'")

        except Exception as e:
            # Let Optuna handle the failed trial via catch=(Exception,) in study.optimize
            raise e

        return score

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — _run_single_optuna_study
# ─────────────────────────────────────────────────────────────────────────────
def _run_single_optuna_study(
    model_key, task_type, X_train, y_train, cv,
    registry_result, n_trials, timeout_per_model, direction
) -> dict:
    """
    Creates and runs one complete Optuna study for one model.
    Deletes the study object after extraction to prevent memory leaks.
    Returns a result dict — never raises.
    """
    # optuna.logging.set_level(optuna.logging.WARNING)
    start_time = time.time()

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
        )
    )

    objective = _build_optuna_objective(
        model_key, task_type, X_train, y_train, cv, registry_result
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_per_model,
        catch=(Exception,),       # don't let one bad trial abort entire study
        show_progress_bar=True   # logger handles progress
    )

    study_duration_s = time.time() - start_time
    trials = study.trials
    n_completed = len([t for t in trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned    = len([t for t in trials if t.state == optuna.trial.TrialState.PRUNED])
    n_failed    = len([t for t in trials if t.state == optuna.trial.TrialState.FAIL])

    # baseline_score set to None here — run_hpo_studies overrides it correctly on line 289
    baseline_score = None

    try:
        best_trial  = study.best_trial   # raises ValueError if no complete trials
        best_params = best_trial.params
        best_score  = best_trial.value

        logger.info(
            f"[{model_key}] Optuna study complete | "
            f"trials: {len(trials)} | completed: {n_completed} | "
            f"pruned: {n_pruned} | failed: {n_failed}"
        )
        logger.info(
            f"[{model_key}] Best score: {best_score:.5f} | Best params: {best_params}"
        )

        result = {
            "model_key":          model_key,
            "status":             "success",
            "fail_reason":        None,
            "best_params":        best_params,
            "best_score":         best_score,
            "baseline_score":     baseline_score,
            "n_trials_completed": n_completed,
            "n_trials_pruned":    n_pruned,
            "n_trials_failed":    n_failed,
            "direction":          direction,
            "study_duration_s":   study_duration_s,
        }

    except ValueError as e:
        # Triggered when all trials failed/pruned or timeout was hit before any trial completed
        logger.error(
            f"[{model_key}] Optuna study failed to find a valid trial: {str(e)}"
        )
        result = {
            "model_key":          model_key,
            "status":             "failed",
            "fail_reason":        str(e),
            "best_params":        {},
            "best_score":         None,
            "baseline_score":     baseline_score,
            "n_trials_completed": n_completed,
            "n_trials_pruned":    n_pruned,
            "n_trials_failed":    n_failed,
            "direction":          direction,
            "study_duration_s":   study_duration_s,
        }

    finally:
        del study   # explicit delete to prevent memory accumulation across models

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — run_hpo_studies
# ─────────────────────────────────────────────────────────────────────────────
def run_hpo_studies(
    selected_models: list[dict],
    X_train,
    y_train,
    task_type: str,
    registry_result: dict,
    n_trials: int,
    timeout_per_model: int
) -> list[dict]:
    """
    Runs HPO studies sequentially (NOT parallel) for all selected models.
    Optuna's TPE sampler is CPU-intensive — parallel studies cause oversubscription.
    Returns all study results including failures so build_hpo_leaderboard has full picture.
    If ALL studies fail, returns empty list to trigger baseline fallback.
    """
    if not selected_models:
        logger.info("No models selected for HPO. Returning empty study results.")
        return []

    scoring_tuple = registry_result["scoring_metric"]
    direction     = scoring_tuple[1]   # "minimize" or "maximize"
    cv            = get_cv_strategy(task_type, y_train)

    study_results = []
    total_models  = len(selected_models)

    for i, model_entry in enumerate(selected_models):
        model_key = model_entry["model_key"]
        baseline  = model_entry.get("cv_mean")   # from trainer leaderboard

        logger.info(
            f"HPO [{i + 1}/{total_models}] Starting study for '{model_key}' | "
            f"trials: {n_trials} | timeout: {timeout_per_model}s"
        )

        wall_start = time.time()

        result = _run_single_optuna_study(
            model_key=model_key,
            task_type=task_type,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            registry_result=registry_result,
            n_trials=n_trials,
            timeout_per_model=timeout_per_model,
            direction=direction
        )

        # Override baseline and duration — single source of truth from run_hpo_studies
        result["baseline_score"]  = baseline
        result["study_duration_s"] = time.time() - wall_start

        if result["status"] == "failed":
            logger.warning(
                f"HPO [{i + 1}/{total_models}] Study for '{model_key}' failed: "
                f"{result.get('fail_reason')}"
            )
        else:
            logger.info(
                f"HPO [{i + 1}/{total_models}] '{model_key}' done in "
                f"{result['study_duration_s']:.1f}s"
            )

        study_results.append(result)

    # If ALL studies failed, return empty list to trigger baseline fallback
    successful_studies = [r for r in study_results if r["status"] == "success"]
    if not successful_studies:
        logger.critical(
            "All HPO studies failed. Returning empty list — "
            "pipeline will fall back to baseline best model."
        )
        return []

    logger.info(
        f"HPO studies complete | "
        f"{len(successful_studies)}/{total_models} succeeded."
    )
    return study_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — _retrain_with_best_params
# ─────────────────────────────────────────────────────────────────────────────
def _retrain_with_best_params(
    model_key: str,
    task_type: str,
    best_params: dict,
    X_train,
    X_test,
    y_train,
    y_test,
    registry_result: dict
) -> dict:
    """
    Retrains the model on the FULL X_train using the best HPO params.
    This is the model that gets evaluated and deployed — not the CV fold model from Optuna.
    n_jobs=-1 is restored for the final fit (no longer inside Optuna's process).
    y_test is passed in but NEVER used here — it belongs to evaluator.py only.
    """
    start    = time.time()
    t_type   = task_type.lower()
    model_def = registry_result.get("models", {}).get(model_key, {})

    try:
        # 1. Fresh instance with HPO params
        model = get_model_instance(task_type, model_key, best_params)
        if t_type == "regression" and y_train is not None:
            if (y_train > 0).all() and skew(y_train) > 0.75:
                model = TransformedTargetRegressor(
                    regressor=model,
                    func=np.log1p,
                    inverse_func=np.expm1
                )

        # 2. Restore n_jobs=-1 — we're no longer inside Optuna's sequential loop
        if hasattr(model, "n_jobs"):
            try:
                model.set_params(n_jobs=-1)
            except Exception:
                pass   # some models reject n_jobs via set_params — ignore safely

        # 3. Fit on full training set with convergence warning capture
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if t_type == "clustering":
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)

            for warn in caught:
                if issubclass(warn.category, ConvergenceWarning):
                    logger.warning(
                        f"[{model_key}] Convergence warning during final refit: {warn.message}"
                    )

        # 4. Predict on X_test
        y_pred  = None
        y_proba = None
        labels  = None

        if t_type == "regression":
            y_pred = model.predict(X_test)
            # Validate predictions — NaN/Inf means the model is broken
            if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                raise ValueError(
                    f"HPO-tuned '{model_key}' produced NaN/Inf predictions. "
                    "Check feature scaling and search space bounds."
                )

        elif t_type == "classification":
            y_pred = model.predict(X_test)
            if model_def.get("supports_predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception as e:
                    logger.warning(
                        f"[{model_key}] predict_proba failed despite flag=True: {e}"
                    )

        elif t_type == "clustering":
            if model_def.get("supports_predict"):
                labels = model.predict(X_train)
            else:
                labels = model.fit_predict(X_train)
            # y_pred stays None for clustering — evaluator uses labels

        refit_time_s = time.time() - start
        logger.info(
            f"[{model_key}] Refit complete in {refit_time_s:.2f}s"
        )

        return {
            "model_key":    model_key,
            "status":       "success",
            "fail_reason":  None,
            "fitted_model": model,
            "best_params":  best_params,
            "y_pred":       y_pred,
            "y_proba":      y_proba,
            "labels":       labels,
            "refit_time_s": refit_time_s,
            "supports_shap": model_def.get("supports_shap", False),
        }

    except MemoryError as e:
        logger.error(f"[{model_key}] MemoryError during final retraining: {e}")
        return {
            "model_key":    model_key,
            "status":       "failed",
            "fail_reason":  f"MemoryError: {e}",
            "fitted_model": None,
            "best_params":  best_params,
            "y_pred":       None,
            "y_proba":      None,
            "labels":       None,
            "refit_time_s": time.time() - start,
            "supports_shap": False,
        }

    except Exception as e:
        logger.error(f"[{model_key}] Error during final retraining: {e}")
        return {
            "model_key":    model_key,
            "status":       "failed",
            "fail_reason":  str(e),
            "fitted_model": None,
            "best_params":  best_params,
            "y_pred":       None,
            "y_proba":      None,
            "labels":       None,
            "refit_time_s": time.time() - start,
            "supports_shap": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — build_hpo_leaderboard
# ─────────────────────────────────────────────────────────────────────────────
def build_hpo_leaderboard(
    study_results: list[dict],
    retrain_results: list[dict],
    trainer_leaderboard: list[dict],
    task_type: str
) -> list[dict]:
    """
    Merges HPO study results with retrain results and the trainer baseline leaderboard.
    Every model from trainer_leaderboard appears in the output — HPO-tuned and baseline.
    Models not selected for HPO or that failed get hpo_improved=None and baseline scores.
    Sort key is guarded against None values.
    """
    study_map  = {r["model_key"]: r for r in study_results  if r.get("status") == "success"}
    retrain_map = {r["model_key"]: r for r in retrain_results if r.get("status") == "success"}
    t_type     = task_type.lower()

    # Infer direction — from study if any succeeded, else from task type
    if study_map:
        direction = next(iter(study_map.values())).get("direction", "maximize")
    else:
        direction = "minimize" if t_type == "regression" else "maximize"

    final_entries = []

    for trainer_entry in trainer_leaderboard:
        model_key      = trainer_entry["model_key"]
        baseline_score = trainer_entry.get("cv_mean", 0.0)
        study          = study_map.get(model_key)
        retrain        = retrain_map.get(model_key)

        entry = {
            "model_key":    model_key,
            "model_name":   trainer_entry.get("model_name", model_key),
            "model_type":   trainer_entry.get("model_type", "unknown"),
            "baseline_score": baseline_score,
            "supports_shap":  trainer_entry.get("supports_shap", False),
        }

        if study and retrain:
            # Successfully tuned and retrained
            hpo_score = study["best_score"]
            improvement = hpo_score - baseline_score if baseline_score is not None else 0.0

            entry["hpo_best_score"]  = hpo_score
            entry["improvement"]     = improvement
            entry["improvement_pct"] = (
                (improvement / abs(baseline_score)) * 100
                if baseline_score and baseline_score != 0 else 0.0
            )

            is_improvement = (
                    (direction == "maximize" and hpo_score > baseline_score) or
                    (direction == "minimize" and hpo_score > baseline_score)  # less negative = better RMSE
            )
            entry["hpo_improved"] = is_improvement

            if not is_improvement:
                logger.warning(
                    f"[{model_key}] HPO score ({hpo_score:.4f}) is worse than baseline "
                    f"({baseline_score:.4f}). Baseline params will be preferred."
                )

            entry["best_params"]        = study["best_params"]
            entry["n_trials_completed"] = study["n_trials_completed"]
            entry["study_duration_s"]   = study.get("study_duration_s", 0.0)
            entry["refit_time_s"]       = retrain.get("refit_time_s", 0.0)
            entry["supports_shap"]      = retrain.get("supports_shap", entry["supports_shap"])

            if t_type == "regression":
                entry["hpo_cv_rmse"] = abs(hpo_score)
                # improvement_pct in RMSE terms (positive = got better, negative = got worse)
                if baseline_score != 0:
                    baseline_rmse = abs(baseline_score)
                    hpo_rmse = abs(hpo_score)
                    entry["improvement_pct"] = ((baseline_rmse - hpo_rmse) / baseline_rmse) * 100
                    # positive % = RMSE went down = improved
                    # negative % = RMSE went up  = got worse

            if t_type in ("classification", "regression"):
                entry["fitted_model"] = retrain.get("fitted_model")
                entry["y_pred"]       = retrain.get("y_pred")
                entry["y_proba"]      = retrain.get("y_proba")

            if t_type == "clustering":
                entry["labels"] = retrain.get("labels")

        else:
            # Skipped HPO, study failed, or retraining failed — use baseline values
            entry["hpo_best_score"]     = baseline_score
            entry["improvement"]        = 0.0
            entry["improvement_pct"]    = 0.0
            entry["hpo_improved"]       = None   # None = not attempted / not determinable
            entry["best_params"]        = {}
            entry["n_trials_completed"] = 0
            entry["study_duration_s"]   = 0.0
            entry["refit_time_s"]       = 0.0

            if t_type == "regression":
                entry["hpo_cv_rmse"] = abs(baseline_score) if baseline_score is not None else 0.0

            if t_type in ("classification", "regression"):
                entry["fitted_model"] = None
                entry["y_pred"]       = None
                entry["y_proba"]      = None

            if t_type == "clustering":
                entry["labels"] = None

        final_entries.append(entry)

    # Sort — guard against None hpo_best_score to prevent TypeError
    final_entries.sort(
        key=lambda x: (
            x.get("hpo_best_score")
            if x.get("hpo_best_score") is not None
            else float("-inf")
        ),
        reverse=True  # ALWAYS descending — works for both minimize(neg) and maximize
    )

    for idx, entry in enumerate(final_entries):
        entry["rank"] = idx + 1

    return final_entries


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — save_hpo_models
# ─────────────────────────────────────────────────────────────────────────────
def save_hpo_models(hpo_leaderboard: list[dict], task_type: str, output_dir: str) -> dict:
    """
    Saves every HPO-tuned model, predictions, and best_params to disk.
    Continues saving remaining models if one fails — does not abort.
    "_hpo" suffix on model files distinguishes from trainer's saves.
    """
    out_dir        = Path(output_dir)
    models_dir     = out_dir / "models"
    best_params_dir = out_dir / "best_params"
    predictions_dir = out_dir / "predictions"
    clustering_dir  = out_dir / "clustering"

    models_dir.mkdir(parents=True, exist_ok=True)
    best_params_dir.mkdir(parents=True, exist_ok=True)

    t_type = task_type.lower()
    if t_type in ("classification", "regression"):
        predictions_dir.mkdir(parents=True, exist_ok=True)
    elif t_type == "clustering":
        clustering_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "saved_models":      {},
        "saved_predictions": {},
        "saved_params":      {},
        "save_failures":     [],
    }

    for entry in hpo_leaderboard:
        model_key   = entry["model_key"]
        best_params = entry.get("best_params", {})

        # 1. Save best_params JSON (even for skipped models if we have something)
        if best_params:
            params_path = best_params_dir / f"{model_key}_best_params.json"
            try:
                with open(params_path, "w") as f:
                    json.dump(_make_json_serializable(best_params), f, indent=4)
                result["saved_params"][model_key] = str(params_path)
            except Exception as e:
                logger.error(f"[{model_key}] Failed to save best_params: {e}")
                if model_key not in result["save_failures"]:
                    result["save_failures"].append(model_key)

        # 2. Save fitted model
        model = entry.get("fitted_model")
        if model is None:
            logger.warning(
                f"[{model_key}] No fitted_model to save "
                f"(HPO skipped or retraining failed)."
            )
            if model_key not in result["save_failures"]:
                result["save_failures"].append(model_key)
            continue   # skip predictions too — nothing was retrained

        model_path = models_dir / f"{model_key}_hpo.joblib"
        try:
            joblib.dump(model, model_path, compress=3)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(
                f"[{model_key}] Saved HPO model → {model_path} ({size_mb:.2f} MB)"
            )
            result["saved_models"][model_key] = str(model_path)
        except Exception as e:
            logger.error(f"[{model_key}] Failed to save model: {e}")
            if model_key not in result["save_failures"]:
                result["save_failures"].append(model_key)

        # 3. Save predictions or clustering labels
        try:
            if t_type in ("classification", "regression"):
                y_pred = entry.get("y_pred")
                if y_pred is not None:
                    pred_path = predictions_dir / f"{model_key}_y_pred.npy"
                    np.save(pred_path, y_pred)
                    result["saved_predictions"][f"{model_key}_y_pred"] = str(pred_path)

                y_proba = entry.get("y_proba")
                if y_proba is not None:
                    proba_path = predictions_dir / f"{model_key}_y_proba.npy"
                    np.save(proba_path, y_proba)
                    result["saved_predictions"][f"{model_key}_y_proba"] = str(proba_path)

            elif t_type == "clustering":
                labels = entry.get("labels")
                if labels is not None:
                    labels_path = clustering_dir / f"{model_key}_labels.npy"
                    np.save(labels_path, labels)
                    result["saved_predictions"][f"{model_key}_labels"] = str(labels_path)

        except Exception as e:
            logger.error(f"[{model_key}] Failed to save predictions/labels: {e}")
            if model_key not in result["save_failures"]:
                result["save_failures"].append(model_key)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — _make_json_serializable
# ─────────────────────────────────────────────────────────────────────────────
def _make_json_serializable(obj):
    """Recursively converts numpy types to native Python for json.dump."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — save_hpo_summary
# ─────────────────────────────────────────────────────────────────────────────
def save_hpo_summary(
    hpo_leaderboard: list[dict],
    study_results: list[dict],
    task_type: str,
    save_paths: dict,
    output_dir: str
) -> dict:
    """
    Writes hpo_summary.json.
    Always writes the file even when leaderboard is empty (failure state).
    Uses datetime.now() — datetime is imported as a class, not a module.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_type = task_type.lower()

    # 1. Top-level stats
    total_hpo_duration = sum(s.get("study_duration_s", 0.0) for s in study_results)
    direction = "maximize"
    n_trials_per_model = 0
    if study_results:
        direction = study_results[0].get("direction", "maximize")
        first = study_results[0]
        n_trials_per_model = (
            first.get("n_trials_completed", 0) +
            first.get("n_trials_pruned", 0) +
            first.get("n_trials_failed", 0)
        )

    # 2. Clean leaderboard for JSON (strip non-serializable objects)
    clean_leaderboard = []
    for entry in hpo_leaderboard:
        model_key = entry["model_key"]
        clean_entry = {
            "rank":               entry.get("rank"),
            "model_key":          model_key,
            "model_name":         entry.get("model_name"),
            "hpo_improved":       entry.get("hpo_improved"),
            "improvement_pct":    entry.get("improvement_pct"),
            "n_trials_completed": entry.get("n_trials_completed"),
            "best_params":        entry.get("best_params", {}),
            "study_duration_s":   entry.get("study_duration_s"),
            "refit_time_s":       entry.get("refit_time_s"),
            "model_path":         save_paths.get("saved_models", {}).get(model_key),
        }
        if t_type == "regression":
            clean_entry["hpo_cv_rmse"]      = entry.get("hpo_cv_rmse")
            base = entry.get("baseline_score", 0.0)
            clean_entry["baseline_cv_rmse"] = abs(base) if base is not None else None
        else:
            clean_entry["hpo_best_score"] = entry.get("hpo_best_score")
            clean_entry["baseline_score"] = entry.get("baseline_score")

        clean_leaderboard.append(clean_entry)

    # 3. Best model entry
    best_model = None
    if clean_leaderboard:
        top = clean_leaderboard[0]
        best_model = {
            "model_key":   top["model_key"],
            "model_name":  top["model_name"],
            "model_path":  top["model_path"],
            "best_params": top["best_params"],
        }
        if t_type == "regression":
            best_model["hpo_cv_rmse"] = top.get("hpo_cv_rmse")
        else:
            best_model["hpo_best_score"] = top.get("hpo_best_score")

    # 4. Study-level stats
    study_level_stats = [
        {
            "model_key":          s.get("model_key"),
            "n_trials_completed": s.get("n_trials_completed", 0),
            "n_trials_pruned":    s.get("n_trials_pruned", 0),
            "n_trials_failed":    s.get("n_trials_failed", 0),
            "study_duration_s":   s.get("study_duration_s", 0.0),
        }
        for s in study_results
    ]

    # 5. Assemble and write
    summary = {
        "task_type":             task_type,
        "hpo_timestamp":         datetime.now().isoformat(timespec="seconds"),
        "n_models_tuned":        len([e for e in clean_leaderboard if e.get("hpo_improved") is not None]),
        "n_trials_per_model":    n_trials_per_model,
        "total_hpo_duration_s":  total_hpo_duration,
        "direction":             direction,
        "leaderboard":           clean_leaderboard,
        "best_model":            best_model,
        "study_level_stats":     study_level_stats,
    }

    safe_summary  = _make_json_serializable(summary)
    summary_path  = out_dir / "hpo_summary.json"

    try:
        with open(summary_path, "w") as f:
            json.dump(safe_summary, f, indent=4)
        logger.info(f"HPO summary saved → {summary_path}")
    except Exception as e:
        logger.error(f"Failed to save hpo_summary.json: {e}")

    return safe_summary


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — get_final_best_model
# ─────────────────────────────────────────────────────────────────────────────
def get_final_best_model(
    hpo_leaderboard: list[dict],
    trainer_leaderboard: list[dict],
    task_type: str
) -> dict:
    """
    Selects the single best model by comparing top HPO candidate vs trainer baseline.
    Falls back to baseline if all HPO failed or no retrained model is available.
    Adds 'source' key: "hpo" | "baseline" for evaluator.py to know origin.
    """
    t_type = task_type.lower()

    if not trainer_leaderboard:
        raise ValueError(
            "get_final_best_model: trainer_leaderboard is empty. "
            "Pipeline cannot select a final model."
        )

    baseline_best  = trainer_leaderboard[0]
    baseline_score = baseline_best.get("cv_mean", 0.0)
    direction      = "minimize" if t_type == "regression" else "maximize"

    # Find best HPO entry that has a successfully retrained model
    best_hpo = None
    for entry in hpo_leaderboard:
        has_model = (
            entry.get("fitted_model") is not None if t_type != "clustering"
            else entry.get("labels") is not None
        )
        if has_model:
            best_hpo = entry
            break

    winner_source = "baseline"
    winner_entry  = baseline_best
    final_score   = baseline_score

    if best_hpo is not None:
        hpo_score = best_hpo.get("hpo_best_score", 0.0)
        hpo_wins  = (
            (direction == "maximize" and hpo_score > baseline_score) or
            (direction == "minimize" and hpo_score < baseline_score)
        )
        if hpo_wins:
            winner_source = "hpo"
            winner_entry  = best_hpo
            final_score   = hpo_score

    model_name = winner_entry.get("model_name", winner_entry.get("model_key", "Unknown"))
    score_str = f"{final_score:.5f}" if final_score is not None else "N/A"
    base_score_str = f"{baseline_score:.5f}" if baseline_score is not None else "N/A"
    
    if winner_source == "hpo":
        logger.info(
            f"Final best model: {model_name} (HPO) | "
            f"score: {score_str} (improved from baseline: {base_score_str})"
        )
    else:
        if best_hpo is not None:
            hpo_score = best_hpo.get("hpo_best_score", 0.0)
            hpo_str = f"{hpo_score:.5f}" if hpo_score is not None else "N/A"
            logger.info(
                f"Final best model: {model_name} (BASELINE) | "
                f"score: {score_str} (HPO did not improve — best HPO: {hpo_str})"
            )
        else:
            logger.info(
                f"Final best model: {model_name} (BASELINE) | "
                f"score: {score_str} (all HPO trials failed or skipped)"
            )

    result              = dict(winner_entry)
    result["source"]    = winner_source
    result["final_score"] = final_score

    # Guarantee all keys evaluator.py expects are present
    for key in ["fitted_model", "y_pred", "y_proba", "model_path", "best_params"]:
        result.setdefault(key, None)
    if t_type == "clustering":
        result.setdefault("labels", None)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — run_hpo  (single entry point for pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
def run_hpo(
    trainer_result: dict,
    X_train,
    X_test,
    y_train,
    y_test,
    task_type: str,
    registry_result: dict,
    n_trials: int,
    top_n_models: int,
    timeout_per_model: int,
    output_dir: str
) -> dict:
    """
    Single entry point called by pipeline.py.
    Chains all HPO steps in order. Returns full result dict for evaluator.py.
    """
    logger.info("=" * 60)
    logger.info(f"HPO STARTED | task: {task_type}")
    logger.info("=" * 60)

    # Step 1 — Validate
    _validate_hpo_inputs(
        trainer_result, X_train, X_test, y_train, y_test,
        task_type, n_trials, top_n_models, timeout_per_model,
        registry_result   # passed directly — NOT nested inside trainer_result
    )

    trainer_leaderboard = trainer_result.get("leaderboard", [])
    if not trainer_leaderboard:
        logger.warning("Trainer leaderboard is empty. Nothing to optimize.")
        return {
            "hpo_leaderboard":  [],
            "final_best_model": {},
            "study_results":    [],
            "save_paths":       {},
            "hpo_summary":      {},
        }

    # Step 2 — Select models for HPO
    selected_models = _select_models_for_hpo(
        leaderboard=trainer_leaderboard,
        top_n_models=top_n_models,
        task_type=task_type
    )

    # Step 5 — Run Optuna studies (calls Steps 3 & 4 internally)
    study_results = run_hpo_studies(
        selected_models=selected_models,
        X_train=X_train,
        y_train=y_train,
        task_type=task_type,
        registry_result=registry_result,
        n_trials=n_trials,
        timeout_per_model=timeout_per_model
    )

    # Step 6 — Retrain with best params for each successful study
    retrain_results = []
    for study in study_results:
        if study.get("status") == "success":
            logger.info(
                f"Retraining '{study['model_key']}' on full dataset with HPO params..."
            )
            retrain_res = _retrain_with_best_params(
                model_key=study["model_key"],
                task_type=task_type,
                best_params=study["best_params"],
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                registry_result=registry_result
            )
            retrain_results.append(retrain_res)
        else:
            logger.warning(
                f"Skipping retraining for '{study['model_key']}' (HPO study failed)."
            )

    # Step 7 — Build combined HPO leaderboard
    hpo_leaderboard = build_hpo_leaderboard(
        study_results=study_results,
        retrain_results=retrain_results,
        trainer_leaderboard=trainer_leaderboard,
        task_type=task_type
    )

    # Step 8 — Save models, predictions, params to disk
    save_paths = save_hpo_models(
        hpo_leaderboard=hpo_leaderboard,
        task_type=task_type,
        output_dir=output_dir
    )

    # Step 9 — Write hpo_summary.json
    hpo_summary = save_hpo_summary(
        hpo_leaderboard=hpo_leaderboard,
        study_results=study_results,
        task_type=task_type,
        save_paths=save_paths,
        output_dir=output_dir
    )

    # Step 10 — Select final best model (HPO vs baseline)
    final_best_model = get_final_best_model(
        hpo_leaderboard=hpo_leaderboard,
        trainer_leaderboard=trainer_leaderboard,
        task_type=task_type
    )

    logger.info("=" * 60)
    logger.info(
        f"HPO COMPLETED | Best: {final_best_model.get('model_name')} "
        f"({final_best_model.get('source', '').upper()})"
    )
    logger.info("=" * 60)

    return {
        "hpo_leaderboard":  hpo_leaderboard,
        "final_best_model": final_best_model,
        "study_results":    study_results,
        "save_paths":       save_paths,
        "hpo_summary":      hpo_summary,
    }