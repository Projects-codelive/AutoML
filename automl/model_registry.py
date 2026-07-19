import copy
import json
import os
from datetime import datetime

import numpy as np
from sklearn import clone
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, \
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression, RidgeClassifier
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

from automl.logger import get_logger
logger = get_logger("model_registry")
_REGISTRY_CACHE = {}


try:
    from xgboost import XGBRegressor, XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost is not installed. 'xgboost' will be excluded from the model registry.")

try:
    from lightgbm import LGBMRegressor, LGBMClassifier

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM is not installed. 'lightgbm' will be excluded from the model registry.")


# Step 1 — _build_regression_models
def _build_regression_models() -> dict:
    registry = {
        "ridge": {
            "model": Ridge(alpha=1.0),
            "name": "Ridge Regression",
            "type": "linear",
            "supports_shap": True,
            "handles_nan": False,
            "tags": ["fast", "interpretable", "baseline"]
        },
        "lasso": {
            "model": Lasso(alpha=0.01, max_iter=5000),
            "name": "Lasso Regression",
            "type": "linear",
            "supports_shap": True,
            "handles_nan": False,
            "tags": ["fast", "interpretable", "feature_selection"]
        },
        "elastic_net": {
            "model": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
            "name": "Elastic Net",
            "type": "linear",
            "supports_shap": True,
            "handles_nan": False,
            "tags": ["fast", "interpretable", "regularized"]
        },
        "random_forest": {
            "model": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "name": "Random Forest",
            "type": "ensemble",
            "supports_shap": True,
            "handles_nan": False,
            "tags": ["strong baseline", "tree_based", "robust"]
        },
        "extra_trees": {
            "model": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "name": "Extra Trees",
            "type": "ensemble",
            "supports_shap": True,
            "handles_nan": False,
            "tags": ["fast", "tree_based", "handles_noise"]
        },
        "gradient_boosting": {
            "model": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "name": "Gradient Boosting",
            "type": "ensemble",
            "supports_shap": True,
            "handles_nan": False,
            "tags": ["strong", "tree_based", "sequential"]
        }
    }
    if HAS_XGBOOST:
        registry["xgboost"] = {
            "model": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0),
            "name": "XGBoost",
            "type": "ensemble",
            "supports_shap": True,
            "handles_nan": True,
            "tags": ["fast", "tree_based", "handles_missing", "gradient_boosting"]
        }
    if HAS_LIGHTGBM:
        registry["lightgbm"] = {
            "model": LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
            "name": "LightGBM",
            "type": "ensemble",
            "supports_shap": True,
            "handles_nan": True,
            "tags": ["fastest", "tree_based", "large_data", "handles_missing"]
        }
    logger.info(f"[ModelRegistry] Loaded 8 model(s) for task 'Regression': [...]")
    return registry

# Step 2 — _build_classification_models
def _build_classification_models() -> dict:
    registry = {
        "logistic_regression": {
            "model": LogisticRegression(multi_class="auto",max_iter=1000, random_state=42, n_jobs=-1),
            "name": "Logistic Regression",
            "type": "linear",
            "supports_shap": True,
            "handles_nan": False,
            "supports_predict_proba": True,
            "tags": ["fast", "interpretable", "probability_output", "baseline"]
        },
        "ridge_classifier": {
            "model": RidgeClassifier(),
            "name": "Ridge Classifier",
            "type": "linear",
            "supports_shap": True,
            "handles_nan": False,
            "supports_predict_proba": False,  # Critical: No predict_proba support
            "tags": ["fast", "interpretable", "many_class"]
        },
        "random_forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "name": "Random Forest",
            "type": "ensemble",
            "supports_shap": True,
            "handles_nan": False,
            "supports_predict_proba": True,
            "tags": ["strong baseline", "tree_based", "robust"]
        },
        "extra_trees": {
            "model": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "name": "Extra Trees",
            "type": "ensemble",
            "supports_shap": True,
            "handles_nan": False,
            "supports_predict_proba": True,
            "tags": ["fast", "tree_based", "less_overfit"]
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "name": "Gradient Boosting",
            "type": "ensemble",
            "supports_shap": True,
            "handles_nan": False,
            "supports_predict_proba": True,
            "tags": ["strong", "tree_based", "imbalanced_data"]
        },
        "knn": {
            "model": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "name": "K-Nearest Neighbors",
            "type": "neighbor",
            "supports_shap": False,  # Via KernelExplainer
            "handles_nan": False,
            "supports_predict_proba": True,
            "tags": ["non_parametric", "lazy_evaluation", "distance_based"]
        }
    }

    if HAS_XGBOOST:
        registry["xgboost"] = {
            "model": XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0, eval_metric="logloss"),
            "name": "XGBoost",
            "type": "ensemble",
            "supports_shap": True,
            "handles_nan": True,
            "supports_predict_proba": True,
            "tags": ["fast", "tree_based", "handles_missing", "native_multiclass"]
        }

    if HAS_LIGHTGBM:
        registry["lightgbm"] = {
            "model": LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
            "name": "LightGBM",
            "type": "ensemble",
            "supports_shap": True,
            "handles_nan": True,
            "supports_predict_proba": True,
            "tags": ["fastest", "tree_based", "large_data", "handles_missing"]
        }
    return registry

# Step 3 — _build_clustering_models
def _build_clustering_models() -> dict:
    registry = {
        "kmeans": {
            "model": KMeans(n_clusters=8, random_state=42, n_init=10),
            "name": "K-Means",
            "type": "clustering",
            "needs_n_clusters": True,
            "supports_predict": True,
            "hpo_param": "n_clusters",
            "tags": ["workhorse", "centroid_based", "distance_based"]
        },
        "minibatch_kmeans": {
            "model": MiniBatchKMeans(n_clusters=8, random_state=42, n_init=10),
            "name": "Mini-Batch K-Means",
            "type": "clustering",
            "needs_n_clusters": True,
            "supports_predict": True,
            "hpo_param": "n_clusters",
            "tags": ["fast", "large_data", "centroid_based"]
        },
        "agglomerative": {
            "model": AgglomerativeClustering(n_clusters=8),
            "name": "Agglomerative Clustering",
            "type": "clustering",
            "needs_n_clusters": True,
            "supports_predict": False,
            "hpo_param": "n_clusters",
            "tags": ["hierarchical", "bottom_up", "no_random_state"]
        },
        "gmm": {
            "model": GaussianMixture(n_components=8, random_state=42),
            "name": "Gaussian Mixture Model",
            "type": "clustering",
            "needs_n_clusters": True,
            "supports_predict": True,
            "hpo_param": "n_components",
            "tags": ["soft_clustering", "probabilistic", "bic_selection"]
        },
        "dbscan": {
            "model": DBSCAN(eps=0.5, min_samples=5),
            "name": "DBSCAN",
            "type": "clustering",
            "needs_n_clusters": False,
            "supports_predict": False,
            "hpo_param": "eps",
            "returns_noise_label": True,
            "tags": ["density_based", "arbitrary_shapes", "auto_n_clusters"]
        },
        "spectral": {
            "model": SpectralClustering(n_clusters=8, random_state=42, n_jobs=-1),
            "name": "Spectral Clustering",
            "type": "clustering",
            "needs_n_clusters": True,
            "supports_predict": False,  # Only has fit_predict()
            "hpo_param": "n_clusters",
            "max_rows_recommended": 5000, # Critical guardrail for O(N^3) time / O(N^2) memory complexity
            "tags": ["graph_based", "non_convex", "manifold"]
        }
    }
    return registry

# Step 4 — _build_regression_search_spaces
def _build_regression_search_spaces():
    hyperparameter = {
        "ridge": [
            {"name": "alpha", "type": "float", "low": 1e-3, "high": 1e3, "log": True}
        ],
        "lasso": [
            {"name": "alpha", "type": "float", "low": 1e-4, "high": 10, "log": True},
            {"name": "max_iter", "type": "int", "low": 1000, "high": 10000}
        ],
        "elastic_net": [
            {"name": "alpha", "type": "float", "low": 1e-4, "high": 10, "log": True},
            {"name": "l1_ratio", "type": "float", "low": 0.01, "high": 0.99}
        ],
        "random_forest": [
            {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
            {"name": "max_depth", "type": "int", "low": 3, "high": 20},
            {"name": "min_samples_split", "type": "int", "low": 2, "high": 20},
            {"name": "min_samples_leaf", "type": "int", "low": 1, "high": 10},
            {"name": "max_features", "type": "categorical", "choices": ["sqrt", "log2", 0.5, 0.7, 1.0]}
        ],
        "extra_trees": [
            {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
            {"name": "max_depth", "type": "int", "low": 3, "high": 20},
            {"name": "min_samples_split", "type": "int", "low": 2, "high": 20},
            {"name": "min_samples_leaf", "type": "int", "low": 1, "high": 10},
            {"name": "max_features", "type": "categorical", "choices": ["sqrt", "log2", 0.5, 0.7, 1.0]}
        ],
        "gradient_boosting": [
            {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
            {"name": "max_depth", "type": "int", "low": 3, "high": 20},
            {"name": "learning_rate", "type": "float", "low": 1e-3, "high": 0.3, "log": True},
            {"name": "subsample", "type": "float", "low": 0.5, "high": 1.0},
            {"name": "min_samples_leaf", "type": "int", "low": 1, "high": 10}
        ],
        "xgboost": [
            {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
            {"name": "max_depth", "type": "int", "low": 3, "high": 20},
            {"name": "learning_rate", "type": "float", "low": 1e-3, "high": 0.3, "log": True},
            {"name": "subsample", "type": "float", "low": 0.5, "high": 1.0},
            {"name": "colsample_bytree", "type": "float", "low": 0.4, "high": 1.0},
            {"name": "reg_alpha", "type": "float", "low": 1e-8, "high": 10, "log": True},
            {"name": "reg_lambda", "type": "float", "low": 1e-8, "high": 10, "log": True}
        ],
        "lightgbm": [
            {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
            {"name": "max_depth", "type": "int", "low": 3, "high": 20},
            {"name": "learning_rate", "type": "float", "low": 1e-3, "high": 0.3, "log": True},
            {"name": "num_leaves", "type": "int", "low": 20, "high": 300},
            {"name": "subsample", "type": "float", "low": 0.5, "high": 1.0},
            {"name": "colsample_bytree", "type": "float", "low": 0.4, "high": 1.0},
            {"name": "reg_alpha", "type": "float", "low": 1e-8, "high": 10, "log": True},
            {"name": "reg_lambda", "type": "float", "low": 1e-8, "high": 10, "log": True}
        ]
    }
    return hyperparameter


# Step 5 — _build_classification_search_spaces
def _get_shared_tree_spaces() -> dict:
    hyperparameter = {
        "random_forest": [
            {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
            {"name": "max_depth", "type": "int", "low": 3, "high": 20},
            {"name": "min_samples_split", "type": "int", "low": 2, "high": 20},
            {"name": "min_samples_leaf", "type": "int", "low": 1, "high": 10},
            {"name": "max_features", "type": "categorical", "choices": ["sqrt", "log2", 0.5, 0.7, 1.0]}
        ],
        "extra_trees": [
            {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
            {"name": "max_depth", "type": "int", "low": 3, "high": 20},
            {"name": "min_samples_split", "type": "int", "low": 2, "high": 20},
            {"name": "min_samples_leaf", "type": "int", "low": 1, "high": 10},
            {"name": "max_features", "type": "categorical", "choices": ["sqrt", "log2", 0.5, 0.7, 1.0]}
        ],
        "gradient_boosting": [
            {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
            {"name": "max_depth", "type": "int", "low": 3, "high": 20},
            {"name": "learning_rate", "type": "float", "low": 1e-3, "high": 0.3, "log": True},
            {"name": "subsample", "type": "float", "low": 0.5, "high": 1.0},
            {"name": "min_samples_leaf", "type": "int", "low": 1, "high": 10}
        ],
        "xgboost": [
            {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
            {"name": "max_depth", "type": "int", "low": 3, "high": 20},
            {"name": "learning_rate", "type": "float", "low": 1e-3, "high": 0.3, "log": True},
            {"name": "subsample", "type": "float", "low": 0.5, "high": 1.0},
            {"name": "colsample_bytree", "type": "float", "low": 0.4, "high": 1.0},
            {"name": "reg_alpha", "type": "float", "low": 1e-8, "high": 10, "log": True},
            {"name": "reg_lambda", "type": "float", "low": 1e-8, "high": 10, "log": True}
        ],
        "lightgbm": [
            {"name": "n_estimators", "type": "int", "low": 50, "high": 500},
            {"name": "max_depth", "type": "int", "low": 3, "high": 20},
            {"name": "learning_rate", "type": "float", "low": 1e-3, "high": 0.3, "log": True},
            {"name": "num_leaves", "type": "int", "low": 20, "high": 300},
            {"name": "subsample", "type": "float", "low": 0.5, "high": 1.0},
            {"name": "colsample_bytree", "type": "float", "low": 0.4, "high": 1.0},
            {"name": "reg_alpha", "type": "float", "low": 1e-8, "high": 10, "log": True},
            {"name": "reg_lambda", "type": "float", "low": 1e-8, "high": 10, "log": True}
        ]
    }
    return hyperparameter
def _build_classification_search_spaces() -> dict:
    params = _get_shared_tree_spaces()
    params['xgboost'] = params['xgboost'] + [
        {"name": "scale_pos_weight", "type": "float", "low": 1.0, "high": 100.0}
    ]
    params["lightgbm"] = params["lightgbm"] + [
        {"name": "is_unbalance", "type": "categorical", "choices": [True, False]}
    ]
    params["logistic_regression"] = [
        {"name": "C", "type": "float", "low": 1e-4, "high": 100.0, "log": True},
        {"name": "solver", "type": "categorical", "choices": ["lbfgs", "saga"]},
        {"name": "penalty", "type": "categorical", "choices": ["l2", "l1", "elasticnet"]}
    ]
    params["ridge_classifier"] = [
        {"name": "alpha", "type": "float", "low": 1e-3, "high": 1e3, "log": True}
    ]
    params["knn"] = [
        {"name": "n_neighbors", "type": "int", "low": 1, "high": 50},
        {"name": "weights", "type": "categorical", "choices": ["uniform", "distance"]},
        {"name": "metric", "type": "categorical", "choices": ["euclidean", "manhattan", "minkowski"]}
    ]
    return params


# Step 6 — _build_clustering_search_spaces
def _build_clustering_search_spaces()->dict:
    params = {
      "kmeans": [
        {"name": "n_clusters", "type": "int", "low": 2, "high": 15}
      ],
      "minibatch_kmeans": [
        {"name": "n_clusters", "type": "int", "low": 2, "high": 15},
        {"name": "batch_size", "type": "int", "low": 100, "high": 1000}
      ],
      "agglomerative": [
        {"name": "n_clusters", "type": "int", "low": 2, "high": 15},
        {"name": "linkage",    "type": "categorical", "choices": ["ward", "complete", "average"],
         "ward_constraint": True},
      ],
      "gmm": [
        {"name": "n_components",    "type": "int",         "low": 2,   "high": 15},
        {"name": "covariance_type", "type": "categorical", "choices": ["full", "tied", "diag", "spherical"]}
      ],
      "dbscan": [
        {"name": "eps",         "type": "float", "low": 0.01, "high": 2.0},
        {"name": "min_samples", "type": "int",   "low": 2,    "high": 30}
      ],
      "spectral": [
        {"name": "n_clusters", "type": "int", "low": 2, "high": 15},
        {"name": "affinity",   "type": "categorical", "choices": ["rbf", "nearest_neighbors"]}
      ]
    }
    return params


# Step 7 — get_models
def get_models(task_type: str):
    normalized_task = task_type.strip().title()
    if normalized_task in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[normalized_task]
    builders = {
        "Regression": _build_regression_models,
        "Classification": _build_classification_models,
        "Clustering": _build_clustering_models,
    }
    if normalized_task not in builders:
        valid_options = ", ".join(builders.keys())
        error_msg = (
            f"Invalid task_type: '{task_type}'. "
            f"Valid options are: {valid_options}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    registry = builders[normalized_task]()
    if not registry:
        error_msg = (
            f"Zero models survived for task_type='{normalized_task}'. "
            f"Check optional dependency installations (xgboost, lightgbm)."
        )
        logger.critical(error_msg)
        raise ValueError(error_msg)
    logger.info(
        f"[ModelRegistry] Loaded {len(registry)} model(s) for task '{normalized_task}': "
        f"{list(registry.keys())}"
    )
    _REGISTRY_CACHE[normalized_task] = registry
    return registry


# Step 8 — get_search_space
def get_search_space(task_type: str, model_key: str):
    normalized_task = task_type.strip().title()
    search_space_builders = {
        "Regression": _build_regression_search_spaces,
        "Classification": _build_classification_search_spaces,
        "Clustering": _build_clustering_search_spaces,
    }
    if normalized_task not in search_space_builders:
        valid_options = ", ".join(search_space_builders.keys())
        error_msg = f"Invalid task_type: '{task_type}'. Valid options are: {valid_options}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    task_search_spaces = search_space_builders[normalized_task]()
    if model_key not in task_search_spaces:
        logger.info(
            f"Model key '{model_key}' not found in {normalized_task} search spaces. "
            f"Returning empty list. (HPO will run with default parameters)."
        )
        return []
    return task_search_spaces[model_key]

# Step 9 — get_model_instance
def get_model_instance(task_type: str, model_key: str, params: dict = None):
    registry = get_models(task_type)

    if model_key not in registry:
        error_msg = (
            f"model_key='{model_key}' not found in '{task_type}' registry. "
            f"Available keys: {list(registry.keys())}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Deep-copy the base model so we start from a pristine default state
    base_model = copy.deepcopy(registry[model_key]["model"])

    if params is None or len(params) == 0:
        return base_model

    # Extract current default constructor params from the base model
    base_params = base_model.get_params()

    # Merge: defaults first, then HPO suggestions override specific keys only
    merged = {**base_params, **params}

    # ── Constraint enforcement for LogisticRegression ──────────────────────
    if task_type.strip().title() == "Classification" and model_key == "logistic_regression":
        penalty = merged.get("penalty", "l2")

        if penalty in ("l1", "elasticnet"):
            # lbfgs does not support l1 or elasticnet — force saga
            if merged.get("solver", "lbfgs") != "saga":
                logger.warning(
                    f"[ModelRegistry] penalty='{penalty}' requires solver='saga'. "
                    f"Overriding solver from '{merged.get('solver')}' → 'saga'."
                )
                merged["solver"] = "saga"

        if penalty == "elasticnet" and "l1_ratio" not in merged:
            # elasticnet also needs l1_ratio; default to 0.5 if hpo didn't suggest it
            merged["l1_ratio"] = 0.5
            logger.warning(
                "[ModelRegistry] penalty='elasticnet' requires l1_ratio. "
                "Defaulting to l1_ratio=0.5."
            )

    # Build a fresh instance with the merged params
    fresh_instance = base_model.__class__(**merged)
    return fresh_instance


# Step 10 — get_scoring_metric
def get_scoring_metric(task_type: str):
    normalized_task = task_type.strip().title()
    metrics_map = {
        "Regression": ("neg_root_mean_squared_error", "minimize", "sklearn"),
        "Classification": ("f1_weighted", "maximize", "sklearn"),
        "Clustering": ("silhouette", "maximize", "custom")
    }
    if normalized_task not in metrics_map:
        valid_options = ", ".join(metrics_map.keys())
        error_msg = f"Invalid task_type: '{task_type}'. Valid options are: {valid_options}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    metric_tuple = metrics_map[normalized_task]
    logger.info(
        f"[ModelRegistry] Scoring metric for '{normalized_task}': "
        f"metric={metric_tuple[0]}, direction={metric_tuple[1]}, source={metric_tuple[2]}"
    )
    return metric_tuple


# Step 11 — get_all_model_keys
def get_all_model_keys(task_type: str):
    models_registry = get_models(task_type)
    valid_keys = []
    for model_key, model_def in models_registry.items():
        if model_def.get("model") is not None:
            valid_keys.append(model_key)
        else:
            logger.debug(
                f"Excluding '{model_key}' from active training keys: "
                f"model object is None (likely due to a failed optional import)."
            )

    if not valid_keys:
        logger.critical(
            f"No valid models survived the import checks for task '{task_type}'. "
            f"The active training list is empty."
        )
    return valid_keys


# Step 12 - validate_registry
def validate_registry() -> dict:
    """
    Validates all model registries (Regression, Classification, Clustering) at startup.
    Verifies object instantiation, required keys, and runs a 10-row smoke test.

    Returns:
        dict: A health report of available and failed models.
    """
    report = {
        "regression": {"available": [], "failed": []},
        "classification": {"available": [], "failed": []},
        "clustering": {"available": [], "failed": []},
        "total_available": 0,
        "total_failed": 0
    }

    # 1. Generate 10-row synthetic datasets for smoke tests
    X_tiny = np.random.rand(10, 5)
    y_reg = np.random.rand(10)
    # Ensure at least 2 classes so GradientBoostingClassifier doesn't crash
    y_clf = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    task_types = ["Regression", "Classification", "Clustering"]

    for task in task_types:
        task_key = task.lower()

        # Rely on get_models (from Step 7) to fetch the definitions
        try:
            registry = get_models(task)
        except Exception as e:
            logger.error(f"Failed to load registry for {task}: {e}")
            continue

        for model_key, definition in registry.items():
            is_valid = True
            fail_reason = ""

            # --- Structural Validation ---
            model = definition.get("model")

            # Dynamically set required base keys based on task
            required_base_keys = ["name", "type", "supports_shap"]
            if task == "Clustering":
                # Clustering inherently does not support SHAP, so don't require the flag
                required_base_keys = ["name", "type"]

            if model is None:
                is_valid = False
                fail_reason = "Model object is None (likely a failed optional import)."

            elif not hasattr(model, "fit") or not callable(model.fit):
                is_valid = False
                fail_reason = "Model object is missing a callable 'fit' method."

            elif not all(k in definition for k in required_base_keys):
                is_valid = False
                fail_reason = f"Missing required base keys ({', '.join(required_base_keys)})."

            elif task == "Classification" and "supports_predict_proba" not in definition:
                is_valid = False
                fail_reason = "Missing 'supports_predict_proba' key."

            elif task == "Clustering" and not all(k in definition for k in ["needs_n_clusters", "supports_predict"]):
                is_valid = False
                fail_reason = "Missing clustering keys ('needs_n_clusters', 'supports_predict')."

            # --- Functional Validation (Smoke Test) ---
            if is_valid:
                try:
                    # Clone to avoid fitting the actual registry baseline model
                    smoke_model = clone(model)

                    # Edge Case: SpectralClustering fails if n_samples < n_clusters.
                    # Set cluster/component parameters to 2 for the 10-row test.
                    if task == "Clustering" and definition.get("needs_n_clusters"):
                        if hasattr(smoke_model, "n_clusters"):
                            smoke_model.set_params(n_clusters=2)
                        elif hasattr(smoke_model, "n_components"): # GMM uses n_components
                            smoke_model.set_params(n_components=2)

                    # Edge case: Suppress stubborn LightGBM console output on tiny data
                    if model_key == "lightgbm":
                        try:
                            smoke_model.set_params(force_col_wise=True, verbose=-1)
                        except Exception:
                            pass

                    # Execute fit
                    if task == "Regression":
                        smoke_model.fit(X_tiny, y_reg)
                    elif task == "Classification":
                        smoke_model.fit(X_tiny, y_clf)
                    else:
                        smoke_model.fit(X_tiny)

                    # Edge Case: DBSCAN returns all -1 on random noise. Log as info, not error.
                    if model_key == "dbscan" and hasattr(smoke_model, "labels_"):
                        if np.all(smoke_model.labels_ == -1):
                            logger.info(
                                f"[validate_registry] DBSCAN smoke test: all points "
                                f"labelled as noise (-1). Normal for random data — not a failure."
                            )

                except Exception as e:
                    is_valid = False
                    fail_reason = f"Smoke test failed: {e}"
                    definition["failed_smoke_test"] = True

            # --- Reporting ---
            if is_valid:
                report[task_key]["available"].append(model_key)
                report["total_available"] += 1
            else:
                report[task_key]["failed"].append({model_key: fail_reason})
                report["total_failed"] += 1
                logger.error(f"Registry Validation Failed | {task} -> {model_key}: {fail_reason}")

    # 3. Log the availability table
    logger.info("\n" + "=" * 47)
    logger.info(f"{'MODEL REGISTRY HEALTH CHECK':^47}")
    logger.info("=" * 47)
    for t in ["regression", "classification", "clustering"]:
        avail = len(report[t]["available"])
        fail = len(report[t]["failed"])
        logger.info(f"  {t.capitalize():<15} | Available: {avail:<3} | Failed: {fail:<3}")
    logger.info("-" * 47)
    logger.info(
        f"  {'TOTAL':<15} | Available: {report['total_available']:<3} "
        f"| Failed: {report['total_failed']:<3}"
    )
    logger.info("=" * 47)

    return report


# Step 13 — save_registry_manifest
def save_registry_manifest(output_dir: str = "artifacts/model_registry") -> None:
    """
    Saves a human-readable JSON manifest of all models in all three registries
    to artifacts/model_registry/registry_manifest.json.

    This becomes part of the model card and PDF report — it tells anyone reading
    the output exactly which algorithms were available.

    Args:
        output_dir: directory where the manifest JSON is written.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "regression_models": [],
        "classification_models": [],
        "clustering_models": [],
    }

    # Serialize each task's registry into a list of lightweight dicts
    task_map = {
        "regression_models": "Regression",
        "classification_models": "Classification",
        "clustering_models": "Clustering",
    }

    for manifest_key, task_type in task_map.items():
        try:
            registry = get_models(task_type)
        except Exception as exc:
            logger.error(
                f"[save_registry_manifest] Could not load registry for "
                f"'{task_type}': {exc}"
            )
            continue

        for model_key, definition in registry.items():
            entry = {
                "key": model_key,
                "name": definition.get("name", ""),
                "type": definition.get("type", ""),
                "supports_shap": definition.get("supports_shap", False),
                "handles_nan": definition.get("handles_nan", False),
                "tags": definition.get("tags", []),
            }

            # Add task-specific fields when present
            if "supports_predict_proba" in definition:
                entry["supports_predict_proba"] = definition["supports_predict_proba"]
            if "needs_n_clusters" in definition:
                entry["needs_n_clusters"] = definition["needs_n_clusters"]
            if "supports_predict" in definition:
                entry["supports_predict"] = definition["supports_predict"]
            if "hpo_param" in definition:
                entry["hpo_param"] = definition["hpo_param"]
            if "returns_noise_label" in definition:
                entry["returns_noise_label"] = definition["returns_noise_label"]
            if "max_rows_recommended" in definition:
                entry["max_rows_recommended"] = definition["max_rows_recommended"]

            manifest[manifest_key].append(entry)

        logger.info(
            f"[save_registry_manifest] {task_type}: "
            f"{len(manifest[manifest_key])} model(s) recorded."
        )

    # Write JSON
    manifest_path = os.path.join(output_dir, "registry_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    total = (
            len(manifest["regression_models"])
            + len(manifest["classification_models"])
            + len(manifest["clustering_models"])
    )
    logger.info(
        f"[save_registry_manifest] Manifest saved → '{manifest_path}' "
        f"({total} total models)."
    )


# Step 14 — run_model_registry  (pipeline entry point)
def get_search_spaces_for_task(task_type: str) -> dict:
    """
    Helper — returns the full search-space dict for a task type.
    run_model_registry() calls this; hpo.py can also call it directly if needed.

    Returns:
        dict keyed by model_key → list of param dicts.

    Raises:
        ValueError: on invalid task_type.
    """
    normalized_task = task_type.strip().title()
    builders = {
        "Regression": _build_regression_search_spaces,
        "Classification": _build_classification_search_spaces,
        "Clustering": _build_clustering_search_spaces,
    }

    if normalized_task not in builders:
        valid_options = ", ".join(builders.keys())
        error_msg = (
            f"Invalid task_type: '{task_type}'. "
            f"Valid options are: {valid_options}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    return builders[normalized_task]()


def run_model_registry(
        task_type: str,
        output_dir: str = "artifacts/model_registry",
) -> dict:
    """
    Single entry point called by pipeline.py (and optionally by trainer.py).
    Validates the registry, saves the manifest, and returns everything
    trainer.py / hpo.py need to start training.

    Args:
        task_type:  "Regression" | "Classification" | "Clustering"
        output_dir: directory for the manifest JSON artifact.

    Returns:
        {
            "models":         dict,   # model_key → model definition dict
            "search_spaces":  dict,   # model_key → list of param dicts
            "scoring_metric": tuple,  # ("metric", "direction", "source")
            "health_report":  dict,   # from validate_registry()
        }
    """
    logger.info(
        f"[run_model_registry] Starting registry initialisation "
        f"for task='{task_type}'."
    )

    health = validate_registry()
    save_registry_manifest(output_dir)
    models = get_models(task_type)
    search_spaces = get_search_spaces_for_task(task_type)
    scoring_metric = get_scoring_metric(task_type)

    logger.info(
        f"[run_model_registry] Done. "
        f"{len(models)} model(s) ready | "
        f"metric={scoring_metric[0]} | direction={scoring_metric[1]}"
    )

    return {
        "models": models,
        "search_spaces": search_spaces,
        "scoring_metric": scoring_metric,
        "health_report": health,
    }