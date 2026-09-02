# automl/cv_utils.py

from scipy.stats import skew
from sklearn.model_selection import KFold, StratifiedKFold
from automl.logger import get_logger

logger = get_logger("cv_utils")

def get_cv_strategy(task_type: str, y_train, n_splits: int = 5):
    """
    Returns the correct CV splitter for the task type.
    Used by both model_trainer.py and hpo.py.

    Returns:
        KFold            — for Regression
        StratifiedKFold  — for Classification
        None             — for Clustering (no CV)
    """
    if task_type == "Clustering":
        logger.info("CV skipped for Clustering — scoring uses full dataset.")
        return None

    if n_splits < 2:
        raise ValueError("Cross-validation requires at least 2 folds.")

    if len(y_train) < n_splits:
        n_splits = min(5, len(y_train))
        logger.warning(
            f"n_splits reduced to {n_splits} — fewer samples than requested folds."
        )

    if task_type == "Regression":
        skewness = skew(y_train)
        logger.debug(f"Target skewness: {skewness:.2f}. Using KFold with shuffle.")
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    if task_type == "Classification":
        class_counts = y_train.value_counts()
        min_class = int(class_counts.min()) if len(class_counts) > 0 else 1
        if min_class < 2 or n_splits > min_class:
            n_splits_safe = max(2, min(n_splits, min_class))
            if min_class < 2 or n_splits_safe < 2:
                logger.warning(f"Class count too small for StratifiedKFold (min class count = {min_class}). Falling back to KFold.")
                n_splits_safe = max(2, min(n_splits, len(y_train)))
                return KFold(n_splits=n_splits_safe, shuffle=True, random_state=42)
            logger.warning(f"n_splits reduced to {n_splits_safe} for StratifiedKFold due to class count.")
            return StratifiedKFold(n_splits=n_splits_safe, shuffle=True, random_state=42)
        
        distribution = y_train.value_counts(normalize=True)
        if (distribution < 0.10).any():
            logger.debug("Imbalanced classes detected. Using StratifiedKFold.")
        else:
            logger.debug("Balanced classes. Using StratifiedKFold.")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    raise ValueError(f"Unknown task_type: '{task_type}'.")