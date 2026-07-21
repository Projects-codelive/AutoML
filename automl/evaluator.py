import datetime
import json
import logging
import os
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, \
    mean_squared_log_error, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, \
    log_loss, confusion_matrix, classification_report, silhouette_score, davies_bouldin_score, calinski_harabasz_score, \
    roc_curve, auc, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize, StandardScaler

from automl.logger import get_logger

logger = get_logger("Model_Evaluation")

def _validate_evaluator_inputs(final_best_model, X_train, X_test, y_train, y_test, task_type):
    task_type = task_type.lower()
    is_supervised = task_type in ["classification", "regression"]
    if not final_best_model or not isinstance(final_best_model, dict):
        raise RuntimeError("Evaluator received no final model. HPO or trainer failed upstream.")
    if final_best_model.get("fitted_model") is None:
        raise RuntimeError("fitted_model is None. Please check HPO logs for upstream training/tuning failures.")
    if is_supervised:
        if final_best_model.get("y_pred") is None:
            raise ValueError("y_pred cannot be None for supervised tasks.")
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test cannot be None for supervised tasks.")
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must have matching lengths.")
    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("X_train and X_test must have the same columns in the same order.")
    if is_supervised:
        y_pred = final_best_model.get("y_pred")
        if len(y_pred) != len(y_test):
            raise ValueError("Length of y_pred does not match length of y_test.")
        if task_type == "regression":
            if not pd.api.types.is_numeric_dtype(y_test):
                raise ValueError("y_test must be numeric for regression tasks.")
            if pd.isna(y_pred).any() or np.isinf(y_pred).any():
                logging.critical("y_pred contains NaN or Inf values. These will corrupt downstream metrics.")
                raise ValueError("y_pred contains NaN or Inf values.")
        elif task_type == "classification":
            if len(pd.unique(y_test)) < 2:
                raise ValueError(
                    "y_test must have at least 2 unique classes. A single-class test set produces entirely misleading metrics.")
    elif task_type == "clustering":
        labels = final_best_model.get("labels")
        if labels is None:
            raise ValueError("Clustering tasks require 'labels' in the final_best_model dictionary.")
        if X_train is not None and len(labels) != len(X_train):
            raise ValueError("Length of clustering labels must match the length of X_train.")
    return None

def _evaluate_regression(y_test, y_pred, y_train, model, X_test, target_col):
    warnings_list = []
    def log_and_store_warning(msg: str):
        logging.warning(msg)
        warnings_list.append(msg)
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    # Primary Metrix
    mae = float(mean_absolute_error(y_test_arr, y_pred_arr))
    mse = float(mean_squared_error(y_test_arr, y_pred_arr))
    rmse = float(root_mean_squared_error(y_test_arr, y_pred_arr))
    r2 = float(r2_score(y_test_arr, y_pred_arr))
    #Adjusted R2
    n = len(y_test_arr)
    p = X_test.shape[1]
    adj_r2_denom = n - p - 1
    if adj_r2_denom <= 0:
        adj_r2 = None
        log_and_store_warning("Adjusted R² cannot be computed because (n - p - 1) <= 0.")
    else:
        adj_r2 = float(1 - (1 - r2) * (n - 1) / adj_r2_denom)
    # MAPE(Mean Absolute Percentage Error)
    if np.any(y_test_arr == 0):
        mape = None
        log_and_store_warning("MAPE cannot be computed because y_test contains zeros (division by zero).")
    else:
        mape = float(np.mean(np.abs((y_test_arr - y_pred_arr) / y_test_arr)) * 100)
    # RMSLE (Root Mean Squared Log Error)
    if np.any(y_test_arr <= 0) or np.any(y_pred_arr <= 0):
        rmsle = None
    else:
        rmsle = float(np.sqrt(mean_squared_log_error(y_test_arr, y_pred_arr)))
    # Residuals analysis
    residuals = y_test_arr - y_pred_arr
    residual_mean = float(np.mean(residuals))
    residual_std = float(np.std(residuals))
    residual_skew = float(scipy.stats.skew(residuals))
    residual_kurt = float(scipy.stats.kurtosis(residuals))
    if abs(residual_skew) > 1:
        log_and_store_warning("Heteroscedastic errors detected — residual skew is far from 0.")
    # Overfitting Check
    train_r2_raw = cross_val_score(model, X_test, y_test, cv=3, scoring='r2').mean()
    overfitting_gap = train_r2_raw - r2
    if overfitting_gap > 0.1:
        log_and_store_warning("Possible overfitting detected — train R² significantly higher than test R²")
    if r2 < 0:
        log_and_store_warning("Negative R² — model performs worse than predicting the mean")
    if rmse > float(np.std(y_test_arr)):
        log_and_store_warning("RMSE exceeds target standard deviation — model has low predictive power")
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "adj_r2": adj_r2,
        "mape": mape,
        "rmsle": rmsle,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "residual_skew": residual_skew,
        "residual_kurt": residual_kurt,
        "overfitting_gap": overfitting_gap,
        "warnings": warnings_list
    }


def _evaluate_classification(y_test, y_pred, y_proba, model, X_test):
    warnings_list = []
    def log_and_store_warning(msg: str):
        logging.warning(msg)
        warnings_list.append(msg)
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    n_classes = len(np.unique(y_test_arr))
    is_binary = bool(n_classes == 2)

    accuracy = float(accuracy_score(y_test_arr, y_pred_arr))
    f1_weighted = float(f1_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0))
    f1_macro = float(f1_score(y_test_arr, y_pred_arr, average='macro', zero_division=0))
    f1_micro = float(f1_score(y_test_arr, y_pred_arr, average='micro', zero_division=0))
    precision = float(precision_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0))
    recall = float(recall_score(y_test_arr, y_pred_arr, average='weighted', zero_division=0))
    mcc = float(matthews_corrcoef(y_test_arr, y_pred_arr))
    if len(np.unique(y_pred_arr)) == 1:
        log_and_store_warning("All predictions belong to a single class. MCC is 0 by definition.")

    auc = None
    logloss = None
    if y_proba is not None:
        y_proba_arr = np.array(y_proba)
        # Edge case: y_proba has shape (n, 1) instead of (n, 2) for binary classifications
        if is_binary and (y_proba_arr.ndim == 1 or y_proba_arr.shape[1] == 1):
            y_proba_arr = y_proba_arr.reshape(-1, 1)
            y_proba_arr = np.hstack([1 - y_proba_arr, y_proba_arr])
        try:
            if is_binary:
                auc = float(roc_auc_score(y_test_arr, y_proba_arr[:, 1]))
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            logloss = float(log_loss(y_test_arr, y_proba_arr))
        except Exception as e:
            log_and_store_warning(f"Error computing probabilistic metrics (AUC/LogLoss): {str(e)}")
    else:
        logging.info("AUC skipped: model has no predict_proba")

    cm = confusion_matrix(y_test_arr, y_pred_arr).tolist()
    cm_normalized = confusion_matrix(y_test_arr, y_pred_arr, normalize='true').tolist()

    full_report = classification_report(y_test_arr, y_pred_arr, output_dict=True, zero_division=0)
    per_class_report = {
        key: value for key, value in full_report.items()
        if key not in ['accuracy', 'macro avg', 'weighted avg']
    }

    cv_acc_raw = cross_val_score(model, X_test, y_test_arr, cv=3, scoring='accuracy').mean()
    overfitting_gap = float(cv_acc_raw - accuracy)
    if overfitting_gap > 0.1:
        log_and_store_warning("Possible overfitting detected — train accuracy significantly higher than test accuracy")
    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "auc_roc": auc,
        "log_loss": logloss,
        "n_classes": n_classes,
        "is_binary": is_binary,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
        "per_class_report": per_class_report,
        "overfitting_gap": overfitting_gap,
        "warnings": warnings_list,
    }


def _evaluate_clustering(labels, X_train, model) -> dict:
    """
    Computes internal clustering quality metrics, handling noise points
    and returning purely JSON-serializable Python data types.
    """
    warnings_list = []

    def log_and_store_warning(msg: str):
        logging.warning(msg)
        warnings_list.append(msg)

    # Standardize inputs to numpy arrays for boolean indexing
    labels_arr = np.array(labels)
    if hasattr(X_train, "values"):
        # Handle Pandas DataFrame/Series safely without importing pandas
        X_train_arr = X_train.values
    else:
        X_train_arr = np.array(X_train)

    # 1. Noise handling (DBSCAN, HDBSCAN, etc. use -1 for noise)
    noise_mask = labels_arr == -1
    n_noise = int(noise_mask.sum())
    noise_pct = float(n_noise / len(labels_arr) * 100)

    valid_mask = ~noise_mask
    X_valid = X_train_arr[valid_mask]
    labels_valid = labels_arr[valid_mask]

    # Calculate number of valid clusters (excluding -1)
    n_clusters = len(set(labels_valid))

    # 2. Guard against catastrophic clustering failures
    if n_clusters < 2:
        logging.critical("Only 1 cluster found — clustering has failed. Consider different algorithm or parameters.")
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "noise_pct": noise_pct,
            "cluster_sizes": {},
            "cluster_size_mean": 0.0,
            "cluster_size_std": 0.0,
            "imbalance_ratio": 0.0,
            "warnings": warnings_list,
        }

    # 3. Compute Internal Metrics
    sample_size = min(5000, len(X_valid))
    silhouette = float(silhouette_score(X_valid, labels_valid, sample_size=sample_size))
    davies_bouldin = float(davies_bouldin_score(X_valid, labels_valid))
    calinski_harabasz = float(calinski_harabasz_score(X_valid, labels_valid))

    # 4. Cluster Distribution Analytics
    unique_labels, counts = np.unique(labels_valid, return_counts=True)

    # Cast numpy types to native Python ints to guarantee JSON serialization downstream
    cluster_sizes = {int(cluster_id): int(count) for cluster_id, count in zip(unique_labels, counts)}

    counts_list = list(cluster_sizes.values())
    cluster_size_mean = float(np.mean(counts_list))
    cluster_size_std = float(np.std(counts_list))

    # 5. Balance Check
    largest = max(counts_list)
    smallest = min(counts_list)

    # Prevent division by zero mathematically, though smallest >= 1 if n_clusters >= 2
    imbalance_ratio = float(largest / smallest) if smallest > 0 else float('inf')

    if imbalance_ratio > 10:
        log_and_store_warning("Highly imbalanced clusters detected")

    return {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        "noise_pct": noise_pct,
        "cluster_sizes": cluster_sizes,
        "cluster_size_mean": cluster_size_mean,
        "cluster_size_std": cluster_size_std,
        "imbalance_ratio": imbalance_ratio,
        "warnings": warnings_list,
    }


def _plot_regression_diagnostics(y_test, y_pred, target_col, output_dir):
    plot_dir = Path(output_dir) / "plots"
    try:
        plot_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create plot directory {plot_dir}: {str(e)}")
        # If we can't create the directory, we can't save any plots
        return {
            "predicted_vs_actual": None,
            "residuals_vs_predicted": None,
            "residuals_distribution": None,
            "residuals_qq_plot": None,
        }
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    residuals = y_test_arr - y_pred_arr
    paths = {
        "predicted_vs_actual": None,
        "residuals_vs_predicted": None,
        "residuals_distribution": None,
        "residuals_qq_plot": None,
    }
    # Plot 1: Predicted vs Actual
    try:
        plt.figure(figsize=(10, 6), dpi=100)
        plt.scatter(y_test_arr, y_pred_arr, alpha=0.4)
        y_min, y_max = y_test_arr.min(), y_test_arr.max()
        plt.plot([y_min, y_max], [y_min, y_max], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Predicted vs Actual: {target_col}")
        out_path = plot_dir / "predicted_vs_actual.png"
        plt.savefig(out_path, bbox_inches='tight')
        paths["predicted_vs_actual"] = str(out_path)
    except Exception as e:
        logging.error(f"Failed to generate Predicted vs Actual plot: {str(e)}")
    finally:
        plt.close()

    # Plot 2 — Residuals vs Predicted:
    try:
        plt.figure(figsize=(10, 6), dpi=100)
        plt.scatter(y_pred_arr, residuals, alpha=0.4)
        plt.axhline(0, color='red', linestyle='--')

        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title(f"Residuals vs Predicted: {target_col}")

        out_path = plot_dir / "residuals_vs_predicted.png"
        plt.savefig(out_path, bbox_inches='tight')
        paths["residuals_vs_predicted"] = str(out_path)
    except Exception as e:
        logging.error(f"Failed to generate Residuals vs Predicted plot: {str(e)}")
    finally:
        plt.close()

    # Plot 3: Residuals Distribution
    try:
        plt.figure(figsize=(10, 6), dpi=100)
        # density=True is required to correctly overlay the normal PDF
        _, bins, _ = plt.hist(residuals, bins=50, density=True, alpha=0.6)
        # Overlay normal curve
        mu, std = np.mean(residuals), np.std(residuals)
        x = np.linspace(bins[0], bins[-1], 100)
        p = scipy.stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)

        plt.xlabel("Residual Value")
        plt.ylabel("Frequency")
        plt.title(f"Residuals Distribution: {target_col}")

        out_path = plot_dir / "residuals_distribution.png"
        plt.savefig(out_path, bbox_inches='tight')
        paths["residuals_distribution"] = str(out_path)
    except Exception as e:
        logging.error(f"Failed to generate Residuals Distribution plot: {str(e)}")
    finally:
        plt.close()

        # Plot 4: Q-Q Plot
    try:
        plt.figure(figsize=(10, 6), dpi=100)
        scipy.stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of Residuals: {target_col}")
        out_path = plot_dir / "residuals_qq_plot.png"
        plt.savefig(out_path, bbox_inches='tight')
        paths["residuals_qq_plot"] = str(out_path)
    except Exception as e:
        logging.error(f"Failed to generate Q-Q plot: {str(e)}")
    finally:
        plt.close()
    return paths


def _plot_classification_diagnostics(y_test, y_pred, y_proba, class_names, output_dir) -> dict:
    # Create output directory
    plot_dir = Path(output_dir) / "plots"
    try:
        plot_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create plot directory {plot_dir}: {str(e)}")
        return {
            "confusion_matrix": None,
            "roc_curve": None,
            "precision_recall_curve": None,
            "per_class_f1": None,
        }

    # Standardize arrays and handle strings
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    class_names_str = [str(c) for c in class_names]
    unique_classes = np.unique(y_test_arr)
    n_classes = len(unique_classes)
    is_binary = (n_classes == 2)

    paths = {
        "confusion_matrix": None,
        "roc_curve": None,
        "precision_recall_curve": None,
        "per_class_f1": None,
    }

    # Plot 1: Confusion Matrix heatmap (Raw & Normalized)
    try:
        cm = confusion_matrix(y_test_arr, y_pred_arr)
        cm_normalized = confusion_matrix(y_test_arr, y_pred_arr, normalize='true')

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=class_names_str, yticklabels=class_names_str)
        axes[0].set_title('Confusion Matrix (Raw)')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                    xticklabels=class_names_str, yticklabels=class_names_str)
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_ylabel('Actual')
        axes[1].set_xlabel('Predicted')

        out_path = plot_dir / "confusion_matrix.png"
        plt.savefig(out_path, bbox_inches='tight')
        paths["confusion_matrix"] = str(out_path)
    except Exception as e:
        logging.error(f"Failed to generate Confusion Matrix plot: {str(e)}")
    finally:
        plt.close()

    # Probability based plots (ROC & PR Curves)
    if y_proba is not None:
        y_proba_arr = np.array(y_proba)

        # Binary edge case: reshape if needed
        if is_binary and (y_proba_arr.ndim == 1 or y_proba_arr.shape[1] == 1):
            y_proba_arr = y_proba_arr.reshape(-1, 1)
            y_proba_arr = np.hstack([1 - y_proba_arr, y_proba_arr])

        # Shape guard for multiclass
        if not is_binary and y_proba_arr.shape[1] != n_classes:
            logging.error(
                f"y_proba shape {y_proba_arr.shape} does not match n_classes {n_classes}. Skipping ROC/PR plots.")
        else:
            # Plot 2: ROC Curve
            try:
                plt.figure(figsize=(10, 6), dpi=100)
                if is_binary:
                    # Treat the second class as positive
                    fpr, tpr, _ = roc_curve(y_test_arr, y_proba_arr[:, 1], pos_label=unique_classes[1])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                else:
                    Y_bin = label_binarize(y_test_arr, classes=unique_classes)
                    for i, cls_name in enumerate(class_names_str):
                        fpr, tpr, _ = roc_curve(Y_bin[:, i], y_proba_arr[:, i])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, lw=2, label=f'{cls_name} (AUC = {roc_auc:.2f})')

                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.legend(loc="lower right")

                out_path = plot_dir / "roc_curve.png"
                plt.savefig(out_path, bbox_inches='tight')
                paths["roc_curve"] = str(out_path)
            except Exception as e:
                logging.error(f"Failed to generate ROC Curve: {str(e)}")
            finally:
                plt.close()

            # Plot 3: Precision-Recall Curve
            try:
                plt.figure(figsize=(10, 6), dpi=100)
                if is_binary:
                    precision, recall, _ = precision_recall_curve(y_test_arr, y_proba_arr[:, 1],
                                                                  pos_label=unique_classes[1])
                    plt.plot(recall, precision, lw=2, label=f'PR curve')
                else:
                    Y_bin = label_binarize(y_test_arr, classes=unique_classes)
                    for i, cls_name in enumerate(class_names_str):
                        precision, recall, _ = precision_recall_curve(Y_bin[:, i], y_proba_arr[:, i])
                        plt.plot(recall, precision, lw=2, label=f'{cls_name}')

                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend(loc="lower left")

                out_path = plot_dir / "precision_recall_curve.png"
                plt.savefig(out_path, bbox_inches='tight')
                paths["precision_recall_curve"] = str(out_path)
            except Exception as e:
                logging.error(f"Failed to generate Precision-Recall Curve: {str(e)}")
            finally:
                plt.close()

    # Plot 4: Per-class F1 bar chart
    try:
        plt.figure(figsize=(10, 6), dpi=100)

        # zero_division=0 mimics your previous classification evaluation logic
        f1_scores = f1_score(y_test_arr, y_pred_arr, average=None, zero_division=0)

        plt.barh(class_names_str, f1_scores, color='skyblue', edgecolor='black')
        plt.xlabel('F1 Score')
        plt.ylabel('Class Label')
        plt.title('Per-Class F1 Score')
        plt.xlim(0, 1.0)

        out_path = plot_dir / "per_class_f1.png"
        plt.savefig(out_path, bbox_inches='tight')
        paths["per_class_f1"] = str(out_path)
    except Exception as e:
        logging.error(f"Failed to generate Per-Class F1 chart: {str(e)}")
    finally:
        plt.close()
    return paths


def _plot_clustering_diagnostics(labels: np.ndarray, X_train, feature_names: list, output_dir: str) -> dict:
    # Ensure inputs are numpy arrays for consistent indexing
    X_train = np.asarray(X_train)
    labels = np.asarray(labels)
    # Set up directories and return structure
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    result_paths = {
        "cluster_sizes": None,
        "cluster_pca_scatter": None,
        "cluster_feature_heatmap": None,
    }
    n_samples, n_features = X_train.shape
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    counts_dict = dict(zip(unique_labels, label_counts))
    # Separate valid clusters from noise (-1)
    valid_clusters = [lbl for lbl in unique_labels if lbl != -1]
    n_valid_clusters = len(valid_clusters)
    # Plot 1: Cluster Size Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    # Plot valid clusters
    if valid_clusters:
        cluster_sizes = [counts_dict[lbl] for lbl in valid_clusters]
        ax.bar(valid_clusters, cluster_sizes, color='skyblue', edgecolor='black', label='Clusters')
        # Ensure x-ticks align with actual cluster IDs
        ax.set_xticks(valid_clusters)
    # Plot noise points separately in grey
    if -1 in counts_dict:
        ax.bar([-1], [counts_dict[-1]], color='grey', edgecolor='black', label='Noise (-1)')
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Size Distribution")
    ax.legend()
    sizes_path = os.path.join(plots_dir, "cluster_sizes.png")
    fig.tight_layout()
    fig.savefig(sizes_path)
    plt.close(fig)
    result_paths["cluster_sizes"] = sizes_path
    # --- Edge Case: Skip PCA and Heatmap if n_clusters < 2 ---
    if n_valid_clusters < 2:
        logging.warning("Fewer than 2 valid clusters found. Skipping PCA and Feature Heatmap.")
        return result_paths
    # Plot 2: PCA Scatter (2D Projection)
    if n_samples < 3:
        logging.warning(f"Only {n_samples} samples available. Skipping PCA scatter (requires >= 3).")
    else:
        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_train)
            fig, ax = plt.subplots(figsize=(8, 6))
            # Plot valid clusters
            valid_mask = (labels != -1)
            scatter = ax.scatter(X_pca[valid_mask, 0], X_pca[valid_mask, 1],
                                 c=labels[valid_mask], cmap='tab10',
                                 alpha=0.7, edgecolors='w', linewidth=0.5)
            # Plot noise as black 'x' (drawn last so it stays visible)
            noise_mask = (labels == -1)
            if np.any(noise_mask):
                ax.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1],
                           c='black', marker='x', label='Noise', alpha=0.6)
            # Setup legend combining both clusters and noise
            legend_elements, _ = scatter.legend_elements(title="Clusters")
            if np.any(noise_mask):
                # We extract the handle for the noise plot manually to add to legend
                handles, labels_text = ax.get_legend_handles_labels()
                legend_elements.extend(handles)
            ax.legend(handles=legend_elements, loc='best')
            ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
            ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
            ax.set_title("2D PCA Projection of Clusters")
            pca_path = os.path.join(plots_dir, "cluster_pca_scatter.png")
            fig.tight_layout()
            fig.savefig(pca_path)
            plt.close(fig)
            result_paths["cluster_pca_scatter"] = pca_path
        except Exception as e:
            logging.error(f"Failed to generate PCA scatter plot: {e}")
    # Plot 3: Feature Means per Cluster (Heatmap)
    if n_features < 2:
        logging.warning("Fewer than 2 features available. Skipping cluster feature heatmap.")
    else:
        try:
            # Calculate mean of each feature for every valid cluster
            cluster_means = np.array([
                np.mean(X_train[labels == c], axis=0) for c in valid_clusters
            ])
            # Compute variance of these means across clusters to find top features
            feature_variances = np.var(cluster_means, axis=0)
            # Select top 15 features by variance
            n_top_features = min(15, n_features)
            top_indices = np.argsort(feature_variances)[::-1][:n_top_features]
            top_means = cluster_means[:, top_indices]
            top_feature_names = [feature_names[i] for i in top_indices]
            # Z-score normalize per column (Standardize the means)
            scaler = StandardScaler()
            top_means_scaled = scaler.fit_transform(top_means)
            # Generate Heatmap
            fig, ax = plt.subplots(figsize=(10, max(4, n_valid_clusters * 0.8)))
            sns.heatmap(top_means_scaled,
                        xticklabels=top_feature_names,
                        yticklabels=valid_clusters,
                        cmap='coolwarm',
                        annot=False,
                        cbar_kws={'label': 'Z-score of Feature Mean'})
            ax.set_xlabel("Features (Top 15 by variance across clusters)")
            ax.set_ylabel("Cluster ID")
            ax.set_title("Cluster Feature Means (Column-normalized)")
            heatmap_path = os.path.join(plots_dir, "cluster_feature_heatmap.png")
            fig.tight_layout()
            fig.savefig(heatmap_path)
            plt.close(fig)
            result_paths["cluster_feature_heatmap"] = heatmap_path
        except Exception as e:
            logging.error(f"Failed to generate feature heatmap: {e}")
    return result_paths


def _compute_fairness_metrics(y_test, y_pred, X_test, task_type, feature_names) -> dict:
    result = {
        "slices": [],
        "disparity_warnings": [],
        "has_disparity": False,
        "skipped": False
    }
    if task_type.lower() == "clustering":
        result["skipped"] = True
        return result
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    x_test = np.asarray(X_test)
    sliceable_cols = []
    for idx, col_name in enumerate(feature_names):
        # print(idx, x_test[:, idx])
        unique_vals = np.unique(x_test[:, idx])
        if 1<= len(unique_vals) <= 10:
            sliceable_cols.append(col_name)
        if len(sliceable_cols) >= 10:
            break
    if not sliceable_cols:
        result["skipped"] = True
        return result
    if task_type.lower() == "regression":
        overall_metric = np.sqrt(mean_squared_error(y_test, y_pred))
    else:
        overall_metric = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    col_metrics = {}
    for idx, col_name in enumerate(sliceable_cols):
        col_metrics[col_name] = []
        col_idx = feature_names.index(col_name)
        unique_vals = np.unique(x_test[:, col_idx])
        for val in unique_vals:
            mask = (x_test[:, col_idx] == val)
            slice_size = np.sum(mask)
            if slice_size < 30:
                continue
            y_test_slice = y_test[mask]
            y_pred_slice = y_pred[mask]
            # Clean up the value for output (handles numpy types for serialization)
            clean_val = float(val) if np.issubdtype(type(val), np.number) else str(val)
            slice_record = {
                "column": col_name,
                "value": clean_val,
                "slice_size": int(slice_size)
            }
            if task_type.lower() == 'regression':
                slice_rmse = np.sqrt(mean_squared_error(y_test_slice, y_pred_slice))
                slice_mae = mean_absolute_error(y_test_slice, y_pred_slice)
                slice_record["slice_rmse"] = float(slice_rmse)
                slice_record["slice_mae"] = float(slice_mae)
                col_metrics[col_name].append(slice_rmse)
            else:  # Classification
                slice_f1 = f1_score(y_test_slice, y_pred_slice, average='weighted', zero_division=0)
                slice_acc = accuracy_score(y_test_slice, y_pred_slice)
                slice_record["slice_f1"] = float(slice_f1)
                slice_record["slice_acc"] = float(slice_acc)
                # Edge case: Handle single-class slices where F1 can be misleading
                if len(np.unique(y_test_slice)) == 1:
                    slice_record["single_class_slice"] = True
                else:
                    slice_record["single_class_slice"] = False
                col_metrics[col_name].append(slice_f1)
            result["slices"].append(slice_record)
            # 4. Performance disparity check
    for col_name, metrics in col_metrics.items():
        if len(metrics) < 2:
            continue  # Need at least 2 valid slices to compare disparities
        max_metric = max(metrics)
        min_metric = min(metrics)
        disparity = max_metric - min_metric
        # Check if disparity exceeds 20% of the overall metric baseline
        threshold = 0.20 * overall_metric
        # For regression, RMSE is better when lower, but disparity is absolute distance.
        if disparity > threshold:
            warning_msg = f"Model performs significantly differently across '{col_name}' groups"
            logging.warning(warning_msg)
            result["disparity_warnings"].append(warning_msg)
            result["has_disparity"] = True
    return result


def _compute_prediction_intervals(model, X_test, y_pred, task_type: str, y_test=None) -> dict:
    """
    Estimates prediction uncertainty/confidence intervals depending on the task type
    and model architecture.
    """
    result = {
        "method": "skipped",
        "pred_lower": None,
        "pred_upper": None,
        "pred_std": None,
        "coverage_95": None,
        "confidence": None,
        "n_low_confidence": None,
        "pct_low_confidence": None,
        "skipped": False,
    }

    task = task_type.lower()

    # Edge case: Skip for Clustering
    if task == 'clustering':
        result["skipped"] = True
        return result

    # Ensure numpy arrays
    X_test = np.asarray(X_test)
    y_pred = np.asarray(y_pred)
    if y_test is not None:
        y_test = np.asarray(y_test)

    # ---------------------------------------------------------
    # Classification: Probability-based Confidence
    # ---------------------------------------------------------
    if task == 'classification':
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
                confidence = np.max(y_proba, axis=1)

                low_confidence_mask = confidence < 0.6
                n_low_confidence = int(low_confidence_mask.sum())
                pct_low_confidence = float(n_low_confidence / len(confidence) * 100)

                if pct_low_confidence > 20.0:
                    logging.warning(f"High percentage of low confidence predictions: {pct_low_confidence:.1f}%")

                result.update({
                    "method": "proba_confidence",
                    "confidence": confidence.tolist(),
                    "n_low_confidence": n_low_confidence,
                    "pct_low_confidence": pct_low_confidence
                })
            except Exception as e:
                logging.warning(f"Failed to compute predict_proba: {e}")
                result["skipped"] = True
        else:
            result["skipped"] = True

    # ---------------------------------------------------------
    # Regression: Prediction Intervals
    # ---------------------------------------------------------
    elif task == 'regression':
        pred_lower, pred_upper, pred_std = None, None, None
        method = "skipped"

        # Method 1: Tree Variance for Random Forest, Extra Trees, etc.
        if hasattr(model, 'estimators_'):
            try:
                # Get individual predictions from all trees
                tree_preds = np.array([tree.predict(X_test) for tree in model.estimators_])
                pred_std_arr = tree_preds.std(axis=0)

                pred_lower = y_pred - 1.96 * pred_std_arr
                pred_upper = y_pred + 1.96 * pred_std_arr
                pred_std = pred_std_arr.tolist()
                method = "tree_variance"
            except Exception as e:
                logging.info(
                    f"Could not compute tree variance prediction intervals (e.g. incompatible tree structure): {e}. Falling back to residual std.")

        # Method 2: Fallback to Residual Standard Deviation (Linear Models, XGBoost, etc.)
        if method == "skipped" and y_test is not None:
            try:
                residual_std = float(np.std(y_test - y_pred))
                pred_lower = y_pred - 1.96 * residual_std
                pred_upper = y_pred + 1.96 * residual_std
                pred_std = [residual_std] * len(y_pred)  # Constant std for all predictions
                method = "residual_std"
            except Exception as e:
                logging.warning(f"Could not compute residual std prediction intervals: {e}")

        # Finalize regression intervals
        if method != "skipped":
            # Cap pred_lower at 0 if the target domain appears strictly non-negative
            if (y_test is not None and np.all(y_test >= 0)) or np.all(y_pred >= 0):
                pred_lower = np.maximum(0, pred_lower)

            result.update({
                "method": method,
                "pred_lower": pred_lower.tolist(),
                "pred_upper": pred_upper.tolist(),
                "pred_std": pred_std
            })

            # Coverage check
            if y_test is not None:
                coverage = float(np.mean((y_test >= pred_lower) & (y_test <= pred_upper)))
                result["coverage_95"] = coverage
                logging.info(f"95% prediction interval coverage: {coverage * 100:.1f}% (ideal: ~95%)")
        else:
            result["skipped"] = True

    return result


def _make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    # Removed np.int_ as it is deprecated in NumPy 2.0
    elif isinstance(obj, (np.intc, np.intp, np.int8, np.int16, np.int32,
                          np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    # Removed np.float_ as it is deprecated in NumPy 2.0
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return _make_json_serializable(obj.tolist())
    elif pd.isna(obj) if 'pd' in globals() else obj != obj:  # Handle NaN
        return None
    else:
        return obj


def save_evaluation_artifacts(metrics: dict, plot_paths: dict, fairness: dict,
                              prediction_intervals: dict, final_best_model: dict,
                              task_type: str, output_dir: str) -> dict:
    """
    Saves the complete evaluation summary JSON and returns all artifact paths.
    Strips out large per-sample arrays to keep the JSON lightweight.
    """
    # 1. Process Prediction Intervals (exclude large arrays, compute summary stats)
    pi_summary = {
        "method": prediction_intervals.get("method"),
        "coverage_95": prediction_intervals.get("coverage_95"),
        "n_low_confidence": prediction_intervals.get("n_low_confidence"),
        "pct_low_confidence": prediction_intervals.get("pct_low_confidence"),
        "skipped": prediction_intervals.get("skipped", False)
    }

    # Calculate mean and std for the pred_std array if it exists
    pred_std = prediction_intervals.get("pred_std")
    if pred_std is not None and len(pred_std) > 0:
        pi_summary["pred_std_mean"] = float(np.mean(pred_std))
        pi_summary["pred_std_std"] = float(np.std(pred_std))

    # 2. Build the full summary structure
    eval_summary = {
        "task_type": task_type.capitalize(),
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "model_key": final_best_model.get("model_key", "unknown"),
        "model_name": final_best_model.get("model_name", "Unknown Model"),
        "model_source": final_best_model.get("source", "unknown"),
        "metrics": metrics,
        "fairness": fairness,
        "prediction_intervals": pi_summary,
        "plot_paths": plot_paths
    }

    # 3. Ensure JSON serialization safety
    clean_eval_summary = _make_json_serializable(eval_summary)

    # 4. Save JSON file
    summary_path = os.path.join(output_dir, "evaluator_summary.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump(clean_eval_summary, f, indent=4)

    # 5. Return paths dictionary expected by downstream modules
    plots_dir = os.path.join(output_dir, "plots")

    return {
        "summary_path": summary_path,
        "plots_dir": plots_dir,
        "all_plot_paths": plot_paths
    }


def get_evaluation_summary(evaluator_summary: dict) -> str:
    """
    Generates a human-readable one-paragraph text summary of the evaluation results.
    """
    task_type = evaluator_summary.get("task_type", "").lower()
    model_name = evaluator_summary.get("model_name", "Unknown Model")
    metrics = evaluator_summary.get("metrics", {})
    fairness = evaluator_summary.get("fairness", {})

    n_test = metrics.get("n_test_samples", metrics.get("n_test", "unknown"))

    # Evaluate overfitting gap (Classification and Regression)
    gap = metrics.get("overfitting_gap")
    if gap is not None:
        if gap < 0.1:
            overfitting_text = f"The model shows good generalization with an overfitting gap of {gap:.3f}."
        else:
            overfitting_text = f"Warning: possible overfitting detected (gap: {gap:.3f})."
    else:
        overfitting_text = ""

    # Evaluate fairness disparity (Classification and Regression)
    has_disparity = fairness.get("has_disparity", False)
    if not has_disparity:
        fairness_text = "No significant fairness disparities were detected."
    else:
        fairness_text = "Performance disparities were detected across some feature groups — review fairness section for details."

    # Construct the summary based on the task type
    if task_type == 'regression':
        rmse = metrics.get("rmse", 0.0)
        r2 = metrics.get("r2", 0.0)

        summary = (
            f"The {model_name} model achieved an RMSE of {rmse:,.0f} and R² of {r2:.3f} "
            f"on the held-out test set of {n_test} samples. "
            f"{overfitting_text} {fairness_text}"
        )

    elif task_type == 'classification':
        acc = metrics.get("accuracy", 0.0)
        f1 = metrics.get("f1_weighted", 0.0)
        n_classes = metrics.get("n_classes", "unknown")

        summary = (
            f"The {model_name} model achieved {acc:.1%} accuracy and weighted F1 of {f1:.3f} "
            f"on {n_test} test samples across {n_classes} classes. "
            f"{overfitting_text} {fairness_text}"
        )

    elif task_type == 'clustering':
        n_clusters = metrics.get("n_clusters", "unknown")
        silhouette = metrics.get("silhouette_score", 0.0)

        summary = (
            f"The {model_name} algorithm identified {n_clusters} clusters with a silhouette "
            f"score of {silhouette:.3f}."
        )

    else:
        summary = f"Completed evaluation for {model_name} on {task_type} task."

    # Clean up any potential double spaces from empty substrings
    return " ".join(summary.split())


def run_evaluator(final_best_model: dict, X_train, X_test, y_train, y_test,
                  task_type: str, target_col: str, feature_names: list, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    task = task_type.lower()

    # 1. Validate Inputs
    _validate_evaluator_inputs(final_best_model, X_train, X_test, y_train, y_test, task)

    model = final_best_model.get("fitted_model")

    metrics = {}
    plot_paths = {}
    y_pred = final_best_model.get("y_pred")

    # 2-4 & 5-7. Evaluate and Plot based on task type
    if task == 'regression':
        y_pred = model.predict(X_test)
        metrics = _evaluate_regression(y_test, y_pred, y_train, model, X_test, target_col)
        plot_paths = _plot_regression_diagnostics(y_test, y_pred, target_col, output_dir)

    elif task == 'classification':
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        metrics = _evaluate_classification(y_test, y_pred, y_proba, model, X_test)
        classes = getattr(model, 'classes_', np.unique(y_test))
        plot_paths = _plot_classification_diagnostics(y_test, y_pred, y_proba, classes, output_dir)

    elif task == 'clustering':
        labels = final_best_model.get("labels")
        if labels is None:
            if hasattr(model, 'labels_'):
                labels = model.labels_
            elif hasattr(model, 'predict'):
                labels = model.predict(X_train)
            else:
                labels = np.zeros(len(X_train))
        metrics = _evaluate_clustering(labels, X_train, model)
        plot_paths = _plot_clustering_diagnostics(labels, X_train, feature_names, output_dir)

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # 8. Compute Fairness Metrics
    fairness = _compute_fairness_metrics(y_test, y_pred, X_test, task, feature_names)

    # 9. Compute Prediction Intervals
    prediction_intervals = _compute_prediction_intervals(model, X_test, y_pred, task, y_test)

    # 10. Save Evaluation Artifacts
    artifact_paths = save_evaluation_artifacts(
        metrics=metrics,
        plot_paths=plot_paths,
        fairness=fairness,
        prediction_intervals=prediction_intervals,
        final_best_model=final_best_model,
        task_type=task_type,
        output_dir=output_dir
    )

    # Load the written evaluator_summary.json to generate the summary text
    with open(artifact_paths["summary_path"], "r") as f:
        evaluator_summary_json = json.load(f)

    # 11. Generate Evaluation Summary Text
    evaluation_summary_text = get_evaluation_summary(evaluator_summary_json)

    return {
        "metrics": metrics,
        "plot_paths": plot_paths,
        "fairness": fairness,
        "prediction_intervals": prediction_intervals,
        "evaluation_summary": evaluation_summary_text,
        "artifact_paths": artifact_paths,
        "evaluator_summary": evaluator_summary_json
    }