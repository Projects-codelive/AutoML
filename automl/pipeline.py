import os
from pathlib import Path
from automl.eda import run_basic_cleaning, run_eda
from automl.feature_engineer import run_feature_engineering
from automl.model_registry import run_model_registry
from automl.task_detector import load_data, determine_type_of_PS
from automl.preprocessor import run_preprocessor
from automl.logger import get_logger
from dotenv import load_dotenv
load_dotenv()

logger = get_logger("pipeline")


def run_pipeline(
    path: str,
    target_col: str = None,
    eda_output_dir: str = "artifacts/eda",
    preprocessor_output_dir: str = "artifacts/preprocessor",
    feature_engineer_output_dir: str = "artifacts/feature_engineering",
    model_registry_output_dir: str = "artifacts/model_registry",
    api_key: str = None
):
    logger.info("=" * 60)
    logger.info("AUTOML PIPELINE STARTED")
    logger.info("=" * 60)

    # ── Step 1 — Load Data ─────────────────────────────────────────
    logger.info("STEP 1 — Loading Data")
    df = load_data(path)
    if df is None:
        logger.error("Pipeline aborted — data loading failed")
        return None
    logger.info(f"Loaded | shape: {df.shape} | columns: {df.columns.tolist()}")

    # ── Step 2 — Basic Cleaning ────────────────────────────────────
    logger.info("STEP 2 — Running Basic Cleaning")
    cleaned_df, cleaning_report = run_basic_cleaning(df)
    logger.info(
        f"Cleaning complete | "
        f"to_numeric: {cleaning_report['converted_to_numeric']} | "
        f"to_boolean: {cleaning_report['converted_to_boolean']} | "
        f"to_datetime: {cleaning_report['converted_to_datetime']} | "
        f"coerced: {cleaning_report['coerced_mostly_numeric']} | "
        f"unchanged: {cleaning_report['unchanged']}"
    )

    # ── Step 3 — Detect Task Type ──────────────────────────────────
    logger.info("STEP 3 — Detecting Task Type")
    task_type = determine_type_of_PS(cleaned_df, target_col=target_col)
    logger.info(f"Detected task type: {task_type} | target column: {target_col}")

    # ── Step 4 — EDA ───────────────────────────────────────────────
    logger.info("STEP 4 — Running EDA")
    eda_summary = run_eda(
        cleaned_df,
        task_type=task_type,
        target_col=target_col,
        cleaning_report=cleaning_report,
        output_dir=eda_output_dir
    )
    logger.info(f"EDA complete | artifacts saved to: {Path(eda_output_dir).resolve()}")
    logger.info(f"High missing cols  : {eda_summary.get('high_missing_cols', [])}")
    logger.info(f"Highly skewed cols : {eda_summary.get('highly_skewed_cols', [])}")
    logger.info(f"Multicollinear pairs: {eda_summary.get('correlation', {}).get('multicollinear_pairs', [])}")

    # ── Step 5 — Preprocessing ─────────────────────────────────────
    logger.info("STEP 5 — Running Preprocessing")
    preprocessor_result = run_preprocessor(
        df=cleaned_df,
        target_col=target_col,
        task_type=task_type,
        eda_summary=eda_summary,
        api_key=api_key,
        output_dir=preprocessor_output_dir
    )
    if preprocessor_result is None:
        logger.error("Pipeline aborted — preprocessing failed")
        return None
    logger.info(f"Preprocessing complete | artifacts saved to: {Path(preprocessor_output_dir).resolve()}")
    logger.info(f"X_train shape: {preprocessor_result['X_train'].shape}")
    logger.info(f"X_test shape : {preprocessor_result['X_test'].shape}")

    # ── Step 6 — Feature Engineering ──────────────────────────────
    logger.info("STEP 6 — Running Feature Engineering")
    fe_result = run_feature_engineering(
        X_train=preprocessor_result["X_train"],
        X_test=preprocessor_result["X_test"],
        y_train=preprocessor_result["y_train"],
        y_test=preprocessor_result["y_test"],
        task_type=task_type,
        target_col=target_col,
        eda_summary=eda_summary,
        api_key=api_key,
        output_dir=feature_engineer_output_dir
    )
    logger.info(f"Feature engineering complete | final X_train: {fe_result['X_train'].shape}")

    # ── Step 7 — Model Registry ──────────────────────────────
    logger.info("Step 7 — Model Registry")
    mr_result = run_model_registry(
        task_type=task_type,
        output_dir=model_registry_output_dir,
    )
    if mr_result is None:
        logger.error("Pipeline aborted — model registry failed")
        return None
    logger.info(f"Model Registry complete | artifacts saved to: {Path(model_registry_output_dir).resolve()}")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

    return {
        "eda_summary":                  eda_summary,
        "preprocessor_result":          preprocessor_result,
        "feature_engineering_result":   fe_result
    }


if __name__ == "__main__":
    result = run_pipeline(
        path="../data/used_cars.csv",
        target_col="price",
        eda_output_dir="../artifacts/eda",
        preprocessor_output_dir="../artifacts/preprocessor",
        feature_engineer_output_dir="../artifacts/feature_engineer",
        model_registry_output_dir= "../artifacts/model_registry",
        api_key=os.getenv("OPENAI_API_KEY")
    )