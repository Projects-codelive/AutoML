from automl.eda import run_basic_cleaning, run_eda
from automl.task_detector import load_data, determine_type_of_PS
from automl.logger import get_logger
logger = get_logger("pipeline")


def run_pipeline(path: str, target_col: str = None, output_dir: str = "artifacts/eda"):
    # Step 1 — Load
    df = load_data(path)
    if df is None:
        return None

    # Step 2 — Clean
    cleaned_df, cleaning_report = run_basic_cleaning(df)

    # Step 3 — Detect task type
    task_type = determine_type_of_PS(cleaned_df, target_col=target_col)
    logger.info(f"Detected task type: {task_type}")  # add this so you can see it in logs

    # Step 4 — EDA (receives already-cleaned df)
    eda_summary = run_eda(cleaned_df, task_type, target_col, cleaning_report=cleaning_report, output_dir=output_dir)
    # eda_summary["cleaning_report"] = cleaning_report  # attach cleaning report here

    return eda_summary