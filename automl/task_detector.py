import pandas as pd
from pathlib import Path
from automl.logger import get_logger

logger = get_logger("task_detector")
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}
def load_data(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    extension = Path(path).suffix.lower()
    logger.info(f"load_data called | file: {path.name}")
    logger.debug(f"Full path resolved: {path.resolve()}")
    # if specified path is not exist
    if not path.exists():
        logger.error(f"File {path} not found.")
        return None
    # checking if extension is supported or not
    if extension not in SUPPORTED_EXTENSIONS:
        logger.error(f"Unsupported extension '{extension}' | "
            f"Supported: {SUPPORTED_EXTENSIONS}")
        return None
    try:
        if extension == '.csv':
            df = pd.read_csv(path)
            logger.debug("Used pd.read_csv")
        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
            logger.debug("Used pd.read_excel")
        elif extension == '.json':
            df = pd.read_json(path)
            logger.debug("Used pd.read_json")
    except Exception as e:
        logger.error(f'An Error Occured: {e}')
        return None
    # Checking if df is empty or not
    if df.empty:
        logger.warning("Empty DataFrame Found")
        return None
    if df.shape[1] < 2:
        logger.warning("For Analysis we need at least 2 columns")
        return None
    # Success Message
    logger.info("Data loaded successfully | "
        f"Shape: {df.shape[0]} rows × {df.shape[1]} cols | "
        f"File: {path.name}")
    return df

def determine_type_of_PS(data, target_col=None):
    if target_col is None:
        logger.info("No target column provided → Task Type: Clustering")
        return "Clustering"
    if target_col not in data.columns:
        logger.error(f"Target column '{target_col}' not found in data")
        raise ValueError(f'Column {target_col} not found in data.')
    target_data = data[target_col]
    uniques_values = target_data.nunique()
    total_rows = len(data)
    is_text = pd.api.types.is_object_dtype(target_data) or pd.api.types.is_string_dtype(target_data)
    is_category = (target_data.dtype.name == 'category')
    is_bool = pd.api.types.is_bool_dtype(target_data)
    if is_text or is_category or is_bool:
        return "Classification"
    elif pd.api.types.is_numeric_dtype(target_data):
        if uniques_values < 15 and (total_rows > uniques_values * 2):
            return "Clustering"
        else:
            return "Regression"
    else:
        return "Unknown"


# output = determine_type_of_PS(load_data(Path("../data/regression_sample.csv")), target_col="Price_Thousands")
# print(output)