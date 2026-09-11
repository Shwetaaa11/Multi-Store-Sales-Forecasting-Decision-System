from pathlib import Path

import pandas as pd


def find_project_root(start_path=None):
    """
    Locate the project root by searching for expected project folders/files.
    """

    if start_path is None:
        start_path = Path.cwd()

    start_path = Path(start_path).resolve()

    candidates = [start_path] + list(start_path.parents)

    for path in candidates:
        if (
            (path / "data").is_dir()
            and (path / "notebooks").is_dir()
        ):
            return path

    raise FileNotFoundError(
        "Project root could not be located."
    )


def get_data_paths(project_root=None):
    """
    Return paths for raw and processed project data.
    """

    if project_root is None:
        project_root = find_project_root()

    project_root = Path(project_root)

    return {
        "root": project_root,
        "raw": project_root / "data" / "raw",
        "processed": project_root / "data" / "processed",
    }


def load_raw_data(project_root=None):
    """
    Load the primary raw datasets.
    """

    paths = get_data_paths(project_root)

    train = pd.read_csv(paths["raw"] / "train.csv")
    features = pd.read_csv(paths["raw"] / "features.csv")
    stores = pd.read_csv(paths["raw"] / "stores.csv")
    test = pd.read_csv(paths["raw"] / "test.csv")

    return train, features, stores, test


def load_processed_data(
    filename,
    project_root=None
):
    """
    Load a processed CSV dataset.
    """

    paths = get_data_paths(project_root)

    file_path = paths["processed"] / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {filename}"
        )

    return pd.read_csv(file_path)


def save_processed_data(
    dataframe,
    filename,
    project_root=None
):
    """
    Save a DataFrame to the processed data directory.
    """

    paths = get_data_paths(project_root)

    paths["processed"].mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = paths["processed"] / filename

    dataframe.to_csv(
        output_path,
        index=False
    )

    return output_path


def validate_unique_keys(
    dataframe,
    columns
):
    """
    Validate that a set of columns forms a unique key.
    """

    duplicate_count = dataframe.duplicated(
        subset=columns
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate records "
            f"for key: {columns}"
        )

    return True


def validate_required_columns(
    dataframe,
    required_columns
):
    """
    Validate that all required columns are present.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True