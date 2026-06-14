"""Generic help functions related to I/O, data manipulation, etc."""


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


def _get_unique_filepath(filepath):
    if "*" not in filepath:
        return filepath
    filepaths = glob.glob(filepath)
    if len(filepaths) != 1:
        raise ValueError(f"Couldn't find unique file matching {filepath}.")
    return filepaths[0]


def read_simulation_parameters(filepath,
                               exclude_cols=None,
                               rename_cols=None,
                               index_col="index",
                               ):
    """Read the simulation parameters from a CSV file

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing the simulation parameters.
        Can include wildcards, but must match exactly one file.
    exclude_cols : list, optional
        _description_, by default None
    rename_cols : list, optional
        _description_, by default None
    index_col : str, optional
        Column to use as the index of the DataFrame, by default "index"

    Returns
    -------
    pd.DataFrame
        DataFrame containing the simulation parameters.

    Raises
    ------
    ValueError
        If no unique file is found matching the filepath.

    """
    df = pd.read_csv(_get_unique_filepath(filepath), index_col=index_col)
    if exclude_cols is not None:
        df.drop(columns=exclude_cols, inplace=True)
    if rename_cols is not None:
        df.rename(columns=rename_cols, inplace=True)
    return df


def read_csv_data(folder_path, column_tranformation=None, progress=True):
    """Read simulation data in CSV format.

    Parameters
    ----------
    folder_path : str
        The parent folder containing the CSV files. The function automatically
        searches for all CSV files within, recursively.
    column_tranformation : dict, optional
        A dictionary mapping old column names to new column names. If a column
        name is mapped to None, that column will be removed. By default None.
    progress : bool, optional
        Whether to display a progress bar while reading files. By default True.

    Returns
    -------
    dict of pd.DataFrame
        Dictionary mapping indices to DataFrames of the found simulation data.

    """
    data = {}

    if column_tranformation is None:
        columns_removing, columns_renaming = None, None
    else:
        columns_renaming = {oldname: newname for oldname, newname
                            in column_tranformation.items()
                            if newname is not None}
        columns_removing = [oldname for oldname, newname
                            in column_tranformation.items()
                            if newname is None]

    csv_paths = _discover_csv_files(folder_path)
    for csv_path in tqdm(csv_paths, desc="Reading data", disable=not progress):
        csv_filename = os.path.basename(csv_path)
        index = int(os.path.splitext(csv_filename)[0])
        filedata = _read_csv_file(csv_path,
                                  columns_removing=columns_removing,
                                  columns_renaming=columns_renaming)

        if index in data:
            raise ValueError(
                f"Duplicate index {index} found in file {csv_path}.")
        data[index] = filedata

    return data


def _discover_csv_files(folderpath):
    return glob.glob(os.path.join(folderpath, "**", "*.csv"), recursive=True)


def _read_csv_file(csv_path, columns_removing=None, columns_renaming=None):
    df = pd.read_csv(csv_path)
    if columns_removing is not None:
        df.drop(columns=columns_removing, inplace=True)
    if columns_renaming is not None:
        df.rename(columns=columns_renaming, inplace=True)
    return df


def save_figure(filepath):
    """Save the current figure to a file."""
    if filepath is None:
        return
    plt.savefig(filepath, dpi=300, bbox_inches="tight", pad_inches=0.02)


def make_onehot(df, categorical_cols):
    cat_cols_set = set(categorical_cols)

    assert len(categorical_cols) == len(cat_cols_set)  # avoid duplicates
    assert len(cat_cols_set - set(df.columns)) == 0    # the cols are there

    new_df = pd.DataFrame()
    for col in df.columns:
        if col in categorical_cols:
            values = df[col]
            unique_values = np.unique(values)
            for unique_value in unique_values:
                new_df[f"{col}={unique_value}"] = np.int_(df[col] == unique_value)
        else:
            new_df[col] = df[col]

    return new_df
