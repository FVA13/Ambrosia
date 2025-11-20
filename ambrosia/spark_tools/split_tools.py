#  Copyright 2022 MTS (Mobile Telesystems)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Dict, Iterable, List, Optional, Union

import ambrosia.spark_tools.stratification as strat_pkg
from ambrosia import types
from ambrosia.tools import split_tools
from ambrosia.tools.import_tools import spark_installed

if spark_installed():
    import pyspark.sql.functions as spark_funcs
    from pyspark.sql import Window

HASH_COLUMN_NAME: str = "__hashed_ambrosia_column"
GROUPS_COLUMN: str = "group"
ROW_NUMBER: str = "__row_number"
EMPTY_VALUE: int = 0


def unite_spark_tables(*dataframes: types.SparkDataFrame) -> types.SparkDataFrame:
    """
    Union all spark dataframes.
    """
    amount_of_dataframes = len(dataframes)
    if not amount_of_dataframes:
        return None
    result = dataframes[0]
    for j in range(1, amount_of_dataframes):
        result = result.union(dataframes[j])
    return result


def add_hash_column(
    dataframe: types.SparkDataFrame,
    id_column: types.ColumnNameType,
    hash_function: str = "sha256",
    salt: Optional[str] = None,
) -> types.SparkDataFrame:
    """
    Returns new dataframe with column hashing id_column.

    Parameters
    ----------
    hash_function: str, default ``sha_256``
        Name of hash function
    """
    salt = "" if salt is None else salt
    column_with_id: types.SparkColumn = spark_funcs.concat(
        spark_funcs.col(id_column).cast("string"), spark_funcs.lit(salt)
    )
    if hash_function == "sha256":
        return dataframe.withColumn(HASH_COLUMN_NAME, spark_funcs.sha2(column_with_id, 256))
    elif hash_function == "sha512":
        return dataframe.withColumn(HASH_COLUMN_NAME, spark_funcs.sha2(column_with_id, 512))
    elif hash_function == "sha1":
        return dataframe.withColumn(HASH_COLUMN_NAME, spark_funcs.sha1(column_with_id))
    else:
        raise ValueError("Incorrect hash function name")


def get_hash_split(
    dataframe: types.SparkDataFrame,
    id_column: types.ColumnNameType,
    groups_size: int,
    labels: Iterable[str],
    groups_number: int = 2,
    # later group_b_indices: types.SparkDataFrame = None,
    hash_function: str = "sha256",
    salt: Optional[str] = None,
) -> types.SparkDataFrame:
    """
    Hash split.
    """
    hashed_dataframe = add_hash_column(dataframe, id_column, hash_function, salt)
    hashed_dataframe = hashed_dataframe.orderBy(HASH_COLUMN_NAME).limit(groups_number * groups_size)

    def udf_make_labels(row_number: int) -> str:
        label_ind = (row_number - 1) // groups_size
        return labels[label_ind]

    window = Window.orderBy(HASH_COLUMN_NAME).partitionBy(spark_funcs.lit(EMPTY_VALUE))
    result = hashed_dataframe.withColumn(ROW_NUMBER, spark_funcs.row_number().over(window)).withColumn(
        GROUPS_COLUMN, spark_funcs.udf(udf_make_labels)(spark_funcs.col(ROW_NUMBER))
    )
    result = result.drop(ROW_NUMBER, HASH_COLUMN_NAME)
    return result


def add_to_required_size(
    dataframe: types.SparkDataFrame,
    used_dataframe: types.SparkDataFrame,
    id_column: types.ColumnNameType,
    groups_size: Union[int, List[int]],
    current_sizes: List[int],
    labels: Iterable[str],
) -> types.SparkDataFrame:
    """
    Add elements for groups to required size.
    """
    not_used_ids: types.SparkDataFrame = dataframe.join(used_dataframe, on=id_column, how="leftanti")
    
    # Handle both single size and list of sizes
    if isinstance(groups_size, list):
        required_sizes: List[int] = [groups_size[j] - current_sizes[j] for j in range(len(groups_size))]
    else:
        required_sizes: List[int] = [groups_size - size_ for size_ in current_sizes]
    
    total_required: int = sum(required_sizes)
    if total_required <= 0:
        # Return empty dataframe with same schema
        return dataframe.limit(0).withColumn(GROUPS_COLUMN, spark_funcs.lit(None).cast("string"))
    
    not_used_ids = not_used_ids.limit(total_required)

    # Now it's linear search, probably there will be not so many groups
    def udf_make_labels_with_find(row_number: int):
        current_total: int = 0
        for j in range(len(required_sizes)):
            current_total += required_sizes[j]
            if not required_sizes[j]:
                continue
            if row_number <= current_total:
                return labels[j]

    return (
        not_used_ids.withColumn(
            ROW_NUMBER,
            spark_funcs.row_number().over(
                Window.orderBy(spark_funcs.lit(EMPTY_VALUE)).partitionBy(spark_funcs.lit(EMPTY_VALUE))
            ),
        )
        .withColumn(GROUPS_COLUMN, spark_funcs.udf(udf_make_labels_with_find)(ROW_NUMBER))
        .drop(ROW_NUMBER)
    )


def get_split(
    dataframe: types.SparkDataFrame,
    split_method: str,
    id_column: types.ColumnNameType,
    groups_size: Optional[int] = None,
    groups_sizes: Optional[List[int]] = None,
    groups_number: int = 2,
    # later group_b_indices: types.SparkDataFrame = None,
    strat_columns: Optional[List] = None,
    hash_function: str = "sha256",
    salt: Optional[str] = None,
    labels: Optional[Iterable[str]] = None,
) -> types.SparkDataFrame:
    """
    Get split.
    """
    total_size: int = dataframe.count()

    # Determine group sizes
    if groups_sizes is not None:
        groups_number = len(groups_sizes)
        if groups_size is not None:
            import warnings
            warnings.warn("groups_size is ignored when groups_sizes is provided")
    elif groups_size is None:
        raise ValueError("Either groups_size or groups_sizes must be provided")
    else:
        groups_sizes = [groups_size] * groups_number

    if sum(groups_sizes) > total_size:
        raise ValueError("Total sample size is more, than shape of table")

    if total_size > dataframe.dropDuplicates([id_column]).count():
        raise ValueError(f"Id column {id_column} contains duplicates, ids must be unique for split")

    if labels is None:
        labels = split_tools.make_labels_for_groups(groups_number)

    stratification = strat_pkg.Stratification()
    stratification.fit(dataframe, strat_columns)
    
    # Calculate strat sizes for each group
    strat_sizes_dict: Dict[int, Dict] = {}
    for group_idx in range(groups_number):
        strat_sizes_dict[group_idx] = stratification.get_group_sizes(groups_sizes[group_idx])
    
    tables_on_stratification = []

    # List of sizes for each group, better to make without calling to cluster
    current_sizes: List[int] = [0] * groups_number

    for strat_value, strat_table in stratification.groups():
        # For Spark, we'll use the minimum size per stratification group
        # and adjust later (similar to pandas approach)
        strat_group_sizes = [strat_sizes_dict[j][strat_value] for j in range(groups_number)]
        min_size = min(strat_group_sizes)
        
        if split_method == "hash":
            current_table = get_hash_split(
                strat_table,
                id_column,
                min_size,
                labels,
                groups_number,
                hash_function,
                salt,
            )
            # Note: Spark hash split creates equal-sized groups
            # We'll need to adjust sizes in add_to_required_size
        else:
            raise ValueError("Split method is not found")
        tables_on_stratification.append(current_table)
        for j in range(groups_number):
            current_sizes[j] += min_size

    # Unite dataframes from each stratification group
    used_ids: types.SparkDataFrame = unite_spark_tables(*tables_on_stratification)
    
    # Update add_to_required_size to handle different group sizes
    # Pass groups_sizes as a list to add_to_required_size
    additional_table: types.SparkDataFrame = add_to_required_size(
        dataframe, used_ids, id_column, groups_sizes, current_sizes, labels
    )
    return unite_spark_tables(used_ids, additional_table)
