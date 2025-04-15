# Copyright [2024] [Nikita Karpov]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pandas as pd
from argparse import ArgumentParser

OUTPUTS_PATH = "./outputs"


def cli_arguments_preprocess() -> tuple:
    """
    Read, parse and preprocess command line arguments:
        - path to source dataset. Required
        - number of moods. By default: all moods (0)

    Sorce dataset file should be named "autotagging_moodtheme.tsv"
    """
    parser = ArgumentParser(description="Preprocessing autotagging moods dataset script. Cleans moods, aggregates them and saves preprocessed .tsv file")
    filename = "autotagging_moodtheme.tsv"

    parser.add_argument("--path", required=True,
                      help="Path to source dataset")
    
    parser.add_argument("--moods", required=False,
                      choices=["2", "4", "all"],
                      help="Number of aggregated moods. By default: all source moods")

    args = parser.parse_args()

    if args.path[-1] != "/":
        args.path += "/"
    
    args.path += filename

    if not args.moods or args.moods == "all":
        args.moods = 0

    return args.path, args.moods

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Reads the dataset from the specified path and returns it as a pandas DataFrame.
    Handles cases where the last column contains multiple values separated by tabs.
    """
    try:
        # Read the dataset without splitting the last column.
        with open(dataset_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        
        columns_number = len(lines[0].split("\t"))

        # Split lines by tabs; last column may contain multiple values.
        data = [line.strip().split("\t", maxsplit=columns_number - 1) for line in lines]
        
        # Create a DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])  # First row as header
        return df
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the last column of the DataFrame by removing the prefix 'mood/theme---'
    and splites multi tags by tabs
    """
    last_column = df.columns[-1]
    df[last_column] = df[last_column].str.replace("mood/theme---", "", regex=False)
    df[last_column] = df[last_column].str.split("\t")
    return df

def clean_columns_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refactors the column names to lower case
    """
    columns = df.columns
    refactored_columns = columns.str.lower()
    df.columns = refactored_columns
    return df

def clean_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the ID columns by removing the prifix "track", "artist", "album" and parses them to int
    Remember that this code cleanes first three columns: "track_id", "artist_id", "album_id" in such order
    """
    track_id_column = df.columns[0]
    artist_id_column = df.columns[1]
    album_id_column = df.columns[2]

    df[track_id_column] = df[track_id_column].str.replace("track_", "", regex=False)
    df[artist_id_column] = df[artist_id_column].str.replace("artist_", "", regex=False)
    df[album_id_column] = df[album_id_column].str.replace("album_", "", regex=False)

    df[track_id_column] = pd.to_numeric(df[track_id_column])
    df[artist_id_column] = pd.to_numeric(df[artist_id_column])
    df[album_id_column] = pd.to_numeric(df[album_id_column])

    return df

def clean_dataset_pipeline(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by applying all cleaning functions in order.
    """
    df = dataset.copy()  # Avoid modifying the original DataFrame
    df = clean_columns_name(df)
    df = clean_target(df)
    df = clean_id_columns(df)
    return df

def merge_moods(dataset: pd.DataFrame, merge_mode: int) -> pd.DataFrame:
    pass

def main(outputs_path):
    dataset_path, moods_merge_mode = cli_arguments_preprocess()

    if not os.path.exists(OUTPUTS_PATH):
        os.mkdir(OUTPUTS_PATH)

    dataset = load_dataset(dataset_path)
    
    if dataset is not None and not dataset.empty:
        print(f"Dataset loaded successfully with {len(dataset)} rows and {len(dataset.columns)} columns.")
        print(dataset.head(n=10), "\n")

        cleaned_dataset = clean_dataset_pipeline(dataset)

        print("Dataset cleaned successfully:")
        print(cleaned_dataset.head(n=10), "\n")
        cleaned_dataset.info()

        tags_distribution_save_path = os.path.join(OUTPUTS_PATH, "tags_distribution.csv")
        tags_distribution = cleaned_dataset["tags"].explode().value_counts()
        tags_distribution.to_csv(tags_distribution_save_path, index=True)
        print("Tags distribution saved successfully. Path: ", tags_distribution_save_path)

        if moods_merge_mode == 0:
            save_path = os.path.join(dataset_path, "dataset_all_moods.tsv")
            cleaned_dataset.to_csv(save_path, sep="\t", index=False)
            print("Dataset saved successfully. Path: ", save_path)
            return

        final_dataset = merge_moods(cleaned_dataset, moods_merge_mode)

        if moods_merge_mode == 2:
            save_path = os.path.join(dataset_path, "dataset_2_moods.tsv")
            
        final_dataset.to_csv(save_path, sep="\t", index=False)
        print("Dataset saved successfully. Path: ", save_path)
    else:
        print("Failed to load dataset.")


if __name__ == "__main__":
    main(outputs_path=OUTPUTS_PATH)

