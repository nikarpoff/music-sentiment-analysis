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


import pandas as pd

# Absolute path to the dataset.
DATASET_PATH = "F:\\dataset\\music\\autotagging_moodtheme.tsv"

def load_dataset(dataset_path=DATASET_PATH) -> pd.DataFrame:
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

def clean_last_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the last column of the DataFrame by removing the prefix 'mood/theme---'
    and replacing tab characters with commas.
    """
    if df is not None and not df.empty:
        last_column = df.columns[-1]
        df[last_column] = df[last_column].str.replace("mood/theme---", "", regex=False)
        df[last_column] = df[last_column].str.replace("\t", ",", regex=False)
    return df

if __name__ == "__main__":
    dataset = load_dataset()

    if dataset is not None:
        print(f"Dataset loaded successfully with {len(dataset)} rows and {len(dataset.columns)} columns.")
        print(dataset.head(n=10))

        clean_last_column(dataset)

        print("\nLast column cleaned.")
        print(dataset.head(n=10))

    else:
        print("Failed to load dataset.")
