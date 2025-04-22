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
import json
import pandas as pd
from dotenv import load_dotenv
from collections import Counter
from argparse import ArgumentParser
from exceptions.exceptions import InvalidConfigException


def cli_arguments_preprocess() -> tuple:
    """
    Read, parse and preprocess command line arguments:
        - path to source dataset. Required
        - path to mel spectrograms. Should be related to dataset path. By default: "melspecs/"
        - number of moods. By default: all moods (0)

    Sorce dataset file should be named "autotagging_moodtheme.tsv"
    """
    parser = ArgumentParser(description="Preprocessing autotagging moods dataset script. Cleans moods, aggregates them and saves preprocessed .tsv file")

    parser.add_argument("--path", required=True,
                      help="Path to source dataset")
    
    parser.add_argument("--mels", required=False,
                      help="Path to mel spectrograms. Should be related to the dataset path. By default: 'melspecs/'")
    
    parser.add_argument("--moods", required=False,
                      choices=["2", "4", "8", "all"],
                      help="Number of aggregated moods. By default: all source moods")

    args = parser.parse_args()

    if args.path[-1] != "/":
        args.path += "/"

    if args.mels is None:
        args.mels = "melspecs/"

    if not args.moods or args.moods == "all":
        args.moods = 0

    return os.path.abspath(args.path), args.mels, args.moods

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

def add_mel_spec_path(df: pd.DataFrame, melspecs_rel_path: str) -> pd.DataFrame:
    """
    Adds the mel spectrograms path to the DataFrame.
    """
    # Add the mel spectrograms path to the DataFrame and reorder columns
    df.insert(df.columns.get_loc("path") + 1, "mel_spec_path", 
              df["path"].apply(lambda x: os.path.join(melspecs_rel_path, x.replace(".mp3", ".npy"))))
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

def merge_moods(dataset: pd.DataFrame, basic_moods: list, ban_moods: list) -> tuple:
    """
    Merges moods tags in several basic moods.
    
    Each unpopular mood merges with the basic mood that most often stands in pairs with it. 
    Removes ban moods from the tags. If the tag can't be merged with any basic mood, row will be removed from the dataset.
    """
    mood_pairs = {}  # Set of basic moods sets of them pairs with pairs counts.
    non_basic_moods = set(dataset["tags"].explode().unique())  # Set of non basic moods.

    for basic_mood in basic_moods:
        if basic_mood not in non_basic_moods:
            raise InvalidConfigException(f"Basic mood {basic_mood} from config not found in dataset. Please check the config file.")
        
        if basic_mood in ban_moods:
            raise InvalidConfigException(f"Basic mood {basic_mood} found in ban list {ban_moods}. Please check the config file.")

        mood_pairs[basic_mood] = {}
        non_basic_moods.remove(basic_mood)

    # Counting cycle to determine mood pairs.
    for i, row in dataset.iterrows():
        tags = row["tags"]
        current_basic_mood = None

        # Try to find basic mood in tags.
        for basic_mood in basic_moods:
            if basic_mood in tags:
                current_basic_mood = basic_mood
                break
        
        # If no basic mood in tags, skip this row.
        if current_basic_mood is None:
            continue
        
        # There is basic mood in tags, so we can build mood pairs.
        for tag in tags:
            if tag != current_basic_mood:
                # If tag is not in basic moods set, we can add it to the set with count 1. Else -> increase count by 1.
                if tag in mood_pairs[current_basic_mood]:
                    mood_pairs[current_basic_mood][tag] += 1
                else:
                    mood_pairs[current_basic_mood][tag] = 1
        
        # Finally, replace current tag with basic mood.
        dataset.loc[i, "tags"] = [current_basic_mood]
    
    # Count the most popular neighbour mood for each non basic mood.
    moods_conformity = {}

    for mood in non_basic_moods:
        # Skip banned moods.
        if mood in ban_moods:
            moods_conformity[mood] = None
            continue

        pairs_counts = {}

        for basic_mood in mood_pairs.keys():
            # Remember pairs counts if tag was paired with basic mood.
            if mood in mood_pairs[basic_mood]:
                pairs_counts[basic_mood] = mood_pairs[basic_mood][mood]

        # Choose the most popular neighbour basic mood from the pairs counts.
        selected_mood = max(pairs_counts, key=pairs_counts.get) if pairs_counts else None
        
        # Remember conformity.
        if selected_mood is not None:
            moods_conformity[mood] = selected_mood
        else:
            moods_conformity[mood] = None

    # Merging cycle to merge moods.
    for i, row in dataset.iterrows():
        tags = row["tags"]

        # In this case we have already merged moods in the dataset.
        if tags[0] in basic_moods:
            continue

        # Now we can start merge other moods.
        tags_to_remove = []
        for tag in tags:
            if moods_conformity[tag] is not None:
                tags[tags.index(tag)] = moods_conformity[tag]
            else:
                tags_to_remove.append(tag)
        
        for tag in tags_to_remove:
            tags.remove(tag)

        # Now select the most popular mood in the tags list.
        if len(tags) == 0 or len(tags) == 1:
            continue

        counter = Counter(tags)
        selected_mood = max(counter, key=counter.get)
        dataset.loc[i, "tags"] = [selected_mood]

    # Remove rows with empty tags and transform the tags-array with only one element into a string 
    # (it is guaranteed that tags contain only single-element arrays).
    dataset = dataset[dataset["tags"].apply(lambda x: len(x) > 0)].assign(tags=lambda df: df["tags"].str[0])

    return dataset, mood_pairs, moods_conformity

def main():
    # Load dataset, environment and read cli arguments.
    dataset_path, melspecs_rel_path, moods_merge_mode = cli_arguments_preprocess()
    load_dotenv()

    # Get required info from environment variables.
    dataset_source_name = os.getenv("SOURCE_DATASET_NAME")
    config_path = os.getenv("CONFIG_PATH")
    outputs_path = os.getenv("OUTPUTS_PATH")

    if not os.path.exists(outputs_path):
        os.mkdir(outputs_path)

    # Load dataset.
    dataset = load_dataset(os.path.join(dataset_path, dataset_source_name))
    
    if dataset is not None and not dataset.empty:
        # Clean dataset.
        print(f"Dataset loaded successfully with {len(dataset)} rows and {len(dataset.columns)} columns.")
        print(dataset.head(n=3), "\n")

        cleaned_dataset = clean_dataset_pipeline(dataset)

        print("Dataset cleaned successfully:")
        print(cleaned_dataset.head(n=3), "\n")
        cleaned_dataset.info()

        print("Add mel spectrograms path to the dataset.")
        cleaned_dataset = add_mel_spec_path(cleaned_dataset, melspecs_rel_path)
        print(cleaned_dataset.head(n=3), "\n")

        # Get target tags distribution (for next merging).
        tags_distribution_save_path = os.path.join(outputs_path, "tags_distribution.csv")
        tags_distribution = cleaned_dataset["tags"].explode().value_counts()
        tags_distribution.to_csv(tags_distribution_save_path, index=True)
        print(f"Tags distribution saved successfully. Path: {tags_distribution_save_path}\n")

        # If user defined moods mode as "all" then script should just save the cleaned data.
        if moods_merge_mode == 0:
            save_path = os.path.join(dataset_path, "dataset_all_moods.tsv")
            cleaned_dataset.to_csv(save_path, sep="\t", index=False)
            print("Cleaned dataset saved successfully. Path: ", save_path)
            return

        # Load config to merge moods.
        with open(os.path.join(config_path, "moods.json")) as file:
            config = json.load(file)

            save_path = os.path.join(dataset_path, f"dataset_{moods_merge_mode}_moods.tsv")
            base_moods = config[f"mode_{moods_merge_mode}"]
            ban_moods = config[f"ban_list_{moods_merge_mode}"]

        # Merge moods in the dataset.
        final_dataset, moods_pairs, moods_conformity = merge_moods(cleaned_dataset, base_moods, ban_moods)

        # Save moods pairs to json file.
        moods_pairs_save_path = os.path.join(outputs_path, f"moods_pairs_{moods_merge_mode}.json")
        moods_conformity_save_path = os.path.join(outputs_path, f"moods_conformity_{moods_merge_mode}.json")

        with open(moods_pairs_save_path, "w", encoding="utf-8") as file:
            json.dump(moods_pairs, file, indent=4)
        print("Moods pairs saved successfully to the output directory.")
        
        with open(moods_conformity_save_path, "w", encoding="utf-8") as file:
            json.dump(moods_conformity, file, indent=4)
        print("Moods conformity saved successfully to the output directory.\n")

        print(f"Final dataset has {len(final_dataset)} rows and {len(final_dataset.columns)} columns.\n")
        print(final_dataset.head(n=10), "\n")
        final_dataset.info()

        tags_distribution = final_dataset["tags"].value_counts()
        print(f"\nTags distribution after merging moods:\n{tags_distribution}\n")

        # Save final dataset.
        final_dataset.to_csv(save_path, sep="\t", index=False)
        print("Dataset saved successfully. Path: ", save_path)
    else:
        print("Failed to load dataset.")


if __name__ == "__main__":
    main()
