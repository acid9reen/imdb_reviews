"""
Module for full dataset preprocessing
Construct all reviews in one csv file -> extract dataset features
"""


import csv
import os
import re
from typing import NoReturn

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def get_rating_from_filename(filename: str) -> int:
    """
    Get rating from filename between '_' and '.'

    >>> get_rating_from_filename("123_42.txt")
    42

    >>> get_rating_from_filename("abc_420.efgh")
    420
    """
    return int(filename[filename.rfind('_') + 1: filename.rfind(".")])


def construct_dataset(raw_data_path: str, out_path: str) -> NoReturn:
    """
    Walk recursively through all folders of raw_data_path and
    search for text files with reviews.
    Parse each file for review and rating.
    Finally, writes review and rating to .csv file.
    """

    with open(out_path, 'w', encoding="utf8", newline='') as out:
        header = ["review", "rating"]

        writer = csv.writer(out)
        writer.writerow(header)

        for (subdir, dirs, files) in os.walk(raw_data_path):
            for file in files:
                # Filter inappropriate files
                if not re.match(r"\d+_\d{1,2}\.txt", file):
                    continue

                with open(os.path.join(subdir, file), 'r', encoding="utf8") as f:
                    review = f.read()
                    rating = get_rating_from_filename(file)
                    writer.writerow([review, rating])


def save_to_csv(
    data: np.ndarray, labels: np.ndarray, out_folder_path: str, filename: str
) -> NoReturn:
    """
    Create dataframe from data and labels with columns 'review' and 'rating',
    save as .csv file with filename name in out_folder_path directory.
    """

    df = pd.DataFrame({"review": data.ravel(), "rating": labels})

    with open(os.path.join(out_folder_path, filename), 'w', encoding="utf8") as out:
        df.to_csv(out)


def split_train_test_data(data_file_path: str, out_folder_path: str) -> NoReturn:
    """
    Split input .csv data file into stratified train.csv and test.csv.
    """

    df = pd.read_csv(data_file_path)
    reviews = df["review"].to_numpy(dtype=str).reshape(-1, 1)
    ratings = df["rating"].to_numpy(dtype=np.int8)

    reviews_train, reviews_test, ratings_train, ratings_test = train_test_split(
        reviews, ratings, test_size=0.3, random_state=42, stratify=ratings
    )

    save_to_csv(reviews_train, ratings_train, out_folder_path, "train.csv")
    save_to_csv(reviews_test, ratings_test, out_folder_path, "test.csv")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
