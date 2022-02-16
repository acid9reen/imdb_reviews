"""
Module for full dataset preprocessing
Construct all reviews in one csv file -> extract dataset features
"""


import csv
import os
import re
from typing import NoReturn


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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
