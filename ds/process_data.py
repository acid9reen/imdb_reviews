"""
Module for full dataset preprocessing
Construct all reviews in one csv file -> clean data ->
-> extract dataset features
"""


import csv
import os
import re
from typing import NoReturn, Callable

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import spacy


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


def remove_punctuations(data: str) -> str:
    """Remove punctuation symbols from input string"""

    punct_tag = re.compile(r"[^\w\s]")
    data = punct_tag.sub("", data)

    return data


def remove_html(data: str) -> str:
    """Remove html tags from input string"""

    html_tag = re.compile(r"<.*?>")
    data = html_tag.sub("", data)

    return data


def remove_url(data: str) -> str:
    """Remove urls from input string"""

    url_clean = re.compile(r"https://\S+|www\.\S+")
    data = url_clean.sub(r"", data)

    return data


def remove_emoji(data: str) -> str:
    """Remove emojis from input string"""

    emoji_clean = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    data = emoji_clean.sub("", data)

    return data


def clean_str(data: str) -> str:
    """
    Apply all removing functions to input string.
    Remove emojis, html tags, urls and punctuation symbols.

    >>> clean_str("Hi <br> name </br>! Welcome to www.some.com 😊")
    'Hi  name  Welcome to  '

    >>> clean_str("Hi <input /> https://abc.com")
    'Hi  '
    """

    cleaning_funcs: list[Callable[[str], str]] = [
        remove_emoji,
        remove_html,
        remove_url,
        remove_punctuations,
    ]
    for func in cleaning_funcs:
        data = func(data)

    return data


def replace_abbr(data: str) -> str:
    """
    Replace some abbreviations and shrink spaces to one.

    >>> replace_abbr("It   doesn't    matter")
    'it does not matter'
    """

    abbr_dict = {
        "what's": "what is",
        "what're": "what are",
        "who's": "who is",
        "who're": "who are",
        "where's": "where is",
        "where're": "where are",
        "when's": "when is",
        "when're": "when are",
        "how's": "how is",
        "how're": "how are",

        "i'm": "i am",
        "we're": "we are",
        "you're": "you are",
        "they're": "they are",
        "it's": "it is",
        "he's": "he is",
        "she's": "she is",
        "that's": "that is",
        "there's": "there is",
        "there're": "there are",

        "i've": "i have",
        "we've": "we have",
        "you've": "you have",
        "they've": "they have",
        "who've": "who have",
        "would've": "would have",
        "not've": "not have",

        "i'll": "i will",
        "we'll": "we will",
        "you'll": "you will",
        "he'll": "he will",
        "she'll": "she will",
        "it'll": "it will",
        "they'll": "they will",

        "isn't": "is not",
        "wasn't": "was not",
        "aren't": "are not",
        "weren't": "were not",
        "can't": "can not",
        "couldn't": "could not",
        "don't": "do not",
        "didn't": "did not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "doesn't": "does not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "won't": "will not",
        r"\s+": " ",  # replace multi space with one single space
    }

    data = data.lower()

    for pattern, repl in abbr_dict.items():
        data = re.sub(pattern, repl, data)

    return data


def lemmatize(data: str) -> str:
    """
    Lemmatize input text.

    >>> lemmatize("I was reading the paper.")
    'I be read the paper .'
    """

    nlp = spacy.load("en_core_web_sm")
    lemmatizer = nlp.get_pipe("lemmatizer")
    doc = nlp(data)
    res = " ".join(token.lemma_ for token in doc)

    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
