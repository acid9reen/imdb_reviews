from setuptools import setup

# Metadata goes in setup.cfg. These are here for GitHub's dependency graph.
setup(
    name="ds",
    install_requires=[
        "nltk == 3.7",
        "numpy == 1.22.2",
        "pandas == 1.4.1",
        "scikit-learn == 1.0.2",
        "spacy == 3.2.2",
        "xgboost == 1.5.2",
    ],
)
