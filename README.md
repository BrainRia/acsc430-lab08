# Lab 08

This project is for lab 08. It uses the `fraud.csv` dataset and compares a couple of machine learning models using one feature at a time.

## What the program does

The main program is in `Lab08.py`.

It does these things:

- loads the fraud dataset
- prints some basic information about the dataset
- finds some strong relationships between columns that start with `V`
- uses `V1` as a sample feature
- picks another feature from the dataset to compare with `V1`
- trains Linear Regression and KNN on both features
- prints the scores
- saves a bar graph called `model_comparison.png`

## Files

- `Lab08.py` - main file that runs the program
- `lab08_utils.py` - helper functions for loading data, training models, and making the graph
- `fraud.csv` - dataset
- `requirements.txt` - libraries needed for the project

## Requirements

Install these libraries first:

```bash
pip install -r requirements.txt
```

## How to run

Run this in the terminal:

```bash
python Lab08.py
```

## Output

When the program runs, it prints:

- dataset size
- first 5 rows
- correlated `V` columns
- model scores for `V1`
- model scores for the chosen feature

It also creates:

- `model_comparison.png`

## Notes

This project is using single-feature prediction, so it is more for practice and comparison than for building a full fraud detection system.
