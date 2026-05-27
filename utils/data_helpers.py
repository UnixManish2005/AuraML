"""
utils/data_helpers.py
Synthetic dataset generators used across all page modules.
Also re-exports PLOTLY_LAYOUT and COLOR_SEQ for convenience.
"""

import numpy as np
import pandas as pd
from utils.styles import PLOTLY_LAYOUT, COLOR_SEQ   # single source of truth


# ── Sample dataset registry ───────────────────────────────────────────────────

SAMPLE_DATASETS = {
    "Iris (Classification)":       "iris",
    "Diabetes (Regression)":        "diabetes",
    "Breast Cancer (Binary)":       "breast_cancer",
    "Wine Quality (Multi-class)":   "wine",
    "House Prices (Synthetic)":     "house",
    "Customer Data (Clustering)":   "customer",
}


def load_sample(name: str) -> pd.DataFrame:
    key = SAMPLE_DATASETS.get(name, name)
    from sklearn import datasets as skd

    if key == "iris":
        d = skd.load_iris()
        df = pd.DataFrame(d.data, columns=d.feature_names)
        df["target"] = d.target
    elif key == "diabetes":
        d = skd.load_diabetes()
        df = pd.DataFrame(d.data, columns=d.feature_names)
        df["target"] = d.target
    elif key == "breast_cancer":
        d = skd.load_breast_cancer()
        df = pd.DataFrame(d.data, columns=d.feature_names)
        df["target"] = d.target
    elif key == "wine":
        d = skd.load_wine()
        df = pd.DataFrame(d.data, columns=d.feature_names)
        df["target"] = d.target
    elif key == "house":
        df = get_house_price_data(300)
    elif key == "customer":
        df = get_customer_data(300)
    else:
        df = pd.DataFrame()
    return df


# ── Synthetic generators ──────────────────────────────────────────────────────

def get_house_price_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    size   = rng.randint(500, 5000, n).astype(float)
    rooms  = rng.randint(1, 10, n).astype(float)
    age    = rng.randint(1, 50, n).astype(float)
    price  = 150 * size + 20_000 * rooms - 500 * age + 30_000 + rng.normal(0, 20_000, n)
    return pd.DataFrame({
        "Size_sqft":  size,
        "Rooms":      rooms,
        "Age_years":  age,
        "Price_USD":  np.maximum(price, 50_000).round(0),
    })


def get_customer_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    age     = rng.randint(18, 70, n)
    income  = rng.randint(15_000, 150_000, n)
    spend   = rng.randint(1, 100, n)
    return pd.DataFrame({
        "Age":            age,
        "Annual_Income":  income,
        "Spending_Score": spend,
    })


def get_loan_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng      = np.random.RandomState(seed)
    income   = rng.randint(20_000, 200_000, n).astype(float)
    credit   = rng.randint(300, 850, n).astype(float)
    debt     = rng.uniform(0.05, 0.9, n)
    score    = (income / 200_000 * 0.4 +
                (credit - 300) / 550 * 0.45 +
                (1 - debt) * 0.15 +
                rng.normal(0, 0.08, n))
    approved = (score > 0.5).astype(int)
    return pd.DataFrame({
        "Income":       income,
        "Credit_Score": credit,
        "Debt_Ratio":   debt.round(3),
        "Approved":     approved,
    })


def get_spam_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng        = np.random.RandomState(seed)
    word_count = rng.randint(50, 500, n).astype(float)
    links      = rng.randint(0, 20, n).astype(float)
    cap_ratio  = rng.uniform(0.0, 1.0, n)
    score      = (links / 20 * 0.5 + cap_ratio * 0.35 +
                  (word_count > 300).astype(float) * 0.15 +
                  rng.normal(0, 0.1, n))
    is_spam    = (score > 0.5).astype(int)
    return pd.DataFrame({
        "Word_Count":    word_count,
        "Links":         links,
        "Capital_Ratio": cap_ratio.round(3),
        "Is_Spam":       is_spam,
    })


def get_student_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng       = np.random.RandomState(seed)
    study     = rng.uniform(0, 10, n)
    sleep     = rng.uniform(4, 10, n)
    attend    = rng.uniform(50, 100, n)
    score     = (study * 6 + sleep * 2 + attend * 0.3 +
                 rng.normal(0, 5, n))
    score     = np.clip(score, 0, 100)
    passed    = (score >= 40).astype(int)
    return pd.DataFrame({
        "Study_Hours":  study.round(1),
        "Sleep_Hours":  sleep.round(1),
        "Attendance_%": attend.round(1),
        "Score":        score.round(1),
        "Passed":       passed,
    })


def get_classification_2d(n: int = 200, n_centers: int = 2, seed: int = 42):
    from sklearn.datasets import make_classification, make_blobs
    if n_centers == 2:
        X, y = make_classification(n_samples=n, n_features=2,
                                    n_informative=2, n_redundant=0,
                                    n_clusters_per_class=1, random_state=seed)
    else:
        X, y = make_blobs(n_samples=n, n_features=2, centers=n_centers,
                           cluster_std=1.2, random_state=seed)
    return X, y
