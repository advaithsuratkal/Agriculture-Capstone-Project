import pandas as pd
import numpy as np
import math
import random


def load_and_preprocess_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"[SUCCESS] Dataset '{file_path}' loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return None

    print("\n--- DATASET OVERVIEW ---")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print("Missing Values Check:")
    print(df.isnull().sum())

    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    print(f"\nMissing values removed: {missing_before - df.isnull().sum().sum()}")

    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Duplicate rows removed: {duplicates_before}")

    print(f"Rows after cleaning: {df.shape[0]}")

    return df


def detect_outliers_iqr(df):
    numeric_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    print("\n--- OUTLIER CAPPING VIA IQR METHOD ---")
    for col in numeric_cols:
        q1 = np.percentile(df[col], 25)
        q3 = np.percentile(df[col], 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        df[col] = np.clip(df[col], lower_bound, upper_bound)
    print("[INFO] Numerical outliers capped successfully.")
    return df


def print_nutrient_summary(df):
    print("\n=== INDIVIDUAL NUTRIENT STATISTICAL SUMMARY ===")
    print(df[['Nitrogen', 'Phosphorous', 'Potassium']].describe().round(2))


def get_random_sample_summary(df, sample_size=1000):
    random.seed()
    actual_size = min(sample_size, len(df))
    random_indices = random.sample(range(0, len(df)), actual_size)

    print(f"\n--- RANDOM SAMPLE PREVIEW (Sample Size: {actual_size}) ---")
    sample_data = df.iloc[random_indices][['Crop Type', 'Soil Type', 'Nitrogen', 'Fertilizer Name']]
    print(sample_data.describe(include='all'))
