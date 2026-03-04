# src/load_data.py
import pandas as pd

def load_dataset():
    # Path relative to project root
    file_path = "data/raw/projectdataset.csv"

    df = pd.read_csv(file_path)

    print("Dataset Loaded Successfully!")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}\n")
    print("Columns in the dataset:")
    print(df.columns.tolist(), "\n")
    print("First 5 rows of the dataset:")
    print(df.head(), "\n")
    print("Missing Values Check:")
    print(df.isnull().sum(), "\n")

    return df