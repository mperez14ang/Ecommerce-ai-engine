import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/amazon_products.csv")
PROCESSED_PATH = Path("data/processed/products_clean.parquet")

def load_raw(nrows = None):
    print("Loading raw data...")
    return pd.read_csv(RAW_PATH, nrows=nrows)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    #Remove duplicates
    df = df.drop_duplicates(subset=["title"])
    #Remove rows without price or title
    df = df.dropna(subset = ['price','title'])
    #Filter price outliers
    df = df[(df['price'] >= 0.5) & (df['price'] <= 5000)]
    #Clean text columns
    df['title'] = df['title'].str.strip().str[:200] # Limit title to 200 chars for consistent embeddings in the recommendation model
    df['category_id'] = df['category_id'].astype(str)
    return df

def save_processed(df: pd.DataFrame):
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"Saved: {len(df):,} products to {PROCESSED_PATH}")

if __name__ == "__main__":
    df_raw = load_raw(nrows=50000)
    df_clean = clean(df_raw)
    save_processed(df_clean)
    print("Done!")





