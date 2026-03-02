import pandas as pd

def load_pairs_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"image1", "image2", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain image_path1, image_path2, label")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1")
    return df