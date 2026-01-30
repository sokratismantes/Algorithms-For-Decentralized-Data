import ast
import pandas as pd


def _parse_list_field(val):
    """
    Converts list-like strings to Python lists.
    Handles NaN and already-list values safely.
    """
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except Exception:
        # fallback: keep as single string
        return [str(val)]


def load_and_preprocess_csv(
    file_path: str,
    max_rows: int = 50_000,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Load CSV, keep up to max_rows rows (sample if larger), and do light preprocessing.

    Expected columns:
    id, title, adult, original_language, origin_country, release_date,
    genre_names, production_company_names, budget, revenue, runtime,
    popularity, vote_average, vote_count
    """
    df = pd.read_csv(file_path)
    print(f"[INFO] Raw rows: {len(df)}")

    # Keep up to max_rows (random sample -> more representative than head())
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    # Ensure title exists and is string
    if "title" not in df.columns:
        raise ValueError("CSV must contain a 'title' column.")
    df["title"] = df["title"].astype(str)

    # Safe datetime parsing
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    # Parse list-like fields into list columns (optional, but useful)
    if "origin_country" in df.columns:
        df["origin_country_parsed"] = df["origin_country"].apply(_parse_list_field)

    if "genre_names" in df.columns:
        df["genre_list"] = df["genre_names"].apply(_parse_list_field)

    if "production_company_names" in df.columns:
        df["production_company_list"] = df["production_company_names"].apply(_parse_list_field)

    print(f"[INFO] Using rows: {len(df)}")
    return df
