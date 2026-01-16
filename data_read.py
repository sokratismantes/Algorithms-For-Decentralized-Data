from pathlib import Path
import pandas as pd
import ast


"""
Expected fields (may vary by dataset):
id,title,adult,original_language,origin_country,release_date,
genre_names,production_company_names,budget,revenue,runtime,popularity,vote_average,vote_count
"""


def parse_list_field(val):
    """Metatrepw listes pou moiazoun me strings se Python list."""
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
        return [str(val)]


def parse_origin_country(val):
    """Origin country lista"""
    return parse_list_field(val)


def _read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {suffix}. Use .csv or .xlsx/.xls")


def load_and_preprocess(file_path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = _read_dataset(path)
    print(f"[INFO] Raw rows: {len(df)}")

    # required column 
    if "title" not in df.columns:
        raise ValueError(f"[ERROR] Column 'title' not found. Columns: {list(df.columns)[:40]}")

    # release_date
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    # list-like fields 
    if "origin_country" in df.columns:
        df["origin_country_parsed"] = df["origin_country"].apply(parse_origin_country)

    if "genre_names" in df.columns:
        df["genre_list"] = df["genre_names"].apply(parse_list_field)

    if "production_company_names" in df.columns:
        df["production_company_list"] = df["production_company_names"].apply(parse_list_field)

    # popularity
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
        if df["popularity"].isna().any():
            # fill NaNs with median to avoid breaking experiments
            df["popularity"] = df["popularity"].fillna(df["popularity"].median(skipna=True))
            print("[INFO] 'popularity' had NaNs -> filled with median.")
    else:
        # proxy options for datasets without popularity
        if "vote_average" in df.columns:
            df["popularity"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
            print("[WARN] 'popularity' missing -> using proxy from 'vote_average'.")
        elif "vote_count" in df.columns:
            df["popularity"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0.0)
            print("[WARN] 'popularity' missing -> using proxy from 'vote_count'.")
        else:
            df["popularity"] = 1.0
            print("[WARN] 'popularity' missing -> defaulting to 1.0 for all rows.")

    print(f"[INFO] Processed rows: {len(df)}")
    return df
