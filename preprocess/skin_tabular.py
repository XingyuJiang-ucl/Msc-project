import pandas as pd
import json

def process_csv_to_json(input_csv: str, output_json: str):
    """
    Read table data from CSV and generate JSON in the specified format:
      - case_id: from column 'img_id'
      - cont_tab: includes only diameter_1, diameter_2, and age; fill missing values with -1; no normalization
      - cat_tab: treat all other columns (except lesion_id, img_id, and cont_tab) as categorical features
          • Use integer encoding: start numbering from 1; reserve 0 for missing or unknown values
    Save the result to the output_json file.
    """
    # Read CSV
    df = pd.read_csv(input_csv, dtype=str)
    # Convert numeric columns to float
    for col in ['diameter_1', 'diameter_2', 'age']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Define continuous and categorical columns
    cont_cols = ['diameter_1', 'diameter_2', 'age']
    skip_cols = ['lesion_id', 'img_id']
    cat_cols = [
        c for c in df.columns
        if c not in cont_cols + ['img_id'] + skip_cols
    ]

    # Fill missing values in continuous columns with -1
    df[cont_cols] = df[cont_cols].fillna(-1)

    # Create integer mappings for each categorical column (1..N), reserving 0 for missing
    cat_mappings = {}
    for col in cat_cols:
        uniques = sorted(df[col].dropna().unique().tolist())
        cat_mappings[col] = {val: idx + 1 for idx, val in enumerate(uniques)}

    # Build JSON records
    records = []
    for _, row in df.iterrows():
        rec = {
            "case_id": row["img_id"],
            "cont_tab": {},
            "cat_tab": {}
        }
        # Fill cont_tab
        for col in cont_cols:
            rec["cont_tab"][col] = (
                row[col] if not pd.isna(row[col]) else -1
            )
        # Fill cat_tab with integer encodings
        for col in cat_cols:
            v = row[col]
            code = cat_mappings[col].get(v, 0)  # Missing or unknown -> 0
            rec["cat_tab"][col] = code

        records.append(rec)

    # Write to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    process_csv_to_json(r"D:\Msc project\skin lesion\metadata.csv", "output.json")
