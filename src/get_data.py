from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Direct URLs to the English subsets hosted on Hugging Face
BASE = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/"
FILES = ["all.jsonl"]

def load_hc3_df():

    data_files = [BASE + f for f in FILES]
    ds = load_dataset("json", data_files=data_files, split="train")  # treats each jsonl as one "train" split
    df = ds.to_pandas()

    # HC3 rows have lists of answers - this explodes them to 1 row per answer with a label.
    meta_cols = [c for c in ["category", "source", "question"] if c in df.columns]

    def explode(col, label):
        if col not in df.columns:
            return pd.DataFrame(columns=["text", "label"] + meta_cols)
        tmp = df[[col] + meta_cols].explode(col, ignore_index=True).rename(columns={col: "text"})
        tmp["label"] = label
        return tmp

    human = explode("human_answers", "human")
    ai = explode("chatgpt_answers", "ai")

    out = pd.concat([human, ai], ignore_index=True)
    out["text"] = out["text"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    out = out[out["text"].str.len() > 0].drop_duplicates().reset_index(drop=True)
    return out

if __name__ == "__main__":

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    df = load_hc3_df()
    print(f"Loaded {len(df)} rows. Humans={(df.label=='human').sum()}  AI={(df.label=='ai').sum()}")

    (data_dir / "hc3_full.parquet").write_bytes(b"") 
    df.to_parquet(data_dir / "hc3_full.parquet", index=False)

    n = min(500, len(df))
    df.sample(n=n, random_state=42).to_csv(data_dir / "hc3_sample_500.csv", index=False)
    print(f"Saved:\n- {data_dir/'hc3_full.parquet'} (local)\n- {data_dir/'hc3_sample_500.csv'} (small sample)")