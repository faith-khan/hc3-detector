from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import json
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# local
from features import StylisticFeatures

def load_and_dedup(parquet_path: Path) -> pd.DataFrame:
    """Load HC3, keep needed cols, and remove exact duplicate texts"""
    df = pd.read_parquet(parquet_path)
    df = df[["text", "label"]].dropna(subset=["text", "label"]).reset_index(drop=True)

    # lightweight normalisation for duplicate detection
    _df = df.assign(
        _text_norm=lambda d: (
            d["text"]
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    )
    before = len(_df)
    df = (
        _df.drop_duplicates(subset=["_text_norm"])
           .drop(columns=["_text_norm"])
           .reset_index(drop=True)
    )
    after = len(df)

    print(f"[dedup] {before} → {after} rows (removed {before-after})")
    print("[labels]\n", df["label"].value_counts())
    return df


def build_pipeline(max_features: int = 5000) -> Pipeline:
    """TF-IDF (1–2 grams) + StylisticFeatures → Logistic Regression."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )

    features = ColumnTransformer(
        transformers=[
            ("tfidf", tfidf, "text"),
            ("style", StylisticFeatures(), "text"),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
    )

    return Pipeline([
        ("features", features),
        ("clf", clf),
    ])


def train_and_eval(df: pd.DataFrame, results_dir: Path, models_dir: Path, max_features: int):
    # split (after dedup)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.20, stratify=df["label"], random_state=42
    )

    # quick check: no exact overlap
    overlap = len(set(X_train).intersection(set(X_test)))
    print(f"[split] train={len(X_train)} test={len(X_test)} overlap={overlap}")

    pipe = build_pipeline(max_features=max_features)
    pipe.fit(X_train.to_frame(name="text"), y_train)

    y_pred = pipe.predict(X_test.to_frame(name="text"))
    proba = pipe.predict_proba(X_test.to_frame(name="text"))
    classes = list(pipe.named_steps["clf"].classes_)
    pos_idx = classes.index("ai") if "ai" in classes else 1
    roc = roc_auc_score((y_test == "ai").astype(int), proba[:, pos_idx])

    # metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    report["roc_auc_ai_positive"] = float(roc)

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # saving metrics
    (results_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"[save] metrics → {results_dir/'metrics.json'}")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix — TF-IDF + Stylistic")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    print(f"[save] confusion matrix → {results_dir/'confusion_matrix.png'}")

    # top features CSV (coefficients)
    tfidf_names = pipe.named_steps["features"].named_transformers_["tfidf"].get_feature_names_out()
    style_names = ["avg_sentence_length", "type_token_ratio", "pronoun_rate"]
    all_feats = np.concatenate([tfidf_names, style_names])
    coefs = pipe.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({"feature": all_feats, "coefficient": coefs})
    coef_df["abs_coeff"] = coef_df["coefficient"].abs()
    coef_df.sort_values("abs_coeff", ascending=False).to_csv(results_dir / "top_features.csv", index=False)
    print(f"[save] top features → {results_dir/'top_features.csv'}")

    # saving pipeline
    model_path = models_dir / "hc3_lr.joblib"
    dump(pipe, model_path)
    print(f"[save] pipeline → {model_path}")

    # brief console summary
    print("\n=== Summary ===")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"F1 (ai): {report['ai']['f1-score']:.3f} | F1 (human): {report['human']['f1-score']:.3f}")
    print(f"ROC-AUC (ai positive): {roc:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/hc3_full.parquet"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--max-features", type=int, default=5000)
    args = ap.parse_args()

    df = load_and_dedup(args.data)
    train_and_eval(df, args.results_dir, args.models_dir, args.max_features)

if __name__ == "__main__":
    main()