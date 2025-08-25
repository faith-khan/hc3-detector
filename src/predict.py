from pathlib import Path
import argparse
from joblib import load
import pandas as pd
import numpy as np

def load_model(model_path: Path):
    """Load the saved pipeline from joblib file."""
    return load(model_path)

def predict_text(model, text: str):
    """Predict label + probability for a given input string."""
    df = pd.DataFrame({"text" : [text]})
    proba = model.predict_proba(df)[0]
    classes = model.classes_
    pred = model.predict(df)[0]

    # class with probability
    probs = {cls: float(p) for cls, p in zip(classes, proba)}
    return pred, probs

def explain_text(model, text: str, top_k: int = 15):
    """Show top positive/negative contributions for this text."""
    df = pd.DataFrame({"text": [text]})

    feats = model.named_steps["features"]
    X = feats.transform(df)

    tfidf = feats.named_transformers_["tfidf"]
    tf_names = tfidf.get_feature_names_out()
    style_names = np.array(["avg_sentence_length", "type_token_ratio", "pronoun_rate"])
    names = np.concatenate([tf_names, style_names])

    clf = model.named_steps["clf"]
    coef = clf.coef_[0]

    X = X.tocoo()
    contribs = []
    for i, v in zip(X.col, X.data):
        contribs.append((names[i], float(v * coef[i]), float(v), float(coef[i])))
    contribs.sort(key=lambda x: x[1], reverse=True) 

    print("\nTop features pushing toward HUMAN:")
    for f, score, val, w in contribs[:top_k]:
        print(f"  {f:30s}  contrib={score: .4f}  value={val: .4f}  weight={w: .4f}")

    print("\nTop features pushing toward AI:")
    for f, score, val, w in sorted(contribs, key=lambda x: x[1])[:top_k]:
        print(f"  {f:30s}  contrib={score: .4f}  value={val: .4f}  weight={w: .4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", type=str, help="Input text to classify")
    ap.add_argument("--model", type=Path, default=Path("models/hc3_lr.joblib"),
                    help="Path to trained model")
    ap.add_argument("--explain", action="store_true", help="Show per-feature contributions")
    args = ap.parse_args()

    model = load_model(args.model)
    print(f"Loaded model from: {args.model}")
    print(f"Classes: {model.classes_}")
    pred, probs = predict_text(model, args.text)

    print(f"\nPredicted label: {pred}")
    for cls, p in probs.items():
        print(f"  {cls}: {p:.4f}")
    
    if args.explain:
        explain_text(model, args.text, top_k=15)

if __name__ == "__main__":
    main()

