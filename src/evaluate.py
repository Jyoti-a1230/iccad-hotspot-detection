"""Test-set evaluation: builds leakage-aware clean test sets and reports
precision/recall/prediction-mean diagnostics for a trained model."""
import os
import pandas as pd
from data import label_from_filename
from pipeline import build_dataset
from splitting import get_base_pattern

def build_test_df(benchmark, root):
    d = os.path.join(root, benchmark, "test")
    rows = []
    for f in os.listdir(d):
        if f.endswith(".png"):
            rows.append({
                "filename": f,
                "group": get_base_pattern(f),
                "label": "HS" if label_from_filename(f) == 1 else "NHS",
            })
    return pd.DataFrame(rows)


def evaluate_on_test(model, benchmark, train_df, root):
    test_df = build_test_df(benchmark, root)
    train_patterns = set(train_df["group"])
    clean_test_df = test_df[~test_df["group"].isin(train_patterns)]
    print(f"Full test: {len(test_df)}, Clean test: {len(clean_test_df)}, Removed: {len(test_df) - len(clean_test_df)}")

    test_ds = build_dataset(test_df, benchmark=benchmark, split="test", root=root, shuffle=False)
    clean_test_ds = build_dataset(clean_test_df, benchmark=benchmark, split="test", root=root, shuffle=False)
    test_preds = model.predict(test_ds)
    clean_test_preds = model.predict(clean_test_ds)

    test_labels = (test_df["label"] == "HS").astype(int).values
    clean_test_labels = (clean_test_df["label"] == "HS").astype(int).values

    results = {}
    for name, preds, labels in [("full_test", test_preds, test_labels), ("clean_test", clean_test_preds, clean_test_labels)]:
        hs_p = preds.flatten()[labels == 1]
        nhs_p = preds.flatten()[labels == 0]
        preds_binary = (preds.flatten() >= 0.5).astype(int)
        tp = ((preds_binary == 1) & (labels == 1)).sum()
        fp = ((preds_binary == 1) & (labels == 0)).sum()
        fn = ((preds_binary == 0) & (labels == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"--- {name} ---")
        print(f"HS mean={hs_p.mean():.4f} (n={len(hs_p)})  NHS mean={nhs_p.mean():.4f} (n={len(nhs_p)})")
        print(f"precision={precision:.3f}  recall={recall:.3f}  (tp={tp}, fp={fp}, fn={fn})")
        results[name] = {"precision": precision, "recall": recall, "hs_mean": hs_p.mean(), "nhs_mean": nhs_p.mean()}
    return results
