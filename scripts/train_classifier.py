#!/usr/bin/env python3
"""
Train the Layer 1 intent classifier.
Model: TF-IDF vectorizer + Logistic Regression
Output: models/layer1_pipeline.joblib

Run: python3 scripts/train_classifier.py
Expected: >=90% cross-validation accuracy on 5 categories
Training time: <5 seconds on i5-7200U
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/training_intents.jsonl")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
PIPELINE_PATH = MODELS_DIR / "layer1_pipeline.joblib"


def load_data():
    lines = DATA_PATH.read_text().strip().split("\n")
    records = [json.loads(l) for l in lines if l.strip()]
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    return texts, labels


def train():
    print("Loading training data...")
    texts, labels = load_data()
    from collections import Counter
    counts = Counter(labels)
    print(f"  {len(texts)} examples, {len(counts)} classes:")
    for label, count in sorted(counts.items()):
        print(f"    {label}: {count}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs",
            class_weight="balanced",
        )),
    ])

    print("\nRunning 5-fold cross-validation...")
    t0 = time.monotonic()
    cv_scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="accuracy")
    cv_ms = (time.monotonic() - t0) * 1000
    print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f} ({cv_ms:.0f}ms)")

    if cv_scores.mean() < 0.90:
        print(f"\nWARNING: CV accuracy {cv_scores.mean():.1%} below 90% — add more training examples.")

    print("\nTraining on full dataset...")
    t0 = time.monotonic()
    pipeline.fit(texts, labels)
    print(f"  Training time: {(time.monotonic() - t0)*1000:.0f}ms")

    # Held-out eval
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    eval_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
                                  sublinear_tf=True, strip_accents="unicode",
                                  analyzer="word", token_pattern=r"\w{2,}")),
        ("clf", LogisticRegression(max_iter=1000, C=5.0,
                                   solver="lbfgs", class_weight="balanced")),
    ])
    eval_pipe.fit(X_tr, y_tr)
    y_pred = eval_pipe.predict(X_te)
    print("\nHeld-out classification report:")
    print(classification_report(y_te, y_pred, target_names=sorted(set(labels))))

    # Latency benchmark
    print("Inference latency (100 calls):")
    t0 = time.monotonic()
    for _ in range(100):
        pipeline.predict(["find all PDFs in my Downloads folder"])
        pipeline.predict_proba(["find all PDFs in my Downloads folder"])
    avg_ms = (time.monotonic() - t0) * 1000 / 100
    print(f"  Average: {avg_ms:.2f}ms per call")
    if avg_ms > 50:
        print(f"  WARNING: {avg_ms:.1f}ms exceeds 50ms target")
    else:
        print(f"  ✓ Latency target met")

    # Save
    joblib.dump(pipeline, PIPELINE_PATH)
    size_kb = PIPELINE_PATH.stat().st_size / 1024
    print(f"\nSaved: {PIPELINE_PATH} ({size_kb:.1f} KB)")

    # Verify load
    loaded = joblib.load(PIPELINE_PATH)
    result = loaded.predict(["find my PDFs"])[0]
    proba = loaded.predict_proba(["find my PDFs"])[0]
    conf = proba[list(loaded.classes_).index(result)]
    print(f"Load verification: '{result}' ({conf:.0%})")
    print("\n✓ Layer 1 classifier trained and saved")
    return cv_scores.mean()


if __name__ == "__main__":
    acc = train()
    if acc < 0.90:
        sys.exit(1)
