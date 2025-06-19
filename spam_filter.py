#!/usr/bin/env python3
"""
Assignment B Designing a Spam Filter
Luca Lindig

Usage for now:
    # 1) Data archives are located in data/raw/
    # 2) Activate the virtual environment (optional)
    # 3) python spam_filter.py

Options:
    --raw-dir     Directory containing *.tar.bz2 archives
    --mail-dir    Target folder for extracted emails
    --model-path  Filename for saving the model
"""
from __future__ import annotations

import argparse
import html
import pathlib
import re
import sys
import tarfile
import email
import email.policy 

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ───────────────────────────────────────── helpers ──────────────────────────────────────────

URL_RE    = re.compile(r"https?://\S+|www\.\S+")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


"""
This function extracts the contents of a .tar.bz2 archive into the specified directory.
It ensures that all files from the archive are unpacked and available for further processing.
"""
def extract_archive(archive_path, extract_to):
    target_dir.mkdir(parents=True, exist_ok=True)
    for archive in raw_dir.glob("*.tar.bz2"):
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(target_dir)
        print(f"✓ extracted {archive.name}")


"""
This function parses an email file, removes its headers, and returns the cleaned body text.
It is used to prepare raw email data for further preprocessing steps.
"""
def parse_email(fp: pathlib.Path) -> str:
    with open(fp, "rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)
    parts: list[str] = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            charset = part.get_content_charset() or "utf-8"
            try:
                parts.append(part.get_payload(decode=True).decode(charset, errors="ignore"))
            except LookupError:  # unknown charset
                parts.append(part.get_payload(decode=True).decode("utf-8", errors="ignore"))
    return "\n".join(parts)


"""
This function cleans the email text by converting it to lowercase, removing special characters,
and replacing URLs and numbers with placeholder tokens. It standardizes the text for model input.
"""
def clean(text: str) -> str:
    text = text.split("\n\n", 1)[-1]            # remove header section
    text = html.unescape(text.lower())
    text = URL_RE.sub(" URL ", text)
    text = NUMBER_RE.sub(" NUMBER ", text)
    text = re.sub(r"[^\w\s]", " ", text)        # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


"""
This function loads all emails from a directory, parses and cleans them, and returns a list of processed emails.
It automates the data loading and preprocessing pipeline for the dataset.
"""
def build_dataframe(mail_dir: pathlib.Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for path in mail_dir.rglob("*"):
        if path.is_file():
            label = "spam" if "spam" in path.parts[-2] else "ham"
            try:
                raw_txt = parse_email(path)
            except Exception as exc:             # skip broken email
                print(f"! skipping {path.name}: {exc}", file=sys.stderr)
                continue
            records.append({"path": str(path), "label": label, "text": raw_txt})
    return pd.DataFrame(records)


"""
This function builds and returns a machine learning pipeline consisting of a vectorizer,
TF-IDF transformer, and a Linear SVC classifier. It encapsulates the model architecture.
"""
def build_pipeline():
    pipe = Pipeline([
        ("vect", CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2))),
        ("tfidf", TfidfTransformer()),
        ("clf", LinearSVC(dual=True))
    ])
    return pipe


"""
This function performs hyperparameter tuning using GridSearchCV on the provided pipeline and data.
It returns the best estimator found during the search.
"""
def tune_hyperparameters(pipeline, X_train, y_train):
    # Reduced grid for testing speed
    param_grid = {
        "vect__min_df": [5],
        "vect__ngram_range": [(1, 1)],
        "tfidf__use_idf": [True],
        "clf__C": [1.0],
    }
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


"""
This function evaluates the trained model on test data, printing classification reports and confusion matrices.
It provides insights into the model's performance.
"""
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=["ham", "spam"]), "\n")


# ─────────────────────────────────────────── main ───────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trains a spam classifier using the SpamAssassin dataset."
    )
    parser.add_argument("--raw-dir", default="data", help="Directory containing *.tar.bz2 archives")
    parser.add_argument("--mail-dir", default="data/mail", help="Target folder for extracted emails")
    parser.add_argument("--model-path", default="spam_classifier.joblib", help="Output file for the model")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data for testing (0-1)")
    args = parser.parse_args()

    # Determine raw_dir with fallback
    raw_dir  = pathlib.Path(__file__).parent / args.raw_dir
    if not raw_dir.exists():
        possible_raw = pathlib.Path(__file__).parent.parent / args.raw_dir
        if possible_raw.exists():
            raw_dir = possible_raw
        else:
            possible_raw = pathlib.Path.cwd() / args.raw_dir
            if possible_raw.exists():
                raw_dir = possible_raw

    # Determine mail_dir with fallback
    mail_dir = pathlib.Path(__file__).parent / args.mail_dir
    if not mail_dir.exists():
        possible_mail = pathlib.Path(__file__).parent.parent / args.mail_dir
        if possible_mail.exists():
            mail_dir = possible_mail
        else:
            possible_mail = pathlib.Path.cwd() / args.mail_dir
            if possible_mail.exists():
                mail_dir = possible_mail

    if not raw_dir.exists():
        print(f"! raw_dir {raw_dir} not found. Please check that the directory {args.raw_dir} exists.", file=sys.stderr)
        sys.exit(1)
    if not mail_dir.exists():
        print(f"! mail_dir {mail_dir} not found. Please check that the directory {args.mail_dir} exists.", file=sys.stderr)
        sys.exit(1)

    print("1/5 Extracting archives …")
    extract_archives(raw_dir, mail_dir)

    print("2/5 Building DataFrame …")
    df = build_dataframe(mail_dir)
    if df.empty or "text" not in df.columns:
        print("! No email data found. Please check that the archive contains emails.", file=sys.stderr)
        sys.exit(1)
    df["clean"] = df["text"].map(clean)
    print("\nClass distribution:\n", df["label"].value_counts(), "\n")

    print("3/5 Train/Test Split …")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=args.test_size,
        stratify=df["label"],
        random_state=42
    )

    print("4/5 Training + Hyperparameter Search …")
    pipeline = build_pipeline()
    best_model = tune_hyperparameters(pipeline, X_train, y_train)
    print("\nBest Parameters:", best_model.get_params(), "\n")

    print("5/5 Evaluation on hold-out set …")
    evaluate_model(best_model, X_test, y_test)

    print(f"Saving model to {args.model_path} …")
    joblib.dump(best_model, args.model_path)
    print("✓ done")


if __name__ == "__main__":
    main()
