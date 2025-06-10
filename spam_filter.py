#!/usr/bin/env python3
"""
Assignment B Designing a Spam Filter
Autor: <dein Name>

Ausführung (Beispiel):
    # 1) Daten-Archive liegen in  data/raw/
    # 2) Virtuelle Umgebung aktivieren (optional)
    # 3) python spam_filter.py

Optionen (siehe --help):
    --raw-dir     Verzeichnis mit *.tar.bz2
    --mail-dir    Ziel­ordner für entpackte Mails
    --model-path  Dateiname für das gespeicherte Modell
"""
from __future__ import annotations

import argparse
import html
import pathlib
import re
import sys
import tarfile
import email

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


def extract_archives(raw_dir: pathlib.Path, target_dir: pathlib.Path) -> None:
    """Entpackt alle .tar.bz2-Archive aus raw_dir in target_dir (rekursiv)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for archive in raw_dir.glob("*.tar.bz2"):
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(target_dir)
        print(f"✓ entpackt {archive.name}")


def parse_email(fp: pathlib.Path) -> str:
    """Liest eine E-Mail-Datei ein und gibt den reinen Textkörper (plain/text) zurück."""
    with open(fp, "rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)

    parts: list[str] = []
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            charset = part.get_content_charset() or "utf-8"
            try:
                parts.append(part.get_payload(decode=True).decode(charset, errors="ignore"))
            except LookupError:  # exotischer Zeichensatz
                parts.append(part.get_payload(decode=True).decode("utf-8", errors="ignore"))
    return "\n".join(parts)


def clean(text: str) -> str:
    """Einfache Vorverarbeitung: Header streichen, kleinschreiben, URLs & Zahlen maskieren, …"""
    text = text.split("\n\n", 1)[-1]            # Header-Bereich entfernen
    text = html.unescape(text.lower())
    text = URL_RE.sub(" URL ", text)
    text = NUMBER_RE.sub(" NUMBER ", text)
    text = re.sub(r"[^\w\s]", " ", text)        # Satzzeichen raus
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_dataframe(mail_dir: pathlib.Path) -> pd.DataFrame:
    """Durchläuft mail_dir rekursiv, liest alle Mails und baut einen DataFrame."""
    records: list[dict[str, str]] = []
    for path in mail_dir.rglob("*"):
        if path.is_file():
            label = "spam" if "spam" in path.parts[-2] else "ham"
            try:
                raw_txt = parse_email(path)
            except Exception as exc:             # defekte Mail?
                print(f"! überspringe {path.name}: {exc}", file=sys.stderr)
                continue
            records.append({"path": str(path), "label": label, "text": raw_txt})
    return pd.DataFrame(records)


# ─────────────────────────────────────────── main ───────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trainiert einen Spam-Classifier auf dem SpamAssassin-Korpus."
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Ordner mit *.tar.bz2-Archiven")
    parser.add_argument("--mail-dir", default="data/mail", help="Zielordner für entpackte Mails")
    parser.add_argument("--model-path", default="spam_classifier.joblib", help="Ausgabedatei Modell")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test­daten-Anteil (0-1)")
    args = parser.parse_args()

    raw_dir  = pathlib.Path(args.raw_dir)
    mail_dir = pathlib.Path(args.mail_dir)

    print("1/5  Archive entpacken …")
    extract_archives(raw_dir, mail_dir)

    print("2/5  DataFrame bauen …")
    df = build_dataframe(mail_dir)
    df["clean"] = df["text"].map(clean)
    print("\nKlassenverteilung:\n", df["label"].value_counts(), "\n")

    print("3/5  Train/Test-Split …")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=args.test_size,
        stratify=df["label"],
        random_state=42
    )

    print("4/5  Training + Hyperparameter-Suche …")
    pipe = Pipeline([
        ("vect", CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2))),
        ("tfidf", TfidfTransformer()),
        ("clf", LinearSVC())
    ])

    param_grid = {
        "vect__min_df":      [3, 5],
        "vect__ngram_range": [(1, 1), (1, 2)],
        "tfidf__use_idf":    [True, False],
        "clf__C":            [0.5, 1.0, 2.0],
    }

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    print("\nBeste Parameter:", search.best_params_, "\n")

    print("5/5  Evaluation auf Hold-out …")
    y_pred = search.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    print("Konfusionsmatrix:\n", confusion_matrix(y_test, y_pred, labels=["ham", "spam"]), "\n")

    print(f"Modell wird nach {args.model_path} gespeichert …")
    joblib.dump(search.best_estimator_, args.model_path)
    print("✓ fertig")


if __name__ == "__main__":
    main()
