"""Data preprocessing, cleaning, leakage auditing, and TF-IDF feature extraction.

Reads raw parquet files, cleans text duplicates, enforces zero cross-split leakage,
and fits a 10,000-feature TF-IDF vectorizer strictly on training data.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import joblib
import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import ROOT, LABELS, write_json

# Filenames of raw Apache Parquet files for each split
FILES = {
    split: f"{split}-00000-of-00001.parquet"
    for split in ["train", "validation", "test"]
}

# Expected sample counts from the original Hugging Face dataset for verification
EXPECTED_COUNTS = {"train": 16000, "validation": 2000, "test": 2000}

# TF-IDF hyperparameter specifications: 10,000 unigrams+bigrams, sublinear TF scaling
TFIDF = {
    "max_features": 10000,
    "ngram_range": [1, 2],
    "stop_words": None,
    "sublinear_tf": True,
    "lowercase": True,
}


# Normalize text into canonical form by case-folding and collapsing whitespace
def canonical(text):
    # casefold() handles full Unicode case mapping; split() collapses multiple whitespace characters
    return " ".join(text.casefold().split())


# Ingest raw Parquet files, validate schema integrity, and perform initial quality audit
def read_splits(raw, expected_counts=None):
    records, audit = {}, {}
    
    # Process each data split (train, validation, test)
    for split, filename in FILES.items():
        # Read Apache Parquet table from disk
        table = pq.read_table(raw / filename)
        
        # Verify required columns exist
        if not {"text", "label"}.issubset(table.column_names):
            raise ValueError(f"{filename} must contain text and label")
        if expected_counts and table.num_rows != expected_counts[split]:
            raise ValueError(f"Unexpected {split} size: {table.num_rows}")
            
        # Verify Hugging Face schema metadata if embedded
        schema_metadata = table.schema.metadata or {}
        if b"huggingface" in schema_metadata:
            features = json.loads(schema_metadata[b"huggingface"])["info"]["features"]
            names = features.get("label", {}).get("names")
            if names is not None and names != LABELS:
                raise ValueError(f"Unexpected label names/order in {filename}")
                
        # Initialize counters for data quality tracking
        rows, counts = [], Counter(
            {
                k: 0
                for k in [
                    "original",
                    "missing_or_empty",
                    "invalid_label",
                    "invalid_text_type",
                ]
            }
        )
        groups = defaultdict(set)
        
        # Iterate over records and validate individual fields
        for index, row in enumerate(table.select(["text", "label"]).to_pylist()):
            counts["original"] += 1
            text, label = row["text"], row["label"]
            
            # Check for null, empty, or whitespace-only strings
            if (
                text is None
                or label is None
                or (isinstance(text, str) and not text.strip())
            ):
                counts["missing_or_empty"] += 1
                continue
                
            # Verify data type of text field
            if not isinstance(text, str):
                counts["invalid_text_type"] += 1
                continue
                
            # Verify label is an integer bounded within 0..5 (boolean excluded)
            if (
                isinstance(label, bool)
                or not isinstance(label, int)
                or not 0 <= label < len(LABELS)
            ):
                counts["invalid_label"] += 1
                continue
                
            rows.append({"text": text, "label": label, "source_row": index})
            # Track set of labels associated with this canonical text
            groups[canonical(text)].add(label)
            
        counts["valid"] = len(rows)
        # Count duplicate sentences with identical text
        counts["duplicate_text_extra_rows"] = len(rows) - len(groups)
        # Count ambiguous texts assigned more than one label within the split
        counts["conflicting_text_groups"] = sum(
            len(values) > 1 for values in groups.values()
        )
        
        # Record audit statistics and schema
        audit[split] = {
            **dict(counts),
            "original_classes": {
                label: sum(row["label"] == i for row in rows)
                for i, label in enumerate(LABELS)
            },
            "columns": table.column_names,
            "schema": str(table.schema),
        }
        records[split] = rows
        
    return records, audit


# Clean datasets and eliminate cross-split data leakage using a strict priority policy
def clean_splits(records, audit):
    # Group all texts per split to identify conflicting labels
    groups = {}
    for split, rows in records.items():
        groups[split] = defaultdict(set)
        for row in rows:
            groups[split][canonical(row["text"])].add(row["label"])
            
    # Audit cross-split raw text overlaps before cleaning
    overlap = {}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        shared = set(groups[a]) & set(groups[b])
        overlap[f"{a}__{b}"] = {
            "text_groups": len(shared),
            "different_label_sets": sum(
                groups[a][key] != groups[b][key] for key in shared
            ),
        }
        
    cleaned, retained_keys = {}, set()
    # Predetermined integrity policy, unrelated to model performance. Never resplit.
    # Processing order: test first, then validation, then train to protect test benchmarks.
    for split in ["test", "validation", "train"]:
        counts = Counter(
            {
                k: 0
                for k in [
                    "conflicting_label_rows_removed",
                    "cross_split_rows_removed",
                    "within_split_rows_removed",
                ]
            }
        )
        local, kept = set(), []
        for row in records[split]:
            key = canonical(row["text"])
            
            # 1. Remove ambiguous texts that have multiple conflicting emotion labels
            if len(groups[split][key]) > 1:
                counts["conflicting_label_rows_removed"] += 1
                continue
                
            # 2. Prevent data leakage: drop if this text was already claimed by an earlier split
            if key in retained_keys:
                counts["cross_split_rows_removed"] += 1
                continue
                
            # 3. Deduplicate multiple identical sentences within the same split
            if key in local:
                counts["within_split_rows_removed"] += 1
                continue
                
            local.add(key)
            kept.append(row)
            
        retained_keys.update(local)
        cleaned[split] = kept
        audit[split].update(dict(counts))
        audit[split]["retained"] = len(kept)
        audit[split]["retained_classes"] = {
            label: sum(row["label"] == i for row in kept)
            for i, label in enumerate(LABELS)
        }
        
    # Strictly assert zero text overlap across any pair of splits
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not {canonical(x["text"]) for x in cleaned[a]} & {
            canonical(x["text"]) for x in cleaned[b]
        }
        
    return cleaned, audit, overlap


# End-to-end data preparation, cleaning, TF-IDF vectorization, and serialization
def prepare(raw, processed, reports, verify_source_counts=True):
    # Guard against replacing existing processed study data without authorization
    if processed.exists() and any(processed.iterdir()):
        raise FileExistsError(
            "Processed folder is nonempty; refusing to replace a study"
        )
    if (reports / "data_quality.json").exists():
        raise FileExistsError("Data quality report already exists")
        
    # Step 1: Read raw parquet files and validate schema
    records, audit = read_splits(raw, EXPECTED_COUNTS if verify_source_counts else None)
    
    # Step 2: Clean duplicates, resolve conflicts, and enforce zero cross-split leakage
    rows, audit, overlaps = clean_splits(records, audit)
    if any(not values for values in rows.values()):
        raise ValueError("Cleaning produced an empty split")
        
    # Verify that all 6 emotion classes are present in the training set
    counts = np.bincount([row["label"] for row in rows["train"]], minlength=len(LABELS))
    if np.any(counts == 0):
        raise ValueError("Training set must contain all six classes")
        
    # Step 3: Configure TF-IDF Vectorizer
    # - max_features=10000: caps vocabulary size
    # - ngram_range=(1, 2): extracts unigrams and bigrams
    # - sublinear_tf=True: applies 1 + log(tf) scaling to temper high-frequency words
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words=None,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )
    
    # Learn vocabulary/IDF from Training only; other splits are transform-only.
    # CRITICAL: Fitting only on training data prevents data leakage from validation/test sets
    vectorizer.fit([row["text"] for row in rows["train"]])
    
    processed.mkdir(parents=True, exist_ok=True)
    
    # Step 4: Transform each split and persist to disk in compressed sparse formats
    for split, values in rows.items():
        # Transform text samples into sparse CSR matrix
        features = vectorizer.transform([row["text"] for row in values])
        
        # Save sparse matrix as SciPy .npz
        save_npz(processed / f"{split}.npz", features)
        # Save integer targets as NumPy 1D array of int64
        np.save(
            processed / f"{split}_labels.npy",
            np.array([row["label"] for row in values], dtype=np.int64),
        )
        # Save raw cleaned text and metadata as JSON
        write_json(processed / f"{split}_rows.json", values)
        
        # Compute sentence length statistics (word count percentiles)
        lengths = np.array([len(row["text"].split()) for row in values])
        audit[split]["word_length"] = dict(
            zip(
                ["min", "median", "p95", "max"],
                map(float, np.percentile(lengths, [0, 50, 95, 100])),
            )
        )
        # Track feature sparsity and coverage
        audit[split]["zero_feature_rows"] = int((features.getnnz(axis=1) == 0).sum())
        audit[split]["mean_nonzero_features"] = float(features.getnnz(axis=1).mean())
        
    # Persist the fitted scikit-learn vectorizer for inference and deployment
    joblib.dump(vectorizer, processed / "vectorizer.joblib")
    write_json(processed / "labels.json", LABELS)
    
    # Compile comprehensive metadata record
    metadata = {
        "configuration": "split",
        "labels": LABELS,
        "tfidf": TFIDF,
        "input_dim": len(vectorizer.vocabulary_),
        "class_weights": None,
        "majority_class": int(counts.argmax()),
        "cleaning_policy": "canonical casefold/whitespace; remove all within-split conflicting-label groups; keep first identical same-label text; retain test > validation > train across splits; no random resplit",
        "splits": audit,
        "raw_cross_split_overlap": overlaps,
        "fit_split": "train",
    }
    write_json(reports / "data_quality.json", metadata)
    
    # Generate exploratory data analysis (EDA) plots
    from .plots import eda

    eda(rows, audit, reports / "figures")
    print(
        "Prepared:", {split: len(values) for split, values in rows.items()}, flush=True
    )
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess and clean raw emotion parquet splits.")
    p.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    p.add_argument("--processed", type=Path, default=ROOT / "data/processed")
    p.add_argument(
        "--reports", type=Path, default=ROOT / "report/dataset_analyze/generated"
    )
    args = p.parse_args()
    prepare(args.raw, args.processed, args.reports)
