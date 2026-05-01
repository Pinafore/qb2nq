"""
pipeline.py
===========
Main pipeline: applies all three heuristics in sequence to a QB text.

Heuristic order:
  1. SplitConjunction   - splits conjoined verb phrases
  2. Interrogative      - converts FTP imperatives to questions
  3. NoWhWords          - converts demonstrative/pronoun sentences to wh-questions

Usage (single text):
    from pipeline import process_text
    result = process_text("This work was written by Tolstoy.", answer="War and Peace")

Usage (JSON file):
    from pipeline import process_json_file
    process_json_file("input.json", "output.json", "output.csv")
"""

import re
import json
import csv
import argparse

from utils.text_utils import clean_text, split_into_sentences, remove_ftp_artifacts
from utils.nlp_utils import extract_canonical_mention_from_text

from split_conjunction import (
    precondition_split_conjunction,
    heuristic_split_conjunction,
    postcondition_split_conjunction,
)
from interrogative import (
    precondition_interrogative,
    heuristic_interrogative,
    postcondition_interrogative,
)
from no_wh import (
    precondition_no_wh,
    heuristic_no_wh,
    postcondition_no_wh,
)


# ============================================================
# CORE: process a list of sentences through the pipeline
# ============================================================

def run_pipeline(sentences, answer, answer_type):
    """
    Run all three heuristics on a list of sentences.
    Returns a list of sentence_record dicts.
    """
    sentence_records = []

    for sentence in sentences:
        sentence_record = {
            "original":        sentence,
            "transformations": [],
            "final":           None
        }

        current_sentences = [{"text": sentence, "parent": None}]

        # ── Heuristic 1: Split Conjunction ──────────────────────
        split_results = []
        for s in current_sentences:
            if precondition_split_conjunction(s["text"]):
                split  = heuristic_split_conjunction(s["text"])
                split  = postcondition_split_conjunction(split)
                sentence_record["transformations"].append({
                    "heuristic": "SplitConjunction",
                    "input":     s["text"],
                    "output":    split
                })
                for clause in split:
                    split_results.append({
                        "text":      clause,
                        "parent":    s["text"],
                        "transform": "SplitConjunction"
                    })
            else:
                split_results.append(s)
        current_sentences = split_results

        # ── Heuristic 2: Imperative → Interrogative ─────────────
        interrogative_results = []
        for s in current_sentences:
            if precondition_interrogative(s["text"]):
                result = heuristic_interrogative(s["text"], answer_type, answer)
                result = postcondition_interrogative(result)
                print(result)
                sentence_record["transformations"].append({
                    "heuristic": "ImperativeToInterrogative",
                    "input":     s["text"],
                    "output":    result,
                    "parent":    s.get("parent", sentence)
                })
                interrogative_results.append({
                    "text":       result,
                    "parent":     s.get("parent", sentence),
                    "transform":  "ImperativeToInterrogative",
                    "skip_no_wh": True
                })
            else:
                interrogative_results.append(s)
        current_sentences = interrogative_results

        # ── Heuristic 3: No WH-Words ────────────────────────────
        wh_results = []
        for s in current_sentences:
            if s.get("skip_no_wh", False):
                wh_results.append(s)
                continue
            if precondition_no_wh(s["text"]):
                result = heuristic_no_wh(s["text"], answer, {answer: answer_type})
                result = postcondition_no_wh(result)
                sentence_record["transformations"].append({
                    "heuristic": "NoWhWords",
                    "input":     s["text"],
                    "output":    result,
                    "parent":    s.get("parent", sentence)
                })
                wh_results.append({
                    "text":      result,
                    "parent":    s.get("parent", sentence),
                    "transform": "NoWhWords"
                })
            else:
                wh_results.append(s)
        current_sentences = wh_results

        # ── FTP cleanup ─────────────────────────────────────────
        current_sentences = [
            {**s, "text": remove_ftp_artifacts(s["text"])}
            if re.search(r'\b(ftp|FTP|Ftp)\b|for\s+\d+\s+points?', s["text"], re.IGNORECASE)
            else s
            for s in current_sentences
        ]

        sentence_record["final"] = [s["text"] for s in current_sentences]
        sentence_records.append(sentence_record)

    return sentence_records


# ============================================================
# PUBLIC API: process a single text string
# ============================================================

def process_text(text, answer="", qanta_id=None):
    """
    Process a single QB text string through the full heuristic pipeline.

    Args:
        text:      The raw QB question text.
        answer:    The answer string (used for answer_type extraction).
        qanta_id:  Optional ID to pass through.

    Returns:
        A dict with keys: qanta_id, answer, answer_type, sentences.
    """
    text   = clean_text(text)
    answer = clean_text(answer) if answer else ""

    answer_type = extract_canonical_mention_from_text(text, answer) if answer else None
    sentences   = split_into_sentences(text)

    sentence_records = run_pipeline(sentences, answer, answer_type)

    return {
        "qanta_id":    qanta_id,
        "answer":      answer,
        "answer_type": answer_type,
        "sentences":   sentence_records
    }


# ============================================================
# PUBLIC API: process a JSON file
# ============================================================

def process_json_file(filepath, output_filepath, csv_filepath):
    """
    Process a JSON file of QB questions through the full pipeline.
    Writes results to a JSON file and a CSV file.

    JSON input format: list of {text, answer, qanta_id} dicts.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    all_output = []
    csv_rows   = []
    row_id     = 1

    for item in data:
        text     = clean_text(item["text"])
        answer   = clean_text(item["answer"])
        qanta_id = item.get("qanta_id", None)

        result = process_text(text, answer, qanta_id)
        all_output.append(result)

        for sentence_record in result["sentences"]:
            for final_sentence in sentence_record["final"]:
                csv_rows.append({
                    "id":       row_id,
                    "qanta_id": qanta_id,
                    "answer":   answer,
                    "original": sentence_record["original"],
                    "final":    final_sentence
                })
                row_id += 1

    # Write JSON
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(all_output, f, indent=4, ensure_ascii=False)
    print(f"JSON output written to {output_filepath}")

    # Write CSV
    with open(csv_filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "qanta_id", "answer", "original", "final"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV output written to {csv_filepath}")

    return all_output


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--output_file", required=True, help="Output file prefix (no extension)")

    args = parser.parse_args()

    output_json = f"{args.output_dir}/{args.output_file}.json"
    output_csv = f"{args.output_dir}/{args.output_file}.csv"

    process_json_file(
        args.input,
        output_json,
        output_csv
    )

    print("Processing complete.")
