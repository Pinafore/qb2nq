"""
utils/text_utils.py
===================
Text cleaning, sentence splitting, and FTP artifact removal.
"""

import re
import spacy

nlp = spacy.load("en_core_web_sm")

from constants import PATTERNS


def clean_text(text):
    """
    Clean text encoding issues before processing.
    Handles common corrupted characters in QB questions.
    """
    replacements = {
        "Õ":        "'",
        "Ô":        "'",
        "Ò":        '"',
        "Ó":        '"',
        "Ñ":        "-",
        "Ð":        "-",
        "×":        "x",
        "Ø":        "o",
        "\u00e9":   "e",
        "\u00e0":   "a",
        "\u00e8":   "e",
        "\u00ea":   "e",
        "\u00eb":   "e",
        "\u00ef":   "i",
        "\u00f4":   "o",
        "\u00fb":   "u",
        "\u00fc":   "u",
        "\u00e7":   "c",
        "\u2019":   "'",
        "\u2018":   "'",
        "\u201c":   '"',
        "\u201d":   '"',
        "\u2013":   "-",
        "\u2014":   "-",
        "\u00a0":   " ",
    }
    for corrupted, correct in replacements.items():
        text = text.replace(corrupted, correct)
    return text


def remove_ftp_artifacts(text):
    """
    Remove any remaining FTP/points patterns from transformed sentences.
    """
    # Case 1: FTP in middle of sentence
    text = re.sub(r',?\s*(ftp|FTP|Ftp)\s*,?\s*', ' ', text)

    # Case 2: Remove "For X points," patterns
    text = re.sub(
        r'(For|for)\s+(ten|10|20|5|15)\s+(Points|points|POINTS)\s*,?\s*',
        '', text
    )

    # Case 3: Remove orphaned "name/identify/give" at start
    text = re.sub(r'^\s*(name|identify|give)\s+', '', text, flags=re.IGNORECASE)

    # Clean up spacing and punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip(',').strip()

    if text:
        text = text[0].upper() + text[1:]

    if text and not text.endswith("?"):
        text = text.rstrip(".") + "?"

    return text


def split_into_sentences(text):
    """
    Split a QB question text into individual sentences,
    correctly handling FTP markers and abbreviations.
    """
    text = clean_text(text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = text.replace("\u00a0", " ").strip()

    # STEP 1: Insert explicit split BEFORE FTP/points patterns
    text = re.sub(r'([.!?])\s+(FTP|ftp|Ftp)', r'\1 <<SPLIT>> \2', text)
    text = re.sub(
        r'([.!?])\s+(For\s+(?:ten|10|20|5|15)\s+(?:Points|points|POINTS))',
        r'\1 <<SPLIT>> \2',
        text
    )

    # STEP 2: Protect known multi-word abbreviations
    ABBREVIATIONS = [
        "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "Rev", "Gen","S","v",
        "Sgt", "Cpl", "Col", "Lt", "Cmdr", "Capt", "St", "Ave",
        "Blvd", "Dept", "Est", "Govt", "Univ", "Assoc", "Bros",
        "U.S", "U.K", "U.N", "E.U", "vs", "etc", "al", "approx",
        "corp", "inc", "ltd", "jan", "feb", "mar", "apr", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ]
    for abbr in ABBREVIATIONS:
        text = re.sub(
            rf'\b({re.escape(abbr)})\.',
            r'\1<<DOT>>',
            text, flags=re.IGNORECASE
        )

    # STEP 3: Split on <<SPLIT>> markers first
    chunks = text.split("<<SPLIT>>")

    result = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # STEP 4: Further split on normal sentence boundaries
        # STEP 4: Further split on normal sentence boundaries
        # Do NOT split if the word before the period is a single letter (initial/abbreviation)
        sub_sentences = re.split(
            r'(?:(?<=[.!?])|(?<=[.!?]["\')\]]))\s+(?=[A-Z])(?<!(?<!\w)\w[.!?]\s)',
            chunk
        )
        for s in sub_sentences:
            s = s.replace("<<DOT>>", ".").strip()
            if s:
                result.append(s)

    print("split sentences: ", result)
    return result if result else [text]
