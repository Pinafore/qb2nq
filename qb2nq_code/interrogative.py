"""
heuristics/interrogative.py
============================
Heuristic 2: Imperative to Interrogative

Transforms FTP-style imperative sentences into interrogative questions.
Example:
  "FTP, name this French composer." -> "Which composer is this?"
"""

import re
import spacy

nlp = spacy.load("en_core_web_sm")

from constants import PATTERNS, DEMONSTRATIVE_DETERMINERS
from utils.nlp_utils import (
    PARSE, get_wh_word, get_verb_position, get_head_of_verb,
    has_relative_clause, get_relative_clause_head,
    is_syntactically_valid, clean_question
)


# ============================================================
# PRECONDITION
# ============================================================

def precondition_interrogative(qbe):
    """Check if sentence contains FTP pattern."""
    normalized = re.sub(r'\s*,\s*', ', ', qbe)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return any(re.search(p, normalized, re.IGNORECASE) for p in PATTERNS)


# ============================================================
# HELPERS
# ============================================================

def build_interrogative_case3(tokens, head, answer_type, answer=None):
    """Build interrogative for the general case (no relative clause, no comma)."""
    if not answer_type or not answer_type.strip():
        answer_type = "thing"

    wh  = get_wh_word(answer_type, answer)
    doc = head.doc

    head_chunk = next(
        (chunk for chunk in doc.noun_chunks if head in chunk),
        None
    )
    chunk_start = head_chunk.start if head_chunk else head.i

    head_subtree = [t for t in tokens if t.i >= chunk_start]

    # Remove demonstrative determiner before head
    head_subtree = [
        t for t in head_subtree
        if not (t.i < head.i and t.text.lower() in DEMONSTRATIVE_DETERMINERS)
    ]

    continuation_full  = " ".join(t.text for t in head_subtree).strip().rstrip("?.")
    continuation_words = continuation_full.split()
    head_word          = continuation_words[0].lower() if continuation_words else ""
    answer_last_word   = answer_type.lower().split()[-1] if answer_type.split() else ""

    if not head_word or not answer_last_word:
        return f"{wh} {answer_type} {continuation_full}?"

    # RULE 1: Head word exactly matches answer_type last word
    if head_word == answer_last_word or (
        nlp(head_word)[0].has_vector and nlp(answer_last_word)[0].has_vector
        and nlp(head_word)[0].similarity(nlp(answer_last_word)[0]) > 0.5
    ):
        continuation_removed = " ".join(continuation_words[1:]).strip()
        if not continuation_removed:
            return f"{wh} {answer_type}?"
        # If continuation already starts with a verb, don't insert "is"
        cont_doc = nlp(continuation_removed)
        first_tok = cont_doc[0] if cont_doc else None
        starts_with_verb = first_tok is not None and first_tok.pos_ in ("VERB", "AUX")
        is_participle = first_tok is not None and first_tok.tag_ in ("VBN", "VBG")
        if starts_with_verb and not is_participle:
            return f"{wh} {answer_type} {continuation_removed}?"
        elif starts_with_verb and is_participle:
            return f"{wh} {answer_type} is {continuation_removed}?"
        candidate_with_is = f"{wh} {answer_type} is {continuation_removed}?"
        if is_syntactically_valid(candidate_with_is):
            return candidate_with_is
        else:
            continuation_doc = nlp(continuation_removed)
            has_verb = any(t.pos_ in ("VERB", "AUX") for t in continuation_doc)
            if has_verb:
                return f"{wh} {answer_type} {continuation_removed}?"
            else:
                if wh == "who":
                    return f"Who is the {answer_type} {continuation_removed}?"
                else:
                    return f"What is the {answer_type} {continuation_removed}?"

    # RULE 2: Head word not exact match -> keep full continuation
    continuation_removed = " ".join(continuation_words[1:]).strip()
    continuation_kept    = continuation_full
    candidate_removed    = f"{wh} {answer_type} is {continuation_removed}?"
    candidate_kept       = f"{wh} {answer_type} is {continuation_kept}?"

    if (head_word not in answer_type.lower() and
            answer_type.lower() not in head_word):
        return candidate_kept

    if is_syntactically_valid(candidate_removed):
        return candidate_removed
    else:
        return candidate_kept


# ============================================================
# HEURISTIC
# ============================================================

def heuristic_interrogative(qbe, answer_type="thing", answer=None):
    """Transform FTP imperative sentence into a wh-question."""
    normalized = re.sub(r'\s*,\s*', ', ', qbe)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    doc    = PARSE(normalized)
    tokens = list(doc)
    wh     = get_wh_word(answer_type, answer)
    final_question = None

    for pattern in PATTERNS:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if not match:
            continue

        pattern_start = match.start()
        preamble      = normalized[:pattern_start].strip().strip(",").strip()

        preamble_starts_with_wh = bool(
            re.match(r'^(which|who|what)\b', preamble, re.IGNORECASE)
        )
        if preamble_starts_with_wh:
            preamble = ""

        after_ftp        = normalized[match.end():].strip()
        after_ftp_doc    = PARSE(after_ftp)
        after_ftp_tokens = list(after_ftp_doc)

        print(f"  [INTERROG DEBUG] after_ftp: '{after_ftp}'")

        # --- TOKEN DEBUG (remove when done) ---
        for t in after_ftp_doc:
            print(f"  [TOKEN DEBUG] '{t.text}' | pos: {t.pos_} | tag: {t.tag_} | dep: {t.dep_}")

        verb_position = get_verb_position(after_ftp_doc)
        print(f"  [INTERROG DEBUG] verb_position: {verb_position}")

        if verb_position is None:
            # after_ftp has no verb — handle directly without falling back to original doc
            root_tok  = next((t for t in after_ftp_doc if t.dep_ == "ROOT"), None)
            first_vbn = next((t for t in after_ftp_doc if t.tag_ == "VBN"), None)
            if root_tok and first_vbn:
                rest = " ".join(
                    t.text for t in after_ftp_doc[first_vbn.i:]
                ).rstrip("?.")
                final_question = f"{wh} {answer_type} is {rest}?"
            elif root_tok:
                rest = " ".join(t.text for t in after_ftp_doc[root_tok.i:]).rstrip("?.")
                final_question = f"{wh} {answer_type} is {rest}?"
            else:
                after_ftp_clean = after_ftp.rstrip("?.").strip()
                final_question  = f"{wh.capitalize()} {after_ftp_clean}?"
            if preamble:
                final_question = f"{preamble}, {final_question}"
            break
        head = get_head_of_verb(after_ftp_doc, verb_position)
        # If ROOT is a noun and head is not the ROOT, use ROOT as head instead
        # If head doesn't match answer_type, try using the nsubj of ROOT instead
        root_tok = next((t for t in after_ftp_doc if t.dep_ == "ROOT"), None)
        if root_tok and root_tok.pos_ in ("VERB", "AUX"):
            nsubj = next((t for t in root_tok.children if t.dep_ in ("nsubj", "nsubjpass")), None)
            if nsubj and nsubj.pos_ in ("NOUN", "PROPN"):
                head = nsubj
        elif root_tok and root_tok.pos_ in ("NOUN", "PROPN") and head != root_tok:
            head = root_tok
        print(f"  [INTERROG DEBUG] head: '{head}'")

        if head is None:
            after_ftp_clean = after_ftp.rstrip("?.").strip()
            final_question  = f"{wh.capitalize()} {after_ftp_clean}?"
            if preamble:
                final_question = f"{preamble}, {final_question}"
            break

        has_relcl, rel_clause_token = has_relative_clause(head)
        print(f"  [INTERROG DEBUG] has_relcl: {has_relcl}, rel_clause_token: {rel_clause_token}")

        # Similarity-based match between head noun and answer_type
        head_doc      = nlp(head.text.lower())
        answer_wd_doc = nlp(answer_type.lower().split()[-1])
        similarity_match = (
            head_doc[0].has_vector and answer_wd_doc[0].has_vector
            and head_doc[0].similarity(answer_wd_doc[0]) > 0.5
        )
        head_matches_answer_type = (
            head.text.lower() == answer_type.lower().split()[-1]
            or answer_type.lower() in head.text.lower()
            or head.text.lower() in answer_type.lower()
            or similarity_match
        )
        print(f"  [INTERROG DEBUG] head_matches_answer_type: {head_matches_answer_type}")

        if has_relcl and head_matches_answer_type:
            print(f"  [RELCL DEBUG] head.i: {head.i}, rel_clause_token.i: {rel_clause_token.i}")
            print(f"  [RELCL DEBUG] tokens from head.i+1: {[t.text for t in after_ftp_tokens[head.i+1:]]}")
            # Take everything after the head noun, not just the relative clause
            left  = head.i + 1
            right = len(after_ftp_tokens) - 1
            continuation = " ".join(t.text for t in after_ftp_tokens[left:right]).strip().rstrip("?.")
            # Remove leading comma
            continuation = continuation.lstrip(", ").strip()

            cont_doc         = PARSE(continuation)
            first_token      = cont_doc[0] if len(cont_doc) > 0 else None
            starts_with_verb = (
                first_token is not None
                and first_token.pos_ in ("VERB", "AUX")
            )
            is_participle = first_token is not None and first_token.tag_ in ("VBN", "VBG")

            if starts_with_verb and not is_participle:
                core = f"{wh} {answer_type} {continuation}?"
            else:
                core = f"{wh} {answer_type} is {continuation}?"

        elif (head.i + 1 < len(after_ftp_tokens)
              and after_ftp_tokens[head.i + 1].text == ","):
            continuation = " ".join(
                t.text for t in after_ftp_tokens[head.i + 2:]
            ).strip().rstrip("?.")
            print(f"  [INTERROG DEBUG] CASE 2: continuation: '{continuation}'")
            print(f"  [INTERROG DEBUG] head.i: {head.i}, next_tok: '{after_ftp_tokens[head.i + 1].text if head.i + 1 < len(after_ftp_tokens) else None}'")
            core = f"{wh} {answer_type} is {continuation}?"

        else:
            print(f"  [INTERROG DEBUG] CASE 3")
            core = build_interrogative_case3(after_ftp_tokens, head, answer_type, answer)
            print(f"  [INTERROG DEBUG] CASE 3 core: '{core}'")

        print(f"  [INTERROG DEBUG] final core: '{core}'")
        final_question = core
        if preamble:
            final_question = f"{preamble}, {core}"
        break

    if final_question is None:
        return qbe

    final_question = final_question.strip().replace(" ?", "?").replace(" .", ".")
    return final_question[0].upper() + final_question[1:]
# ============================================================
# POSTCONDITION
# ============================================================

def postcondition_interrogative(qbe):
    return clean_question(qbe)
