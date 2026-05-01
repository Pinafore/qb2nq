"""
heuristics/no_wh.py
====================
Heuristic 3: No WH-Words

Transforms sentences that lack wh-words (but have demonstratives or
pronoun subjects) into wh-questions.

Includes the sub-entity fix:
  WRONG:  "This declaration prompts her lover..." 
          -> "Which work prompts her lover..." (semantically wrong)
  FIXED:  "This declaration prompts her lover..."
          -> "In which work does a declaration prompt her lover..."
"""

import re
import spacy

nlp = spacy.load("en_core_web_sm")

from constants import DEMONSTRATIVE_DETERMINERS, POSSESSIVE_PRONOUNS
from utils.nlp_utils import (
    get_answer_type, get_wh_word, has_wh_word,
    clean_question, is_sub_entity
)
from interrogative import precondition_interrogative


# ============================================================
# PRECONDITION
# ============================================================

def precondition_no_wh(qbe):
    """
    Trigger NoWhWords only if:
    - No FTP/points pattern anywhere in sentence
    - No wh-words used as question words
    - Contains a demonstrative or pronoun/noun subject
    """
    if precondition_interrogative(qbe):
        return False

    if has_wh_word(qbe):
        return False

    doc = nlp(qbe)
    for chunk in doc.noun_chunks:
        if any(t.text.lower() in DEMONSTRATIVE_DETERMINERS for t in chunk):
            return True
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.pos_ == "PRON":
            return True
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.pos_ in ("NOUN", "PROPN"):
            return True
    return False


# ============================================================
# REPLACEMENT TARGET FINDER
# ============================================================

def find_best_replacement_target(doc, tokens, answer_type):
    """
    Find the best token/span in the sentence to replace with the wh-phrase.
    Returns (target_type, target) tuple.
    """
    SUBJECT_PRONOUNS = {"he", "she", "it", "they", "we", "i", "you"}
    used_tokens = set()

    def lock_span(span):
        for t in span:
            used_tokens.add(t.i)

    # Priority 0: demonstrative NP ("this/these X")
    for chunk in doc.noun_chunks:
        if chunk[0].text.lower() in ("this", "these"):
            lock_span(chunk)
            return ("demonstrative_chunk", chunk)

    # Priority 1: possessive demonstrative ("this work's")
    for token in tokens:
        if token.i in used_tokens:
            continue
        if token.dep_ == "poss" and token.head:
            if token.text.lower() in ("this", "these"):
                chunk = next((c for c in doc.noun_chunks if token.head in c), None)
                if chunk:
                    lock_span(chunk)
                    return ("demonstrative_chunk", chunk)

    # Priority 2: "it"/"its"
    for dep_group in [("pobj", "dobj"), ("nsubj", "nsubjpass", "attr")]:
        for token in tokens:
            if token.i in used_tokens:
                continue
            if token.text.lower() in ("it", "its","them","they") and token.dep_ in dep_group:
                used_tokens.add(token.i)
                if token.text.lower() == "its":
                    return ("possessive", token)
                return ("object_pronoun", token)

    # Priority 3: subject pronouns (he/she/they/etc.)
    for token in tokens:
        if token.i in used_tokens:
            continue
        if (token.dep_ in ("nsubj", "nsubjpass")
                and token.pos_ == "PRON"
                and token.text.lower() in SUBJECT_PRONOUNS
                and token.text.lower() != "it"):
            used_tokens.add(token.i)
            return ("pronoun_subject", token)

    # Priority 4: possessive pronouns
    for token in tokens:
        if token.i in used_tokens:
            continue
        if token.text.lower() in POSSESSIVE_PRONOUNS and token.dep_ == "poss":
            used_tokens.add(token.i)
            return ("possessive", token)

    # Priority 5: noun possessive
    for token in tokens:
        if token.i in used_tokens:
            continue
        if (token.dep_ == "poss"
                and token.pos_ in ("NOUN", "PROPN")
                and token.text.lower() != answer_type.lower()):
            used_tokens.add(token.i)
            return ("noun_possessive", token)

    # Priority 6: noun subject fallback
    for token in tokens:
        if token.i in used_tokens:
            continue
        if token.dep_ in ("nsubj", "nsubjpass") and token.pos_ in ("NOUN", "PROPN"):
            subj_chunk = next(
                (chunk for chunk in doc.noun_chunks if token in chunk), None
            )
            used_tokens.add(token.i)
            return ("noun_subject", subj_chunk if subj_chunk else token)
            """if token.text.lower() != answer_type.lower():
                return ("noun_subject_possessive", subj_chunk if subj_chunk else token)
            else:
                return ("noun_subject", subj_chunk if subj_chunk else token)"""

    return (None, None)


# ============================================================
# HEURISTIC
# ============================================================
def is_human_answer_type(answer_type):
    answer_doc = nlp(answer_type.split()[-1])  # use last word e.g. "composer" from "French composer"
    person_doc = nlp("person")
    if answer_doc[0].has_vector and person_doc[0].has_vector:
        return answer_doc[0].similarity(person_doc[0]) > 0.5
    return False

def heuristic_no_wh(qbe, answer, canonical_mentions):
    """Transform sentence without wh-words into a wh-question."""

    if has_wh_word(qbe):
        return qbe

    doc         = nlp(qbe)
    tokens      = list(doc)
    answer_type = get_answer_type(answer, canonical_mentions)
    wh          = get_wh_word(answer_type, answer)
    which_phrase = f"{wh} {answer_type}"

    target_type, target = find_best_replacement_target(doc, tokens, answer_type)

    # ----------------------------------------------------------
    # DEMONSTRATIVE CHUNK
    # ----------------------------------------------------------
    if target_type == "demonstrative_chunk":
        chunk      = target
        chunk_head = chunk.root  # e.g. "declaration", "play", "work"

        # FIX: Check if the demonstrative noun is a sub-entity of the answer type
        # e.g. "This declaration" inside a "work" -> use "In which work does..." framing
        if is_sub_entity(chunk_head, answer_type):
            modified = re.sub(r'\b(this|these)\b', 'the', qbe, flags=re.IGNORECASE).rstrip("?.")
            final_question = f"In {which_phrase} {modified}?"
        else:
            # Original behavior: direct replacement
            this_pos   = chunk.start
            next_tok   = tokens[this_pos + 1] if this_pos + 1 < len(tokens) else None
            after_next = tokens[this_pos + 2] if this_pos + 2 < len(tokens) else None

            if next_tok and after_next and after_next.text == "'s":
                possessed_noun = next_tok.text
                end_pos = this_pos + 3
                before  = " ".join(t.text for t in tokens[:this_pos])
                after   = " ".join(t.text for t in tokens[end_pos:])
                parts   = [p for p in [before, f"{wh} {possessed_noun}'s", after] if p.strip()]
                final_question = " ".join(parts)

            elif next_tok and next_tok.text in ('"', "'", "\u201c", "\u2018", "``"):
                closing_quotes = {'"': '"', "'": "'", "\u201c": "\u201d", "\u2018": "\u2019", "``": "''"}
                closing_quote  = closing_quotes.get(next_tok.text, '"')
                end_pos = this_pos + 2
                while end_pos < len(tokens) and tokens[end_pos].text not in (closing_quote, '"', "\u201d", "''"):
                    end_pos += 1
                end_pos += 1
                before = " ".join(t.text for t in tokens[:this_pos])
                after  = " ".join(t.text for t in tokens[end_pos:])
                parts  = [p for p in [before, which_phrase, after] if p.strip()]
                final_question = " ".join(parts)

            else:
                before = " ".join(t.text for t in tokens[:chunk.start])
                after  = " ".join(t.text for t in tokens[chunk.end:])
                parts  = [p for p in [before, which_phrase, after] if p.strip()]
                final_question = " ".join(parts)

    # ----------------------------------------------------------
    # PRONOUN SUBJECT
    # ----------------------------------------------------------
    elif target_type == "pronoun_subject":
        token = target
        final_question = " ".join(
            which_phrase if t.i == token.i else t.text
            for t in tokens
        )

    # ----------------------------------------------------------
    # POSSESSIVE (pronoun or "its")
    # ----------------------------------------------------------
    elif target_type == "possessive":
        token = target
        person_pronouns = {"his", "her", "their"}
        if token.text.lower() in person_pronouns and not is_human_answer_type(answer_type):
            rest = qbe.rstrip("?.")
            final_question = f"In {which_phrase} {rest}?"
        else:
            final_question = " ".join(
                f"{wh} {answer_type}'s" if t.i == token.i else t.text
                for t in tokens
            )

    # ----------------------------------------------------------
    # NOUN POSSESSIVE
    # ----------------------------------------------------------
    elif target_type == "noun_possessive":
        token = target
        final_question = " ".join(
            f"{which_phrase}'s" if t.i == token.i else t.text
            for t in tokens
        )

    # ----------------------------------------------------------
    # OBJECT PRONOUN
    # ----------------------------------------------------------
    elif target_type == "object_pronoun":
        token = target
        final_question = " ".join(
            which_phrase if t.i == token.i else t.text
            for t in tokens
        )

    # ----------------------------------------------------------
    # NOUN SUBJECT POSSESSIVE
    # ----------------------------------------------------------
    elif target_type == "noun_subject_possessive":
        if hasattr(target, "start"):
            before    = " ".join(t.text for t in tokens[:target.start])
            after     = " ".join(t.text for t in tokens[target.end:])
            subj_text = " ".join(t.text for t in tokens[target.start:target.end])
            parts     = [p for p in [before, f"{which_phrase}'s {subj_text}", after] if p.strip()]
            final_question = " ".join(parts)
        else:
            final_question = " ".join(
                f"{which_phrase}'s {target.text}" if t.i == target.i else t.text
                for t in tokens
            )

    # ----------------------------------------------------------
    # NOUN SUBJECT
    # ----------------------------------------------------------
    elif target_type == "noun_subject":
        # Check if subject noun is semantically different from answer_type
        subj_tok = target.root if hasattr(target, "root") else target
        subj_doc = nlp(subj_tok.text.lower())
        answer_wd_doc = nlp(answer_type.lower().split()[-1])
        same_type = (
            subj_doc[0].has_vector and answer_wd_doc[0].has_vector
            and subj_doc[0].similarity(answer_wd_doc[0]) > 0.5
        )
        if not same_type:
            # Subject is a different entity (e.g. "protagonist") — prepend instead
            rest = qbe.rstrip("?.")
            final_question = f"In {which_phrase} {rest}?"
        elif hasattr(target, "start"):
            before = " ".join(t.text for t in tokens[:target.start])
            after  = " ".join(t.text for t in tokens[target.end:])
            parts  = [p for p in [before, which_phrase, after] if p.strip()]
            final_question = " ".join(parts)
        else:
            final_question = " ".join(
                which_phrase if t.i == target.i else t.text
                for t in tokens
            )

    # ----------------------------------------------------------
    # FALLBACK
    # ----------------------------------------------------------
    else:
        rest = qbe.rstrip("?.")
        final_question = f"In {which_phrase} {rest}?"

    final_question = final_question.strip().replace(" ?", "?").replace(" .", ".")
    return final_question[0].upper() + final_question[1:]


# ============================================================
# POSTCONDITION
# ============================================================

def postcondition_no_wh(qbe):
    return clean_question(qbe)
