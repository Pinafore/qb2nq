"""
utils/nlp_utils.py
==================
Shared NLP utility functions used across heuristics:
  - Answer type extraction
  - WH-word selection
  - Syntactic helpers (subject, verb, head noun)
  - Relative clause detection
  - Syntactic validity checking
  - Question cleaning
"""

import re
import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

from constants import (
    WH_WORDS, TARGET_VERBS, DEMONSTRATIVE_DETERMINERS,
    POSSESSIVE_PRONOUNS, SUB_ENTITY_TYPES
)


# ============================================================
# PARSE
# ============================================================

def PARSE(text):
    return nlp(text)


# ============================================================
# ANSWER TYPE
# ============================================================

def get_answer_type(answer, canonical_mentions):
    if answer in canonical_mentions:
        return canonical_mentions[answer]
    doc = nlp(answer)
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "PER"):            return "person"
        elif ent.label_ in ("GPE", "LOC"):             return "place"
        elif ent.label_ in ("ORG"):                    return "organization"
        elif ent.label_ in ("DATE", "TIME"):           return "time"
        elif ent.label_ in ("PRODUCT", "WORK_OF_ART"): return "thing"
    for token in doc:
        if token.pos_ == "PROPN":
            return "person"
    return answer


def extract_canonical_mention_from_text(text, answer):
    """Extract canonical mention (answer type) directly from QB text."""
    doc = nlp(text)
    mention_counts = defaultdict(int)

    for chunk in doc.noun_chunks:
        if chunk[0].text.lower() in DEMONSTRATIVE_DETERMINERS:
            clean_tokens = []
            for t in chunk[1:]:
                if t.text in ("'s", "'") or t.dep_ in ("prep", "poss"):
                    break
                clean_tokens.append(t.text)
            if clean_tokens:
                mention = " ".join(clean_tokens)
                mention_counts[mention] += 1

    if mention_counts:
        result = max(mention_counts, key=lambda m: mention_counts[m])
        return result if result.strip() else "thing"

    # Fallback: infer type from answer using NER
    answer_doc = nlp(answer)
    for ent in answer_doc.ents:
        if ent.label_ in ("PERSON", "PER"):            return "person"
        elif ent.label_ in ("GPE", "LOC"):             return "place"
        elif ent.label_ in ("ORG"):                    return "organization"
        elif ent.label_ in ("DATE", "TIME"):           return "time"
        elif ent.label_ in ("PRODUCT", "WORK_OF_ART"): return "thing"

    for token in answer_doc:
        if token.pos_ == "PROPN":
            return "person"

    return "thing"


# ============================================================
# WH-WORD SELECTION
# ============================================================

def get_wh_word(answer_type, answer=None):
    """
    Dynamically determine wh-word.
    Currently always returns "which" — extend here for "who" vs "which" logic.
    """
    if answer:
        doc = nlp(answer)
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "PER"):
                return "which"

    doc = nlp(answer_type)
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "PER"):
            return "which"

    head_noun = doc[-1]
    person_doc = nlp("person")
    if head_noun.has_vector and person_doc[0].has_vector:
        similarity = head_noun.similarity(person_doc[0])
        if similarity > 0.5:
            return "which"

    return "which"


# ============================================================
# WH-WORD DETECTION
# ============================================================

def has_wh_word(question):
    """
    Returns True only if a wh-word is used as a QUESTION word,
    not as a subordinating conjunction.
    """
    doc = nlp(question)
    for token in doc:
        if token.text.lower() not in WH_WORDS:
            continue
        print(f"  [WH DEBUG] '{token.text}' | pos: {token.pos_} | tag: {token.tag_} | dep: {token.dep_}")

        if token.i <= 1:
            return True

        if token.text.lower() in ("when", "where", "how", "why"):
            prev_tokens = [t for t in doc if t.i < token.i and not t.is_space]
            if prev_tokens:
                prev = prev_tokens[-1]
                if prev.pos_ in ("VERB", "AUX") or prev.tag_ in ("VBG", "VBN", "VBZ", "VBD"):
                    continue
                if prev.text in (",", ";", ":"):
                    prev_tokens2 = [t for t in doc if t.i < prev.i and not t.is_space]
                    if prev_tokens2 and prev_tokens2[-1].pos_ in ("VERB", "AUX"):
                        continue

        if (token.dep_ in ("advmod", "attr", "nsubj", "dobj")
                and token.head.dep_ == "ROOT"):
            return True

    return False


# ============================================================
# SYNTACTIC HELPERS
# ============================================================

def get_subject(doc):
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            return token
    return None


def is_pronoun(token):
    return token.pos_ == "PRON"


def is_substring(pattern, question):
    return bool(re.search(pattern, question))


def get_verb_position(doc):
    tokens = list(doc)

    # Pass 1: TARGET_VERBS acting as actual verbs
    for token in tokens:
        if (token.text.lower() in TARGET_VERBS
                and token.pos_ == "VERB"
                and token.dep_ in ("ROOT", "relcl", "advcl", "ccomp", "xcomp")):
            return token.i

    # Pass 2: finite ROOT verb with a subject
    for token in tokens:
        if token.pos_ in ("VERB", "AUX") and token.dep_ == "ROOT":
            has_subject = any(c.dep_ in ("nsubj", "nsubjpass") for c in token.children)
            if has_subject:
                return token.i

    # Pass 3: relcl verbs
    for token in tokens:
        if (token.pos_ in ("VERB", "AUX")
                and token.dep_ == "relcl"
                and token.i > 0):
            return token.i

    # Pass 4: any finite verb, excluding participles
    # Pass 4: any finite verb, excluding participles, adjectival clauses,
    # and tokens that are likely nouns mistagged as verbs (e.g. "tongue")
    # Pass 4: any finite verb, excluding participles, adjectival clauses,
    # and verbs inside quoted/subordinate titles (dep: conj under pobj)
    for token in tokens:
        if (token.pos_ == "VERB"
                and token.dep_ not in ("acl", "amod", "partmod", "conj")
                and token.tag_ not in ("VBN", "VBG")
                and token.dep_ != "nsubj"):
            next_tok = tokens[token.i + 1] if token.i + 1 < len(tokens) else None
            if next_tok and next_tok.tag_ == "VBN":
                continue
            return token.i

    # Pass 5: ROOT with no subject (last resort)
    # Pass 5: ROOT only if it's actually a verb
    for token in tokens:
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
            return token.i

    return None  # give up — ROOT is a noun, not a verb

    return None


def get_head_of_verb(doc, verb_position):
    verb_token = doc[verb_position]

    # 1. Direct children with clear roles
    for child in verb_token.children:
        if child.dep_ in ("dobj", "attr") and child.pos_ in ("NOUN", "PROPN"):
            return child

    # 2. Copular verb
    if verb_token.lemma_ == "be":
        for child in verb_token.children:
            if child.dep_ == "attr":
                return child
        for token in doc[verb_position + 1:]:
            if token.pos_ in ("NOUN", "PROPN") and token.dep_ in ("attr", "appos"):
                return token

    # 3. relcl: walk UP to the noun it modifies
    if verb_token.dep_ == "relcl":
        head = verb_token.head
        if head.pos_ in ("NOUN", "PROPN"):
            return head

    # 4. Windowed fallback
    window = doc[verb_position + 1: verb_position + 6]
    for token in window:
        if token.pos_ in ("NOUN", "PROPN") and token.dep_ not in ("pobj",):
            return token

    # 5. Last resort
    for token in window:
        if token.pos_ in ("NOUN", "PROPN"):
            return token

    return None


def has_relative_clause(head_token):
    for child in head_token.children:
        if child.dep_ in ("relcl", "acl"):
            return True, child
    return False, None


def get_relative_clause_head(rel_clause_token):
    return rel_clause_token


# ============================================================
# SUB-ENTITY DETECTION (fixes semantic replacement errors)
# ============================================================

def is_sub_entity(chunk_head, answer_type):
    """
    Returns True if the demonstrative noun is a sub-entity of the answer type.
    E.g. "declaration" is a sub-entity of "work" → use "In which work does..." framing.
    """
    sub_entities = SUB_ENTITY_TYPES.get(answer_type.lower(), set())
    return chunk_head.text.lower() in sub_entities


# ============================================================
# SYNTACTIC VALIDITY
# ============================================================

def is_syntactically_valid(sentence):
    doc = nlp(sentence)
    tokens = list(doc)

    root = next((t for t in tokens if t.dep_ == "ROOT"), None)
    if root is None:
        return False

    has_subject = any(t.dep_ in ("nsubj", "nsubjpass") for t in tokens)
    if not has_subject:
        return False

    INVALID_AFTER_IS = {"of", "from", "at", "on", "for", "to", "with", "by"}
    for i, token in enumerate(tokens):
        if token.text.lower() == "is" and i + 1 < len(tokens):
            if tokens[i + 1].text.lower() in INVALID_AFTER_IS:
                return False

    return True


# ============================================================
# QUESTION CLEANING
# ============================================================

import re
import spacy

def clean_question(qbe):
    qbe = qbe.strip()

    # =========================
    # 1. Remove filler words early
    # =========================
    FILLER_WORDS = {"also", "too", "as well"}

    qbe = " ".join(
        t for t in qbe.split()
        if t.lower() not in FILLER_WORDS
    )

    # =========================
    # 2. Remove redundant WH copula patterns
    # =========================
    qbe = re.sub(r'\bis\s+(which|who|that)\b', '', qbe, flags=re.IGNORECASE)

    qbe = re.sub(
        r'\b(is|was|were|are)\s+(that\s+)?(?=[a-z]+ing\s)',
        r'\2',
        qbe,
        flags=re.IGNORECASE
    )

    qbe = re.sub(r'\b(is|was|were|are|be|been)\s+\1\b', r'\1', qbe, flags=re.IGNORECASE)

    qbe = re.sub(r'\s+', ' ', qbe).strip()

    # =========================
    # 3. spaCy parse
    # =========================
    doc = nlp(qbe.rstrip("?"))
    tokens = list(doc)

    has_any_verb = any(
        t.pos_ in ("VERB", "AUX") or
        t.tag_ in ("VBN", "VBG", "VBZ", "VBD", "VBP", "VB")
        for t in tokens
    )

    # =========================
    # 4. WH repair if no verb
    # =========================
    if not has_any_verb:
        wh_match = re.match(r'^(which|who|what)\s+', qbe, re.IGNORECASE)
        if wh_match:
            rest = qbe[wh_match.end():].rstrip("?").strip()
            rest_doc = nlp(rest)

            first_noun = next((t for t in rest_doc if t.pos_ in ("NOUN", "PROPN")), None)

            person_doc = nlp("person")
            is_person = (
                first_noun
                and first_noun.has_vector
                and person_doc[0].has_vector
                and first_noun.similarity(person_doc[0]) > 0.4
            )

            if is_person:
                qbe = f"Who is the {rest}?"
            else:
                qbe = f"What is the {rest}?"

    # =========================
    # 5. Re-parse for casing fix
    # =========================
    doc = nlp(qbe)
    tokens = list(doc)

    ner_spans = set()
    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            ner_spans.add(i)

    # =========================
    # 6. Rebuild sentence
    # =========================
    corrected_tokens = []

    for i, token in enumerate(tokens):

        # ---- FIX: another → one ----
        text = "one" if token.lower_ == "another" else token.text

        # capitalization rules
        if i == 0:
            corrected_tokens.append(text[0].upper() + text[1:])

        elif token.i in ner_spans:
            corrected_tokens.append(text)

        elif token.pos_ == "PROPN":
            corrected_tokens.append(text)

        elif token.is_upper and len(text) > 1:
            corrected_tokens.append(text if len(text) <= 3 else text.lower())

        elif text[0].isupper() and i > 0 and token.pos_ not in ("PROPN",):
            prev_token = tokens[i - 1]
            if prev_token.text in (".", "!", "?", ":"):
                corrected_tokens.append(text)
            else:
                corrected_tokens.append(text.lower())

        else:
            corrected_tokens.append(text)

    qbe = " ".join(corrected_tokens)

    # =========================
    # 7. Fix punctuation spacing
    # =========================
    qbe = qbe.replace(" ?", "?").replace(" .", ".").replace(" ,", ",")

    # FIX: Hussey 's → Hussey's
    qbe = re.sub(r"\s+'\s*", "'", qbe)
    qbe = re.sub(r"\s+'s\b", "'s", qbe)

    qbe = re.sub(r'\s+', ' ', qbe).strip()

    # =========================
    # 8. Ensure question mark
    # =========================
    if not qbe.endswith("?"):
        qbe = qbe.rstrip(".") + "?"
    CONTRACTION_FIXES = {
    "ca n't": "can't",
    "wo n't": "won't",
    "do n't": "don't",
    "did n't": "didn't",
    "is n't": "isn't",
    "are n't": "aren't",
    "was n't": "wasn't",
    "were n't": "weren't",
    "has n't": "hasn't",
    "have n't": "haven't",
    "had n't": "hadn't",
    "could n't": "couldn't",
    "would n't": "wouldn't",
    "should n't": "shouldn't",
    "must n't": "mustn't"
    }

    for k, v in CONTRACTION_FIXES.items():
        qbe = re.sub(rf"\b{k}\b", v, qbe)
    #sqbe = fix_redundant_identity_patterns(qbe)
    return qbe 

def fix_redundant_identity_patterns(qbe):

    qbe = re.sub(
        r'^(Which|What|Who)\s+([a-z]+)\s+is\s+\2\s+of\s+',
        r'Who is the \2 of ',
        qbe,
        flags=re.IGNORECASE
    )

    return qbe