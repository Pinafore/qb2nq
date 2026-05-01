"""
heuristics/split_conjunction.py
================================
Heuristic 1: Split Conjunction

Splits a sentence with conjoined verb phrases into separate clauses.
Example:
  "Name this author who wrote X and received the Nobel Prize."
  -> ["Name this author who wrote X.", "Name this author who received the Nobel Prize."]
"""

import re
import spacy

nlp = spacy.load("en_core_web_sm")

from constants import PATTERNS, TARGET_VERBS


# ============================================================
# PRECONDITION
# ============================================================

def precondition_split_conjunction(qbe):
    """Returns True if the sentence has a conjunction between two verb phrases."""
    doc = nlp(qbe)
    for token in doc:
        if token.dep_ == "cc":
            if token.head.pos_ not in ("VERB", "AUX"):
                continue
            if token.head.dep_ not in ("ROOT", "conj"):
                continue
            for sibling in token.head.children:
                if sibling.dep_ == "conj" and sibling.pos_ in ("VERB", "AUX"):
                    if sibling.head.dep_ in ("ROOT", "conj"):
                        return True
    return False


# ============================================================
# HEURISTIC
# ============================================================

def heuristic_split_conjunction(qbe):
    """
    Split conjunction into multiple clauses.
    - If conjunct has its own subject -> use it as-is
    - If conjunct has no subject -> prepend shared subject noun chunk
    """
    doc    = nlp(qbe)
    tokens = list(doc)

    # Find all valid split points
    split_points = []
    for token in doc:
        if token.dep_ != "cc":
            continue
        head = token.head
        if head.pos_ not in ("VERB", "AUX"):
            continue
        if head.dep_ not in ("ROOT", "conj"):
            continue
        conj_sibling = next(
            (t for t in head.conjuncts
             if t.pos_ in ("VERB", "AUX")
             and t.dep_ in ("ROOT", "conj")),
            None
        )
        if conj_sibling:
            split_points.append((token, head, conj_sibling))

    if not split_points:
        return [qbe]

    # Extract shared prefix (subject noun chunk)
    shared_prefix = ""

    # Try FTP preamble + subject noun phrase first
    ftp_match = None
    for pattern in PATTERNS:
        m = re.search(pattern, qbe, re.IGNORECASE)
        if m:
            ftp_match = m
            break

    if ftp_match:
        after_ftp     = qbe[ftp_match.start():]
        after_ftp_doc = nlp(after_ftp)
        ftp_verb = next(
            (t for t in after_ftp_doc if t.text.lower() in ("name", "identify", "give")),
            None
        )
        if ftp_verb:
            head_noun = next(
                (t for t in after_ftp_doc
                 if t.dep_ in ("dobj", "nsubj")
                 and t.pos_ in ("NOUN", "PROPN")
                 and t.i > ftp_verb.i),
                None
            )
            if head_noun:
                head_noun_chunk = next(
                    (chunk for chunk in after_ftp_doc.noun_chunks if head_noun in chunk),
                    None
                )
                if head_noun_chunk:
                    prefix_end    = ftp_match.start() + head_noun_chunk.end_char
                    shared_prefix = qbe[:prefix_end].strip()

    # Fallback: use root verb's subject noun chunk
    # Fallback: use the actual nsubj of the ROOT verb
    if not shared_prefix:
        root_verb = next((t for t in tokens if t.dep_ == "ROOT"), None)
        print(f"  [SPLIT DEBUG] root_verb: '{root_verb.text if root_verb else None}' | dep: {root_verb.dep_ if root_verb else None}")
        if root_verb:
            subj = next(
                (t for t in root_verb.children if t.dep_ in ("nsubj", "nsubjpass", "csubj")),
                None
            )
            # Fallback: search all tokens for nsubj if not found under ROOT
            if subj is None:
                print(f"  [SPLIT DEBUG] all deps: {[(t.text, t.dep_, t.pos_) for t in tokens]}")
                print(f"  [SPLIT DEBUG] all deps: {[(t.text, t.dep_) for t in tokens]}")
                subj = next(
                    (t for t in tokens if t.dep_ in ("nsubj", "nsubjpass", "csubj")),
                    None
                )
                print(f"  [SPLIT DEBUG] fallback subj: '{subj.text if subj else None}'")
            print(f"  [SPLIT DEBUG] subj: '{subj.text if subj else None}'")
            if subj:
                print(f"  [SPLIT DEBUG] subj.subtree: {[t.text for t in subj.subtree]}")
                subj_subtree_end = max(t.i for t in subj.subtree) + 1
                shared_prefix = " ".join(
                    t.text for t in tokens[:subj_subtree_end]
                ).strip().rstrip(",")
                print(f"  [SPLIT DEBUG] shared_prefix result: '{shared_prefix}'")
    #print(f"  [SPLIT DEBUG] shared_prefix: '{shared_prefix}'")

    cc_positions = [sp[0].i for sp in split_points]

    # First clause: everything up to first cc
    first_cc_pos        = cc_positions[0]
    first_clause_tokens = [
        t for t in tokens
        if t.i < first_cc_pos
    ]
    first_clause = " ".join(t.text for t in first_clause_tokens).strip().rstrip(",")
    if not first_clause.endswith((".", "?", "!")):
        first_clause += "."
    clauses = [first_clause]

    # Subsequent clauses
    for i, (cc_token, head_verb, conj_verb) in enumerate(split_points):
        next_cc_pos     = cc_positions[i + 1] if i + 1 < len(cc_positions) else len(tokens)
        conjunct_tokens = [
            t for t in tokens
            if t.i > cc_token.i and t.i < next_cc_pos and t.pos_ != "PUNCT"
        ]

        print(f"  [SPLIT DEBUG] conj_verb: '{conj_verb.text}' | dep: {conj_verb.dep_}")
        print(f"  [SPLIT DEBUG] conjunct_tokens: {[t.text for t in conjunct_tokens]}")

        conj_has_subject = any(
            t.dep_ in ("nsubj", "nsubjpass") and t.head == conj_verb
            for t in conjunct_tokens
        )
        print(f"  [SPLIT DEBUG] conj_has_subject: {conj_has_subject}")

        if conj_has_subject:
            clause = " ".join(t.text for t in conjunct_tokens).strip()
        else:
            left_edge_i = min(t.i for t in conj_verb.subtree)
            conj_start_idx = next(
                (idx for idx, t in enumerate(conjunct_tokens) if t.i >= left_edge_i),
                0
            )
            verb_onward = " ".join(
                t.text for t in conjunct_tokens[conj_start_idx:]
            ).strip()
            clause = f"{shared_prefix} {verb_onward}".strip()

        print(f"  [SPLIT DEBUG] clause: '{clause}'")

        if not clause.endswith((".", "?", "!")):
            clause += "."
        clauses.append(clause)

    return clauses


# ============================================================
# POSTCONDITION
# ============================================================

def postcondition_split_conjunction(qbe_list):
    print(qbe_list)
    return [q.strip() for q in qbe_list if q.strip()]
