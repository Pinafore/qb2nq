"""
constants.py
============
All shared constants used across heuristics.
"""

WH_WORDS = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how"}

TARGET_VERBS = {"name", "give", "identify"}

DEMONSTRATIVE_DETERMINERS = {"this", "these", "that", "those"}

POSSESSIVE_PRONOUNS = {"its", "his", "her", "their", "my", "our", "your"}

# FTP / "For X points" patterns used in quiz bowl questions
PATTERNS = [
    # FTP with optional comma and optional extra words before verb
    r'(ftp|FTP|Ftp)\s*,?\s*(give|identify|name)\s+(this|these|which|who)',

    # For X points with optional comma
    r'(For|for)\s+(ten|10|20|5|15)\s+(Points|points|POINTS)\s*,?\s*(give|identify|name)\s+(this|these|which|who)',

    # FTP anywhere in sentence — catch all remaining cases
    r'\bFTP\b',
    r'\bftp\b',
    r'\bFtp\b',
]

# Nouns that are sub-entities within an answer type
# Used to detect when a demonstrative refers to something INSIDE the answer,
# not the answer itself (e.g. "This declaration" inside a "work")
SUB_ENTITY_TYPES = {
    "work":         {"declaration", "scene", "act", "speech", "moment", "passage",
                     "line", "event", "letter", "monologue", "action", "statement"},
    "person":       {"action", "decision", "choice", "behavior", "remark", "statement"},
    "place":        {"area", "region", "location", "district", "zone"},
    "organization": {"policy", "decision", "rule", "act"},
}
