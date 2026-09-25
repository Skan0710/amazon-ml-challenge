"""
Script-aware normalisation utilities for business names and addresses (Phase 2).

Design rules
------------
* Never destroy information: every function returns *additional* representations,
  the raw value is always kept (``NameRepr.raw`` / ``AddressRepr.raw``).
* Deterministic and offline: only the Python standard library is used.
* Script-aware: Latin accents are stripped **only on Latin characters**. Indic
  text is never passed through generic accent stripping (that deletes vowel signs).
* Conservative: legal forms, aliases, websites and states are recognised through
  explicit, configurable tables — no fuzzy guessing.

Representation ladder (names)::

    raw -> basic -> punct -> canonical (legal forms mapped) -> core (legal forms,
    junk, leading 'the'/'m/s' removed) -> core_latin (transliterated core) -> compact / phonetic

Representation ladder (addresses)::

    raw -> basic -> components (noise-free, canonical tokens, order kept) -> normalized
        -> structured parts (house number, street, city, state, unit, po box, remaining tokens)
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Iterable, Optional

__all__ = [
    "normalize_basic_text", "is_missing_value", "strip_latin_accents",
    "detect_script", "script_breakdown",
    "transliterate_text", "phonetic_key",
    "NameRepr", "normalize_name", "extract_name_features",
    "AddressRepr", "normalize_address", "extract_address_features", "normalize_state",
    "word_tokens", "alnum_tokens", "char_ngrams", "numeric_tokens", "normalize_records", "normalize_records_chunk",
    "LEGAL_FORM_VARIANTS", "ALIAS_MARKERS", "US_STATES", "INDIA_STATES", "STREET_SUFFIXES",
]

# =============================================================================
# Configuration tables (explicit and editable)
# =============================================================================

#: Canonical legal form -> (variants, strength). Variants are token sequences *after*
#: punctuation normalisation (lower-case, dots removed). "strong" forms are removed from
#: the core name wherever they occur; "weak" forms (words that also occur as ordinary
#: words, e.g. "company", "private") are only removed at the start/end of the name.
#: Indic variants are *transliterations* (see ``transliterate_text``) of the native
#: legal words that occur in the training sources, e.g. लिमिटेड / లిమిటెడ్ -> "limited".
LEGAL_FORM_VARIANTS: dict[str, tuple[tuple[str, ...], str]] = {
    "private limited": ((
        "private limited", "pvt ltd", "pvt limited", "private ltd", "p ltd", "pvt ltd co",
        "pra li",  # Indic abbreviation प्रा. लि. / પ્રા. લિ. / ਪ੍ਰਾ. ਲਿ.
    ), "strong"),
    "public limited": (("public limited", "public ltd"), "strong"),
    "limited liability partnership": (("limited liability partnership",), "strong"),
    "limited liability company": (("limited liability company", "limited liability co"), "strong"),
    "llc": (("llc",), "strong"),
    "llp": (("llp", "elelpi"), "strong"),          # "elelpi": एलएलपी / ఎల్‌ఎల్‌పీ / എൽഎൽപി ...
    "pllc": (("pllc",), "strong"),
    "plc": (("plc",), "strong"),
    "inc": (("inc", "incorporated"), "strong"),
    "corp": (("corp", "corporation"), "strong"),
    "limited": (("ltd", "limited", "limtid", "limitet", "limittad", "li"), "weak"),
    "private": (("pvt", "private", "praivet", "praibhet", "piraivet", "praivatt", "pra"), "weak"),
    "co": (("co", "company"), "weak"),
    "lp": (("lp",), "weak"),
    "pc": (("pc",), "weak"),
    "pa": (("pa",), "weak"),
    "opc": (("opc",), "weak"),
}
# "ltd"/"pvt"/"corp" are unambiguous even though their long forms are "weak".
_STRONG_SINGLE = {"ltd", "pvt"}
# Weak forms that are only removed at the END of a name ("Co Op Bank" keeps "co").
_END_ONLY = {"co", "pa", "pc", "lp", "opc"}
# Tokens that are only treated as legal forms when *adjacent to another legal form*
# (too short / too ambiguous on their own).
_CONTEXT_ONLY = {"li", "pra"}

#: Alias / trade-name markers (matched on basic-normalised text, dots already removed).
ALIAS_MARKERS: tuple[str, ...] = (
    "formerly known as", "also known as", "doing business as", "trading as",
    "a/k/a", "f/k/a", "d/b/a", "t/a", "aka", "fka", "dba", "formerly",
)

WEBSITE_TLDS: tuple[str, ...] = (
    "com", "net", "org", "in", "co.in", "org.in", "net.in", "co", "biz", "info", "us", "io",
    "co.uk", "me", "online", "shop", "store", "tech", "site",
)

MISSING_TOKENS = frozenset({"", "null", "<null>", "none", "nan", "n/a", "na", "-", "--", "nil"})

US_STATES: dict[str, str] = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    "puerto rico": "pr", "guam": "gu", "us virgin islands": "vi", "virgin islands": "vi",
    "american samoa": "as", "northern mariana islands": "mp",
}

#: Indian states / UTs: every accepted spelling -> one canonical code. Native-script
#: spellings are the exact forms that occur in the training sources (S2/S3).
INDIA_STATES: dict[str, str] = {
    # English names and common variants
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "keralam": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz",
    "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj",
    "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "andaman and nicobar islands": "an", "chandigarh": "ch", "delhi": "dl", "nct of delhi": "dl",
    "jammu and kashmir": "jk", "ladakh": "la", "lakshadweep": "ld", "puducherry": "py",
    "pondicherry": "py", "dadra and nagar haveli and daman and diu": "dn",
    "dadra and nagar haveli": "dn", "daman and diu": "dd",
    # alternative codes -> canonical code
    "ct": "cg", "or": "od", "ts": "tg", "ut": "uk",
    # native-script spellings observed in S2/S3
    "महाराष्ट्र": "mh", "दिल्ली": "dl", "उत्तर प्रदेश": "up", "हरियाणा": "hr", "राजस्थान": "rj",
    "बिहार": "br", "मध्य प्रदेश": "mp", "ಕರ್ನಾಟಕ": "ka", "தமிழ்நாடு": "tn", "পশ্চিমবঙ্গ": "wb",
    "ગુજરાત": "gj", "తెలంగాణ": "tg", "ఆంధ్రప్రదేశ్": "ap", "കേരളം": "kl", "ਪੰਜਾਬ": "pb",
    "ଓଡ଼ିଶା": "od",
}
_INDIA_CODES = frozenset(INDIA_STATES.values()) | {"ct", "or", "ts", "ut"}
_US_CODES = frozenset(US_STATES.values())

#: Street-type and address-word canonicalisation (USPS-style abbreviations).
STREET_SUFFIXES: dict[str, str] = {
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd", "drive": "dr",
    "lane": "ln", "court": "ct", "place": "pl", "boulevard": "blvd", "circle": "cir",
    "highway": "hwy", "hiway": "hwy", "parkway": "pkwy", "terrace": "ter", "trail": "trl",
    "square": "sq", "crossing": "xing", "point": "pt", "ridge": "rdg", "mount": "mt",
    "mountain": "mtn", "fort": "ft", "heights": "hts", "center": "ctr", "centre": "ctr",
    "expressway": "expy", "freeway": "fwy", "turnpike": "tpke", "extension": "ext",
    "junction": "jct", "village": "vlg", "valley": "vly", "creek": "crk", "grove": "grv",
    "harbor": "hbr", "island": "is", "meadows": "mdws", "station": "sta", "saint": "st",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "apartment": "apt", "suite": "ste", "building": "bldg", "floor": "fl", "room": "rm",
    "near": "nr", "opposite": "opp", "post": "po", "number": "no", "house": "h",
}
_ORDINAL_WORDS = {
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
    "eleventh": "11th", "twelfth": "12th",
}
_UNIT_WORDS = frozenset({"unit", "apt", "ste", "rm", "fl", "bldg", "pmb"})

# =============================================================================
# Basic text
# =============================================================================

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_PUNCT_UNIFY = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'", "\u201c": '"', "\u201d": '"',
                              "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ", "\u3000": " "})
_WS_RE = re.compile(r"\s+")
# a single letter followed by a dot, repeated: l.l.c. / p.c. / u.s.a -> llc / pc / usa
_DOTTED_ABBR_RE = re.compile(r"(?<![^\W\d_])((?:[^\W\d_]\.){1,}[^\W\d_])\.?(?![^\W\d_])")
_NON_DECIMAL_DOT_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")
_DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)*?[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"
    r"\.(" + "|".join(re.escape(t) for t in sorted(WEBSITE_TLDS, key=len, reverse=True)) + r")/?$"
)
_DOMAIN_ANY_RE = re.compile(
    r"(?:^|\s)(?:www\.)?[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.(?:" + "|".join(re.escape(t) for t in WEBSITE_TLDS) + r")(?=\s|$)"
)


def is_missing_value(value: object) -> bool:
    """True for None, NaN, empty/whitespace strings and placeholder tokens (null, N/A, <NULL> ...)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    s = str(value).strip().casefold()
    return s in MISSING_TOKENS


def _to_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value if isinstance(value, str) else str(value)


def normalize_basic_text(text: object) -> str:
    """Conservative normalisation that keeps every alphanumeric character and every script.

    NFKC -> remove zero-width chars -> unify quotes/dashes -> casefold -> collapse dotted
    abbreviations (``L.L.C.`` -> ``llc``) -> other non-decimal dots become spaces (website
    tokens such as ``example.com`` are kept intact) -> collapse whitespace.

    >>> normalize_basic_text("  ACME   Pvt. Ltd. ")
    'acme pvt ltd'
    """
    s = _to_text(text)
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).translate(_ZERO_WIDTH).translate(_PUNCT_UNIFY).casefold()
    if "." in s:
        parts = []
        for tok in s.split():
            if _DOMAIN_RE.match(strip_latin_accents(tok.strip("()[]<>*,;:!\"'"))):
                parts.append(tok)          # keep websites intact
                continue
            tok = _DOTTED_ABBR_RE.sub(lambda m: m.group(1).replace(".", ""), tok)
            parts.append(_NON_DECIMAL_DOT_RE.sub(" ", tok))
        s = " ".join(parts)
    return _WS_RE.sub(" ", s).strip()


_LATIN_SPECIAL = str.maketrans({"ø": "o", "æ": "ae", "œ": "oe", "đ": "d", "ł": "l", "ı": "i", "ð": "d", "þ": "th"})


def _is_latin_char(ch: str) -> bool:
    o = ord(ch)
    return o < 0x0250 or 0x1E00 <= o <= 0x1EFF or 0x2C60 <= o <= 0x2C7F or 0xA720 <= o <= 0xA7FF


def strip_latin_accents(text: str) -> str:
    """Remove diacritics from *Latin* letters only (``Cónsumer`` -> ``Consumer``).

    Combining marks that follow a non-Latin base character (e.g. Devanagari vowel signs)
    are preserved, so Indic text is never damaged.
    """
    if not text or text.isascii():
        return text
    out: list[str] = []
    prev_latin = False
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch):
            if not prev_latin:
                out.append(ch)
            continue
        prev_latin = _is_latin_char(ch)
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out)).translate(_LATIN_SPECIAL)


# =============================================================================
# Script detection
# =============================================================================

_INDIC_BLOCKS: tuple[tuple[int, int, str], ...] = (
    (0x0900, 0x097F, "devanagari"), (0x0980, 0x09FF, "bengali"), (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"), (0x0B00, 0x0B7F, "oriya"), (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"), (0x0C80, 0x0CFF, "kannada"), (0x0D00, 0x0D7F, "malayalam"),
    (0x0D80, 0x0DFF, "sinhala"), (0xA8E0, 0xA8FF, "devanagari"), (0x1CD0, 0x1CFF, "devanagari"),
)
INDIC_SCRIPTS = frozenset(b[2] for b in _INDIC_BLOCKS)


def _char_script(ch: str) -> Optional[str]:
    o = ord(ch)
    for lo, hi, name in _INDIC_BLOCKS:
        if lo <= o <= hi:
            return name
    cat = unicodedata.category(ch)
    if not cat.startswith("L"):
        return None                      # digits, punctuation, marks after Latin, spaces ...
    if _is_latin_char(ch):
        return "latin"
    try:
        return unicodedata.name(ch).split()[0].lower()   # e.g. 'cyrillic', 'arabic', 'cjk'
    except ValueError:
        return "unknown"


def script_breakdown(text: object) -> dict[str, int]:
    """Count letters (and Indic signs) per script, e.g. ``{'latin': 5, 'telugu': 12}``."""
    counts: dict[str, int] = {}
    for ch in _to_text(text):
        sc = _char_script(ch)
        if sc is not None:
            counts[sc] = counts.get(sc, 0) + 1
    return counts


def detect_script(text: object) -> str:
    """Script family of ``text``: ``'latin'``, ``'indic'``, ``'mixed'``, ``'other'`` or ``'unknown'``.

    * ``'indic'`` covers all Brahmic scripts used in India (Devanagari, Bengali, Gurmukhi,
      Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam, Sinhala) — use
      :func:`script_breakdown` for the specific script.
    * ``'mixed'`` = letters from more than one family (e.g. ``'North ఇంటర్నేషనల్'``).
    * ``'unknown'`` = no letters at all (None, empty, digits/punctuation only).
    """
    s = _to_text(text)
    if not s:
        return "unknown"
    if s.isascii():
        return "latin" if any(c.isalpha() for c in s) else "unknown"
    fams = set()
    for sc in script_breakdown(s):
        fams.add("indic" if sc in INDIC_SCRIPTS else ("latin" if sc == "latin" else "other"))
    if not fams:
        return "unknown"
    return fams.pop() if len(fams) == 1 else "mixed"


# =============================================================================
# Transliteration (Brahmic scripts -> loose Latin), offline and table driven
# =============================================================================
# All nine Indic Unicode blocks share the ISCII-derived layout: the same offset from the
# block start denotes the same letter (0x15 = KA in Devanagari, Bengali, ..., Malayalam).
# One offset table therefore transliterates every script. The output is a *loose,
# ASCII, English-oriented* romanisation (long/short vowels merged, retroflex and dental
# merged) because the Indic names in this data are phonetic spellings of English words
# ("प्राइवेट लिमिटेड" -> "praivet limited"). It is not a scholarly transliteration.

_BLOCK_BASE = {name: lo for lo, hi, name in _INDIC_BLOCKS if lo < 0x0E00}
_SCHWA_DELETING = frozenset({"devanagari", "bengali", "gurmukhi", "gujarati", "oriya"})

_CONS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j",
         0x1D: "jh", 0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t",
         0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b",
         0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh",
         0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h"}
_INDEP_VOWELS = {0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
                 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o",
                 0x14: "au", 0x60: "ri", 0x61: "li", 0x72: "a"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e",
           0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x4E: "e",
           0x4F: "aw", 0x62: "li", 0x63: "li"}
_NUKTA_SHIFT = {"k": "q", "j": "z", "ph": "f", "d": "r", "dh": "rh", "g": "g", "kh": "kh", "s": "sh"}
# script-specific extra consonants: (script, offset) -> (latin, has_inherent_vowel)
_EXTRA_CONS = {
    ("devanagari", 0x58): ("q", True), ("devanagari", 0x59): ("kh", True), ("devanagari", 0x5A): ("g", True),
    ("devanagari", 0x5B): ("z", True), ("devanagari", 0x5C): ("r", True), ("devanagari", 0x5D): ("rh", True),
    ("devanagari", 0x5E): ("f", True), ("devanagari", 0x5F): ("y", True),
    ("bengali", 0x5C): ("r", True), ("bengali", 0x5D): ("rh", True), ("bengali", 0x5F): ("y", True),
    ("bengali", 0x4E): ("t", False),
    ("oriya", 0x5C): ("r", True), ("oriya", 0x5D): ("rh", True), ("oriya", 0x5F): ("y", True),
    ("oriya", 0x71): ("w", True),
    ("gurmukhi", 0x59): ("kh", True), ("gurmukhi", 0x5A): ("g", True), ("gurmukhi", 0x5B): ("z", True),
    ("gurmukhi", 0x5C): ("r", True), ("gurmukhi", 0x5E): ("f", True),
    ("malayalam", 0x7A): ("n", False), ("malayalam", 0x7B): ("n", False), ("malayalam", 0x7C): ("r", False),
    ("malayalam", 0x7D): ("l", False), ("malayalam", 0x7E): ("l", False), ("malayalam", 0x7F): ("k", False),
    ("malayalam", 0x4E): ("r", False),
    ("telugu", 0x58): ("ts", True), ("telugu", 0x59): ("dz", True),
}
_SIGNS = {0x00: "n", 0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}   # candrabindu, anusvara, visarga, tippi
_ML_TTA = ("\u0D31\u0D4D\u0D31", "\u0D1F\u0D4D\u0D1F")  # Malayalam RRA+virama+RRA is pronounced "tt"
_ML_NTA = ("\u0D28\u0D4D\u0D31", "\u0D28\u0D4D\u0D1F")  # Malayalam NA+virama+RRA is pronounced "nt"
# per-script letter overrides for the loan-word spellings seen in the data
_SCRIPT_CONS = {("tamil", 0x1A): "s"}                                 # Tamil CA in loanwords ~ "s"
_SCRIPT_VOWEL = {("gurmukhi", 0x10): "e", ("gurmukhi", 0x48): "e"}    # Punjabi AI / AI-sign ~ "e"


def _translit_run(run: str, script: str) -> str:
    """Transliterate one contiguous run of a single Indic script (one word)."""
    base = _BLOCK_BASE[script]
    if script == "malayalam":
        run = run.replace(*_ML_TTA).replace(*_ML_NTA)
    # units: [kind, latin, vowel]  kind: C consonant (vowel None=inherent, ''=virama), V vowel, S sign, D digit
    units: list[list] = []
    aytham = False
    for ch in run:
        off = ord(ch) - base
        extra = _EXTRA_CONS.get((script, off))
        if extra is not None:
            units.append(["C", extra[0], None if extra[1] else ""])
        elif off in _CONS:
            lat = _SCRIPT_CONS.get((script, off), _CONS[off])
            if aytham and lat == "p":
                lat = "f"
            aytham = False
            units.append(["C", lat, None])
        elif off == 0x3C:                                   # nukta
            if units and units[-1][0] == "C":
                units[-1][1] = _NUKTA_SHIFT.get(units[-1][1], units[-1][1])
        elif off in _MATRAS:
            v = _SCRIPT_VOWEL.get((script, off), _MATRAS[off])
            if units and units[-1][0] == "C" and units[-1][2] is None:
                units[-1][2] = v
            else:
                units.append(["V", v, None])
        elif off == 0x4D:                                   # virama
            if units and units[-1][0] == "C":
                units[-1][2] = ""
        elif script == "tamil" and off == 0x03:            # aytham: ஃப = f
            aytham = True
        elif off in _SIGNS and not (script == "gurmukhi" and off == 0x71):
            units.append(["S", _SIGNS[off], None])
        elif off in _INDEP_VOWELS:
            units.append(["V", _SCRIPT_VOWEL.get((script, off), _INDEP_VOWELS[off]), None])
        elif 0x66 <= off <= 0x6F:
            units.append(["D", str(off - 0x66), None])
        # everything else (dandas, length marks, addak, avagraha ...) is dropped

    letters = [k for k, u in enumerate(units) if u[0] in "CV"]
    if script in _SCHWA_DELETING and len(letters) > 1:
        def has_vowel(k: int) -> bool:
            u = units[k]
            return u[0] == "V" or (u[0] == "C" and u[2] != "")
        # 1) word-final inherent vowel is silent
        last = letters[-1]
        if units[last][0] == "C" and units[last][2] is None and last == len(units) - 1:
            units[last][2] = ""
        # 2) inherent vowel directly before an independent vowel is silent (एल-एल-पी -> el-el-pi)
        for k in range(len(units) - 1):
            if units[k][0] == "C" and units[k][2] is None and units[k + 1][0] == "V":
                units[k][2] = ""
        # 3) medial schwa deletion  V C [a] C V  (right to left)
        first = letters[0]
        for k in range(len(units) - 2, first, -1):
            u = units[k]
            if u[0] != "C" or u[2] is not None:
                continue
            nxt, prv = units[k + 1], units[k - 1]
            if nxt[0] == "C" and has_vowel(k + 1) and prv[0] in "CV" and has_vowel(k - 1):
                u[2] = ""
    out = []
    for kind, lat, vowel in units:
        if kind == "C":
            out.append(lat + ("a" if vowel is None else vowel))
        else:
            out.append(lat)
    return "".join(out)


def _indic_block(ch: str) -> Optional[str]:
    o = ord(ch)
    if 0x0900 <= o <= 0x0D7F:
        return _INDIC_BLOCKS[(o - 0x0900) >> 7][2]
    return None


@lru_cache(maxsize=1 << 16)
def _translit_token(token: str) -> str:
    out: list[str] = []
    run: list[str] = []
    run_script: Optional[str] = None
    for ch in token:
        sc = _indic_block(ch)
        if sc is not None and sc == run_script:
            run.append(ch)
            continue
        if run:
            out.append(_translit_run("".join(run), run_script))  # type: ignore[arg-type]
            run = []
        if sc is not None:
            run, run_script = [ch], sc
        else:
            run_script = None
            out.append(ch)
    if run:
        out.append(_translit_run("".join(run), run_script))  # type: ignore[arg-type]
    return "".join(out)


def transliterate_text(text: object) -> str:
    """Latin rendering of ``text``: Indic runs are transliterated, Latin accents stripped.

    Latin characters are kept (casefolded), other scripts are left unchanged, whitespace
    is preserved token by token so that native tokens and Latin tokens stay aligned.

    >>> transliterate_text("इंडियन इंटरनेशनल प्राइवेट लिमिटेड")
    'indiyan intarneshnal praivet limited'
    """
    s = normalize_basic_text(text)
    if not s:
        return ""
    if s.isascii():
        return s
    return strip_latin_accents(" ".join(_translit_token(t) for t in s.split(" ")))


_PHON_RULES = (("chh", "c"), ("ch", "c"), ("sh", "s"), ("zh", "j"), ("ph", "f"), ("bh", "b"), ("kh", "k"),
               ("gh", "g"), ("jh", "j"), ("th", "t"), ("dh", "d"), ("ck", "k"), ("q", "k"), ("x", "ks"),
               ("z", "j"), ("w", "v"), ("c", "k"),
               # script-motivated merges: Bengali/Oriya spell English "v" with b/bh, and Tamil script
               # has no voiced/unvoiced contrast (trading -> டிரேடிங் "tireting"), so voicing is ignored
               ("v", "b"), ("b", "p"), ("d", "t"), ("g", "k"))
_VOWEL_RE = re.compile(r"(?<=.)[aeiouy]")
_REPEAT_RE = re.compile(r"(.)\1+")


def phonetic_key(text: object) -> str:
    """Loose consonant-skeleton key for cross-script comparison (per token, space separated).

    Aspirates collapse (bh->b, th->t ...), c/q->k, z->j, v/w/b->p and voiced stops merge with
    unvoiced ones (d->t, g->k), a leading vowel becomes ``a``, later vowels are dropped and
    repeated letters collapse. ``'private'`` and ``'praivet'`` -> ``'prpt'``.
    Intended as an *input to similarity scoring*, not as an exact-match identity.
    """
    s = transliterate_text(text)
    keys = []
    for tok in s.split():
        tok = "".join(c for c in tok if c.isascii() and c.isalnum())
        if not tok:
            continue
        if tok.startswith("yu"):                   # yunaited ~ united
            tok = tok[1:]
        for a, b in _PHON_RULES:
            tok = tok.replace(a, b)
        tok = _REPEAT_RE.sub(r"\1", _VOWEL_RE.sub("", tok))
        if tok[0] in "aeiou":                      # eastern ~ ishtarn
            tok = "a" + tok[1:]
        keys.append(tok)
    return " ".join(keys)


# =============================================================================
# Tokenisation utilities
# =============================================================================

# token characters: letters/digits of any script, plus combining marks (Indic vowel signs are
# marks and are *not* matched by \w in Python's re), minus dandas.
_TOKEN_RE = re.compile(r"(?:[^\W_]|[\u0300-\u036F\u0900-\u0963\u0966-\u0DFF])+")
_NUM_RE = re.compile(r"\d+")


def word_tokens(text: object) -> list[str]:
    """Whitespace tokens of the basic-normalised text (punctuation kept)."""
    s = normalize_basic_text(text)
    return s.split() if s else []


def alnum_tokens(text: object, strip_accents: bool = True) -> list[str]:
    """Letter/digit tokens of any script; punctuation splits tokens, apostrophes are dropped
    (``wenonah's`` -> ``wenonahs``). Latin accents are stripped when ``strip_accents``."""
    s = normalize_basic_text(text)
    if not s:
        return []
    s = s.replace("'", "")
    if strip_accents:
        s = strip_latin_accents(s)
    return _TOKEN_RE.findall(s)


def numeric_tokens(text: object) -> list[str]:
    """All digit runs with leading zeros removed (``'00123 Rd, 5291b'`` -> ``['123', '5291']``)."""
    s = normalize_basic_text(text)
    return [n.lstrip("0") or "0" for n in _NUM_RE.findall(s)]


def char_ngrams(text: object, n: int = 3, pad: bool = True) -> list[str]:
    """Character n-grams of each alnum token (``#`` marks token boundaries when ``pad``).

    Works on any script. Returns a list (order kept, duplicates kept) so callers can build
    either sets or TF-IDF counts.
    """
    grams: list[str] = []
    for tok in alnum_tokens(text):
        t = f"#{tok}#" if pad else tok
        if len(t) < n:
            grams.append(t)
            continue
        grams.extend(t[i:i + n] for i in range(len(t) - n + 1))
    return grams


# =============================================================================
# Names
# =============================================================================

@dataclass(frozen=True, slots=True)
class NameRepr:
    """All representations of one business name. ``raw`` is never modified."""
    raw: Optional[str]
    basic: str                    # normalize_basic_text
    punct: str                    # punctuation-normalised tokens, '&'->'and', Latin accents stripped
    canonical: str                # punct with legal forms replaced by canonical labels
    core: str                     # legal forms / junk / leading 'the', 'm/s' removed (original script)
    transliterated: str           # punct rendered in Latin (identity for Latin names)
    core_latin: str               # core rendered in Latin  <- main cross-script comparison key
    compact: str                  # core_latin without spaces and 'and' (compare with website labels)
    phonetic: str                 # phonetic_key(core_latin)
    script: str                   # detect_script(raw)
    legal_forms: tuple[str, ...]  # canonical legal forms found (in order)
    alias_marker: Optional[str]   # e.g. 'aka', 'dba', 'formerly known as'
    aliases: tuple[str, ...]      # core_latin of each side of an alias expression
    website: Optional[str]        # e.g. 'zionterm.com'
    website_label: Optional[str]  # e.g. 'zionterm'

    @property
    def variants(self) -> tuple[str, ...]:
        """Distinct Latin keys this name can be matched on (core, alias sides, website label)."""
        seen: list[str] = []
        for v in (self.core_latin, *self.aliases, self.website_label or ""):
            if v and v not in seen:
                seen.append(v)
        return tuple(seen)

    def to_dict(self) -> dict:
        return asdict(self)


def _build_legal_index() -> tuple[dict[tuple[str, ...], tuple[str, str]], int]:
    idx: dict[tuple[str, ...], tuple[str, str]] = {}
    for label, (variants, strength) in LEGAL_FORM_VARIANTS.items():
        for v in variants:
            toks = tuple(v.split())
            st = "strong" if (strength == "strong" or (len(toks) == 1 and toks[0] in _STRONG_SINGLE)) else "weak"
            idx[toks] = (label, st)
    return idx, max(len(k) for k in idx)


_LEGAL_INDEX, _LEGAL_MAXLEN = _build_legal_index()


def _legal_spans(tokens: list[str]) -> list[tuple[int, int, str, str]]:
    """Greedy longest-match legal-form spans: (start, end, label, strength)."""
    spans = []
    i = 0
    while i < len(tokens):
        for L in range(min(_LEGAL_MAXLEN, len(tokens) - i), 0, -1):
            hit = _LEGAL_INDEX.get(tuple(tokens[i:i + L]))
            if hit:
                spans.append((i, i + L, hit[0], hit[1]))
                i += L
                break
        else:
            i += 1
    # context-only tokens ("li", "pra") count only when next to another legal span
    keep = []
    for k, sp in enumerate(spans):
        if sp[1] - sp[0] == 1 and tokens[sp[0]] in _CONTEXT_ONLY:
            neighbours = [s for s in spans if s is not sp and (s[1] == sp[0] or s[0] == sp[1])]
            if not neighbours:
                continue
        keep.append(sp)
    return keep


_ALIAS_RE = re.compile(
    r"\s(" + "|".join(re.escape(m) for m in sorted(ALIAS_MARKERS, key=len, reverse=True)) + r")(?:\s*[:\-])?\s"
)
_LEADING_JUNK_RE = re.compile(r"^[^\w\u0900-\u0DFF]+")
_OCR_MAP = str.maketrans({"0": "o", "1": "l", "5": "s"})
_OCR_TOKEN_RE = re.compile(r"^[a-z]*[015][a-z]*$")


def _ocr_fold(tok: str) -> str:
    """Undo single OCR digit substitutions inside otherwise alphabetic Latin words
    (``5ervices`` -> ``services``, ``wi1cox`` -> ``wilcox``). Tokens with <3 letters or
    more than one digit are untouched (``1st``, ``24th``, ``b2b`` stay as they are)."""
    if (_OCR_TOKEN_RE.match(tok) and sum(c.isdigit() for c in tok) == 1
            and sum(c.isalpha() for c in tok) >= 3):
        return tok.translate(_OCR_MAP)
    return tok


def _punct_tokens(basic: str) -> list[str]:
    s = basic.replace("'", "").replace("&", " and ").replace("+", " ")
    return _TOKEN_RE.findall(strip_latin_accents(s))


def _core_tokens(tokens: list[str], latin: list[str]) -> tuple[list[int], list[str]]:
    """Indices of tokens kept in the core name + canonical legal forms found."""
    spans = _legal_spans(latin)
    drop: set[int] = set()
    for s, e, _, st in spans:
        if st == "strong":
            drop.update(range(s, e))
    weak = [sp for sp in spans if sp[3] == "weak"]
    changed = True
    while changed:                                   # peel weak forms off both ends
        changed = False
        kept = [i for i in range(len(tokens)) if i not in drop]
        if not kept:
            break
        for s, e, label, _ in weak:
            span = set(range(s, e))
            if span & drop:
                continue
            if (s == kept[0] and label not in _END_ONLY) or e - 1 == kept[-1]:
                drop |= span
                changed = True
    labels = [label for s, e, label, _ in spans if all(i in drop for i in range(s, e))]
    keep = [i for i in range(len(tokens)) if i not in drop]
    # dangling connectors left behind by legal-form removal ("j p morgan and [co]")
    while len(keep) > 1 and latin[keep[-1]] in ("and", "the", "of"):
        keep = keep[:-1]
    while len(keep) > 1 and latin[keep[0]] == "and":
        keep = keep[1:]
    # leading 'the' and Indian 'm/s' (messrs)
    while len(keep) > 1 and latin[keep[0]] == "the":
        keep = keep[1:]
    if len(keep) > 2 and latin[keep[0]] == "m" and latin[keep[1]] == "s":
        keep = keep[2:]
    if not keep:                                     # never return an empty core
        keep = list(range(len(tokens)))
    return keep, _canonical_legal_set(labels)


def _canonical_legal_set(labels: list[str]) -> list[str]:
    """Deduplicate legal forms and merge 'private' + 'limited' into 'private limited'."""
    out: list[str] = []
    for lab in labels:
        if lab not in out:
            out.append(lab)
    if "private" in out and "limited" in out:
        out = [x for x in out if x not in ("private", "limited")]
        out.insert(0, "private limited")
    return out


@lru_cache(maxsize=1 << 18)
def _normalize_name_cached(raw: str) -> NameRepr:
    basic = normalize_basic_text(raw)
    script = detect_script(raw)

    # website-like names (whole name is a domain, after junk prefix)
    website = website_label = None
    probe = strip_latin_accents(_LEADING_JUNK_RE.sub("", basic).strip())
    m = _DOMAIN_RE.match(probe) if " " not in probe else None
    if m:
        website = probe.removeprefix("https://").removeprefix("http://").removeprefix("www.").rstrip("/")
        website_label = m.group(1).split(".")[-1].replace("-", "")

    # alias expressions: "X a/k/a Y", "X dba: Y", "X formerly known as Y" (both sides non-empty)
    alias_marker = None
    alias_parts: list[str] = []
    am = _ALIAS_RE.search(f" {basic} ")
    if am:
        left, right = f" {basic} "[:am.start()].strip(), f" {basic} "[am.end():].strip()
        if _TOKEN_RE.search(left) and _TOKEN_RE.search(right):
            alias_marker = am.group(1)
            alias_parts = [left, right]
            basic_wo_marker = f"{left} {right}"
        else:
            basic_wo_marker = basic
    else:
        basic_wo_marker = basic

    tokens = _punct_tokens(basic)
    punct = " ".join(tokens)
    latin_tokens = [_translit_token(t) if not t.isascii() else t for t in tokens]
    latin_tokens = [strip_latin_accents(t) for t in latin_tokens]
    transliterated = " ".join(latin_tokens)

    # legal forms: recognised on Latin tokens, applied to aligned native tokens
    spans = _legal_spans(latin_tokens)
    canon_tokens: list[str] = []
    i = 0
    for s, e, label, _ in spans:
        canon_tokens.extend(tokens[i:s])
        canon_tokens.append(label)
        i = e
    canon_tokens.extend(tokens[i:])

    core_src = _punct_tokens(basic_wo_marker) if alias_marker else tokens
    core_lat = ([strip_latin_accents(_translit_token(t)) if not t.isascii() else t for t in core_src]
                if alias_marker else latin_tokens)
    keep, legal_forms = _core_tokens(core_src, core_lat)
    if website_label:
        website_label = _ocr_fold(website_label)
        core = core_latin = website_label
    else:
        core = " ".join(_ocr_fold(core_src[i]) if core_src[i].isascii() else core_src[i] for i in keep)
        core_latin = " ".join(_ocr_fold(core_lat[i]) for i in keep)

    aliases: list[str] = []
    for part in alias_parts:
        ptoks = _punct_tokens(part)
        plat = [strip_latin_accents(_translit_token(t)) if not t.isascii() else t for t in ptoks]
        pkeep, _ = _core_tokens(ptoks, plat)
        a = " ".join(_ocr_fold(plat[i]) for i in pkeep)
        if a:
            aliases.append(a)

    return NameRepr(
        raw=raw, basic=basic, punct=punct, canonical=" ".join(canon_tokens), core=core,
        transliterated=transliterated, core_latin=core_latin,
        compact="".join(t for t in core_latin.split() if t != "and") or core_latin.replace(" ", ""),
        phonetic=phonetic_key(core_latin), script=script, legal_forms=tuple(legal_forms),
        alias_marker=alias_marker, aliases=tuple(aliases), website=website, website_label=website_label,
    )


_EMPTY_NAME = NameRepr(None, "", "", "", "", "", "", "", "", "unknown", (), None, (), None, None)


def normalize_name(raw: object) -> NameRepr:
    """Build every representation of a business name (cached; safe for None/NaN/empty).

    >>> n = normalize_name("Wonderland Energy-Private-Limited")
    >>> n.core_latin, n.legal_forms
    ('wonderland energy', ('private limited',))
    """
    s = _to_text(raw)
    if not s.strip():
        return _EMPTY_NAME if raw is None else replace(_EMPTY_NAME, raw=s, basic=normalize_basic_text(s))
    return _normalize_name_cached(s)


def extract_name_features(name: object) -> dict:
    """Flat, model-ready descriptors of a name (accepts a raw value or a :class:`NameRepr`)."""
    n = name if isinstance(name, NameRepr) else normalize_name(name)
    raw = n.raw or ""
    breakdown = script_breakdown(raw)
    return {
        "name_script": n.script,
        "name_scripts": ",".join(sorted(breakdown)),
        "name_is_indic": n.script in ("indic", "mixed") and any(s in INDIC_SCRIPTS for s in breakdown),
        "name_is_missing": not n.basic,
        "name_n_tokens": len(n.core_latin.split()),
        "name_n_chars": len(n.compact),
        "name_legal_forms": "|".join(n.legal_forms),
        "name_has_legal_form": bool(n.legal_forms),
        "name_is_website": n.website is not None,
        "name_has_alias": n.alias_marker is not None,
        "name_alias_marker": n.alias_marker or "",
        "name_has_digit": any(c.isdigit() for c in n.punct),
        "name_has_junk_prefix": bool(_LEADING_JUNK_RE.match(raw.strip())) if raw.strip() else False,
        "name_has_latin_accent": raw != strip_latin_accents(raw),
        "name_has_ocr_digit": any(_ocr_fold(t) != t for t in n.transliterated.split()),
        "name_is_all_upper": raw.isupper(),
        "name_contains_domain": bool(_DOMAIN_ANY_RE.search(n.basic)),
    }


# =============================================================================
# Addresses
# =============================================================================

@dataclass(frozen=True, slots=True)
class AddressRepr:
    """All representations of one address. ``raw`` is never modified."""
    raw: Optional[str]
    is_missing: bool
    basic: str                       # normalize_basic_text
    components: tuple[str, ...]      # cleaned, canonical-token comma components, ORIGINAL ORDER
    normalized: str                  # ', '.join(components)
    tokens: tuple[str, ...]          # all canonical tokens in order
    house_number: Optional[str]      # token as written, normalised ('181b', 'c-71', '3700-3702')
    house_number_core: Optional[str] # leading digit group without zeros ('181', '71', '3700')
    street: Optional[str]            # street component without the house number ('farragut ave')
    city: Optional[str]
    places: tuple[str, ...]          # every digit-free, non-state component (city, district, locality ...)
    state: Optional[str]             # canonical code ('ny', 'mh')
    state_raw: Optional[str]         # component the state was read from
    unit: Optional[str]
    po_box: Optional[str]
    numbers: tuple[str, ...]         # every digit group, zero-stripped, unique, in order
    remaining_tokens: tuple[str, ...]# tokens not used by street/city/state/unit/po box
    noise_removed: int               # count of null/n/a/## artefacts removed
    script: str

    @property
    def component_set(self) -> frozenset[str]:
        """Order-insensitive view of the components (for shuffled-address comparison)."""
        return frozenset(self.components)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["component_set"] = sorted(self.component_set)
        return d


_EMPTY_ADDRESS = AddressRepr(
    raw=None, is_missing=True, basic="", components=(), normalized="", tokens=(), house_number=None,
    house_number_core=None, street=None, city=None, places=(), state=None, state_raw=None, unit=None,
    po_box=None, numbers=(), remaining_tokens=(), noise_removed=0, script="unknown")

_HASH_RE = re.compile(r"#{2,}")
# address tokens keep internal '-' and '/' (house numbers such as c-71, 12-1-331/c/1, 3700-3702)
_ADDR_TOKEN_RE = re.compile(
    r"(?:[^\W_]|[̀-ͯऀ-ॣ०-෿])+(?:[-/](?:[^\W_]|[̀-ͯऀ-ॣ०-෿])+)*")
_SPLIT_RE = re.compile(r"[-/]")
_LEAD_ZERO_RE = re.compile(r"(?<!\d)0+(?=\d)")
_POBOX_RE = re.compile(r"\b(?:p o box|po box|pobox|post box|post office box)\s*(?:no\s*)?([a-z0-9-]+)")
_PMB_RE = re.compile(r"\bpmb\s*([a-z0-9-]+)")
_HOUSE_TOKEN_RE = re.compile(r"^[a-z]{0,3}-?\d+[a-z]{0,2}(?:[-/][a-z0-9]+)*$")
_GLUED_MARKER_RE = re.compile(r"^h?no(\d[a-z0-9/-]*)$")          # 'no3-1621', 'hno12'
_MARKER_PREFIX = frozenset({"h", "plot", "flat", "door", "d", "shop", "survey", "sy", "kh", "khasra", "gat", "ward"})
_NON_STREET_WORDS = frozenset({"n", "s", "e", "w", "ne", "nw", "se", "sw", "no", "po", "h", "apt", "ste", "bldg",
                               "fl", "rm", "nr", "opp"})
_STREET_WORDS = frozenset(STREET_SUFFIXES.values()) - _NON_STREET_WORDS


def normalize_state(value: object, country: Optional[str] = None) -> Optional[str]:
    """Canonical state code for a state name/code/native spelling, or None.

    Country-aware: ``'IN'`` is Indiana for the US, ``'GA'`` is Georgia (US) or Goa (India).
    Without a country only unambiguous values are resolved. No fuzzy matching.
    """
    s = normalize_basic_text(value)
    if not s:
        return None
    s = _WS_RE.sub(" ", s.replace("&", " and ").replace("-", " ")).strip()
    c = (country or "").strip().casefold()
    us = US_STATES.get(s) or (s if s in _US_CODES else None)
    ind = INDIA_STATES.get(s) or (s if s in _INDIA_CODES else None)
    if c in ("us", "usa", "united states"):
        return us
    if c in ("india", "in", "ind"):
        return ind
    if us and ind:                                   # e.g. 'ar' = Arkansas / Arunachal Pradesh
        return None
    return us or ind


def _canon_word(tok: str) -> str:
    return _ORDINAL_WORDS.get(tok) or STREET_SUFFIXES.get(tok) or tok


def _clean_component(comp: str) -> tuple[list[str], int]:
    """Canonical tokens of one comma component + number of noise artefacts removed.

    Tokens with digits keep internal '-' / '/' and lose leading zeros ('00123' -> '123',
    'c-045' -> 'c-45'); pure words are split on '-' / '/' and canonicalised
    (street types, directions, ordinal words).
    """
    noise = 0
    if "#" in comp:
        noise += len(_HASH_RE.findall(comp))
        comp = comp.replace("#", "")
    out: list[str] = []
    for t in _ADDR_TOKEN_RE.findall(strip_latin_accents(comp.replace("'", ""))):
        if any(ch.isdigit() for ch in t):
            out.append(_LEAD_ZERO_RE.sub("", t))
            continue
        for p in _SPLIT_RE.split(t):
            if not p:
                continue
            if p in ("null", "nan", "none"):
                noise += 1
                continue
            out.append(_canon_word(p))
    if out and out[-1] == "cdp":
        out = out[:-1]
    return out, noise


def _find_house(toks: list[str]) -> Optional[tuple[str, int, int]]:
    """House number in one component via explicit markers: (number, span_start, span_end)."""
    for j, t in enumerate(toks):
        m = _GLUED_MARKER_RE.match(t)
        if m:
            start = j - 1 if j > 0 and toks[j - 1] in _MARKER_PREFIX else j
            return m.group(1), start, j + 1
        if j > 0 and toks[j - 1] == "box":           # "po box no 1117" is a PO box, not a house
            continue
        if t in ("no", "hno") and j + 1 < len(toks) and _HOUSE_TOKEN_RE.match(toks[j + 1]):
            start = j - 1 if j > 0 and toks[j - 1] in _MARKER_PREFIX else j
            return toks[j + 1], start, j + 2
    return None


@lru_cache(maxsize=1 << 18)
def _normalize_address_cached(raw: str, country: str) -> AddressRepr:
    basic = normalize_basic_text(raw)
    if is_missing_value(basic):
        return replace(_EMPTY_ADDRESS, raw=raw, basic=basic, noise_removed=1)

    noise = 0
    comps: list[list[str]] = []
    raw_comps: list[str] = []
    for part in basic.split(","):
        part = part.strip().strip('"').strip()
        if is_missing_value(part):
            noise += 1 if part else 0
            continue
        toks, n = _clean_component(part)
        noise += n
        if toks:
            comps.append(toks)
            raw_comps.append(part)

    # ---- state: last component that is exactly a known state (country aware, no fuzzy match)
    state = state_raw = None
    used: set[int] = set()
    for k in range(len(comps) - 1, -1, -1):
        code = normalize_state(raw_comps[k], country) or normalize_state(" ".join(comps[k]), country)
        if code:
            state, state_raw = code, raw_comps[k]
            comps[k] = [code]
            used.add(k)
            break

    # ---- PO box / PMB / unit components
    po_box = unit = None
    for k, toks in enumerate(comps):
        if k in used:
            continue
        text = " ".join(toks)
        m = _POBOX_RE.search(text)
        if m and po_box is None:
            po_box = _LEAD_ZERO_RE.sub("", m.group(1))
            if not _POBOX_RE.sub("", text).strip():
                used.add(k)
            continue
        m = _PMB_RE.search(text)
        if m and unit is None:
            unit = f"pmb {_LEAD_ZERO_RE.sub('', m.group(1))}"
            if len(toks) <= 2:
                used.add(k)
        elif unit is None and toks[0] in _UNIT_WORDS and k > 0:
            unit = text
            used.add(k)

    free = [k for k in range(len(comps)) if k not in used]

    # ---- house number: (1) explicit marker, (2) component starting with a number-like token
    house: Optional[str] = None
    house_idx: Optional[int] = None
    house_span: tuple[int, int] = (0, 0)
    for k in free:
        hit = _find_house(comps[k])
        if hit:
            house, house_idx, house_span = hit[0], k, (hit[1], hit[2])
            break
    if house is None:
        for k in free:
            if _HOUSE_TOKEN_RE.match(comps[k][0]):
                house, house_idx, house_span = comps[k][0], k, (0, 1)
                break

    # ---- street: a component with a street-type word, else the house component if it has words
    street = None
    street_idx: Optional[int] = None
    order = ([house_idx] if house_idx is not None else []) + [k for k in free if k != house_idx]
    for k in order:
        if any(t in _STREET_WORDS for t in comps[k]):
            street_idx = k
            break
    if street_idx is None and house_idx is not None:
        rest = comps[house_idx][:house_span[0]] + comps[house_idx][house_span[1]:]
        if any(t.isalpha() and len(t) > 1 for t in rest):
            street_idx = house_idx
    if street_idx is not None:
        stoks = comps[street_idx]
        if street_idx == house_idx:
            stoks = stoks[:house_span[0]] + stoks[house_span[1]:]
        elif house is None and _HOUSE_TOKEN_RE.match(stoks[0]):
            house, house_idx, house_span = stoks[0], street_idx, (0, 1)
            stoks = stoks[1:]
        street = " ".join(stoks) or None
        used.add(street_idx)
    if house_idx is not None and house_idx != street_idx:
        rest = comps[house_idx][:house_span[0]] + comps[house_idx][house_span[1]:]
        if not any(t.isalpha() and len(t) > 1 for t in rest):
            used.add(house_idx)                     # component was only "c-71" / "h no 69"
    house_core = None
    if house is not None:
        digits = _NUM_RE.search(house)
        house_core = (digits.group(0).lstrip("0") or "0") if digits else None

    places = tuple(" ".join(comps[k][:-1] if len(comps[k]) >= 2 and comps[k][-1] == "city" else comps[k])
                   for k in range(len(comps))
                   if k not in used and not any(ch.isdigit() for t in comps[k] for ch in t))
    # ---- city: last remaining component without digits ("mesa city" -> "mesa")
    city = None
    for k in range(len(comps) - 1, -1, -1):
        if k in used or any(ch.isdigit() for t in comps[k] for ch in t):
            continue
        toks = comps[k][:-1] if len(comps[k]) >= 2 and comps[k][-1] == "city" else comps[k]
        city = " ".join(toks)
        used.add(k)
        break

    comp_strs = tuple(" ".join(t) for t in comps)
    numbers: list[str] = []
    for n in _NUM_RE.findall(" ".join(comp_strs)):
        n = n.lstrip("0") or "0"
        if n not in numbers:
            numbers.append(n)
    return AddressRepr(
        raw=raw, is_missing=not comps, basic=basic, components=comp_strs, normalized=", ".join(comp_strs),
        tokens=tuple(t for toks in comps for t in toks), house_number=house, house_number_core=house_core,
        street=street, city=city, places=places, state=state, state_raw=state_raw, unit=unit, po_box=po_box,
        numbers=tuple(numbers), remaining_tokens=tuple(t for k, toks in enumerate(comps) if k not in used for t in toks),
        noise_removed=noise, script=detect_script(raw),
    )


def normalize_address(raw: object, country: Optional[str] = None) -> AddressRepr:
    """Build every representation of an address (cached; safe for None/NaN/empty).

    ``country`` (``'US'`` / ``'India'``) disambiguates state codes; it is optional.

    >>> a = normalize_address("00123 YEAGER RD, COALLTON, WV", "US")
    >>> a.house_number_core, a.street, a.city, a.state
    ('123', 'yeager rd', 'coallton', 'wv')
    """
    s = _to_text(raw)
    if is_missing_value(s):
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return _EMPTY_ADDRESS
        return replace(_EMPTY_ADDRESS, raw=s, basic=normalize_basic_text(s), noise_removed=1 if s.strip() else 0)
    return _normalize_address_cached(s, (country or "").strip().casefold())


def extract_address_features(address: object, country: Optional[str] = None) -> dict:
    """Flat, model-ready descriptors of an address (raw value or :class:`AddressRepr`)."""
    a = address if isinstance(address, AddressRepr) else normalize_address(address, country)
    raw = a.raw or ""
    return {
        "addr_is_missing": a.is_missing,
        "addr_script": a.script,
        "addr_n_components": len(a.components),
        "addr_n_tokens": len(a.tokens),
        "addr_has_house_number": a.house_number is not None,
        "addr_house_number": a.house_number or "",
        "addr_house_number_core": a.house_number_core or "",
        "addr_street": a.street or "",
        "addr_city": a.city or "",
        "addr_state": a.state or "",
        "addr_has_state": a.state is not None,
        "addr_unit": a.unit or "",
        "addr_po_box": a.po_box or "",
        "addr_n_numbers": len(a.numbers),
        "addr_numbers": " ".join(a.numbers),
        "addr_noise_removed": a.noise_removed,
        "addr_is_all_upper": raw.isupper(),
        "addr_has_zero_padded_number": bool(re.search(r"(?<!\d)0\d", raw)),
    }


# =============================================================================
# Batch helpers (for later materialisation, e.g. to Parquet, and for process pools)
# =============================================================================

def normalize_records(names: Iterable[object], addresses: Iterable[object],
                      countries: Iterable[object]) -> list[dict]:
    """Normalise aligned (name, address, country) sequences into flat dicts.

    The output columns are the compact, reusable subset intended for later phases
    (candidate generation / features); the raw values are *not* repeated here and
    should be joined back by ``entity_id``.
    """
    out: list[dict] = []
    for name, addr, country in zip(names, addresses, countries):
        c = _to_text(country) or None
        n = normalize_name(name)
        a = normalize_address(addr, c)
        out.append({
            "name_basic": n.basic, "name_core": n.core, "name_core_latin": n.core_latin,
            "name_compact": n.compact, "name_phonetic": n.phonetic, "name_script": n.script,
            "name_legal_forms": "|".join(n.legal_forms), "name_aliases": "|".join(n.aliases),
            "name_website_label": n.website_label or "",
            "addr_normalized": a.normalized, "addr_components": "|".join(a.components),
            "addr_house_number": a.house_number or "", "addr_house_number_core": a.house_number_core or "",
            "addr_street": a.street or "", "addr_city": a.city or "", "addr_state": a.state or "",
            "addr_unit": a.unit or "", "addr_po_box": a.po_box or "", "addr_numbers": " ".join(a.numbers),
            "addr_is_missing": a.is_missing,
        })
    return out


def normalize_records_chunk(chunk: tuple[list, list, list]) -> list[dict]:
    """Single-argument wrapper of :func:`normalize_records` for ``multiprocessing.Pool.map``."""
    return normalize_records(*chunk)
