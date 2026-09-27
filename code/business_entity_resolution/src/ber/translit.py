"""Indic-script -> Latin transliteration.

Two layers:
1. Rule based: ``indic_transliteration`` (MIT licence) converts a native-script
   token to IAST, which we then simplify to plain ASCII with a few phonetic
   tweaks (anusvara -> n, drop the inherent final schwa, ...).
2. Learned: a dictionary ``rule_translit -> english_token`` mined from the
   *training* ground truth by aligning native-script names with the Latin
   Source-1 names they match (see ``learn_translit_dict``).  This recovers the
   English spelling of common words ("praivet" -> "private").

Unknown tokens fall back to the rule-based output, so unseen words still get a
reasonable phonetic Latin form.
"""
import functools
import re
import unicodedata
from collections import Counter, defaultdict

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

_BLOCKS = [
    (0x0900, 0x097F, sanscript.DEVANAGARI),
    (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA),
    (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU),
    (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
]
_NATIVE_RE = re.compile(r"[ऀ-ൿ]")
_IAST_FIX = str.maketrans({"ṃ": "n", "ṁ": "n", "ṅ": "n", "ñ": "n", "ṇ": "n", "ḥ": "", "ṛ": "r", "ṝ": "r"})


def has_native(s):
    return bool(_NATIVE_RE.search(s))


def _script_of(ch):
    o = ord(ch)
    for lo, hi, sc in _BLOCKS:
        if lo <= o <= hi:
            return sc
    return None


def _strip_marks(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


@functools.lru_cache(maxsize=500_000)
def rule_translit_token(tok):
    """Rule-based transliteration of a single native-script token to ASCII."""
    sc = None
    for ch in tok:
        sc = _script_of(ch)
        if sc:
            break
    if sc is None:
        return tok
    try:
        out = transliterate(tok, sc, sanscript.IAST)
    except Exception:  # pragma: no cover - library edge cases
        return tok
    out = out.translate(_IAST_FIX)
    out = _strip_marks(out).lower()
    # Drop the inherent final schwa when the native word ends in a consonant
    last = tok[-1]
    if (
        out.endswith("a")
        and len(out) > 2
        and unicodedata.category(last) == "Lo"
        and "VOWEL" not in unicodedata.name(last, "")
    ):
        out = out[:-1]
    out = re.sub(r"[^a-z0-9]", "", out)
    return out


class Transliterator:
    """Rule-based transliteration corrected by a learned token dictionary."""

    def __init__(self, mapping=None):
        self.mapping = dict(mapping or {})

    @functools.lru_cache(maxsize=500_000)
    def token(self, tok):
        rt = rule_translit_token(tok)
        return self.mapping.get(rt, rt)

    def text(self, s):
        """Transliterate every native-script token inside ``s`` (keeps Latin text)."""
        if not has_native(s):
            return s
        parts = re.split(r"(\s+|[,/()\-])", s)
        return "".join(self.token(p) if has_native(p) else p for p in parts)


def _latin_tokens(s):
    return re.findall(r"[a-z0-9]+", _strip_marks(s).lower())


def learn_translit_dict(pairs, min_count=2, min_share=0.5):
    """Learn ``rule_translit(native_token) -> english_token``.

    ``pairs`` yields (latin_text, native_text) for records known to match. Only
    pairs with identical token counts are aligned (position by position); the
    mapping is kept when it is seen ``min_count`` times and covers at least
    ``min_share`` of that native token's alignments.
    """
    co = defaultdict(Counter)
    for latin, native in pairs:
        lt = _latin_tokens(latin)
        nt = [t for t in re.split(r"[\s,/()\-]+", native) if t]
        if not lt or len(lt) != len(nt):
            continue
        for a, b in zip(lt, nt):
            if has_native(b):
                co[rule_translit_token(b)][a] += 1
    mapping = {}
    for rt, cnt in co.items():
        eng, c = cnt.most_common(1)[0]
        tot = sum(cnt.values())
        if c >= min_count and c / tot >= min_share:
            mapping[rt] = eng
    return mapping
