"""Business-name and address normalisation.

Design principles (see EDA notes in the documentation):
* raw fields are never modified; normalisation produces *additional* views;
* native Indic script is transliterated (rule based + dictionary learned on
  the training ground truth, see ``translit.py``);
* accents are folded, case is lowered, punctuation is removed;
* names are split into ``core`` tokens (identity-bearing), canonical legal
  forms, titles and alias parts ("X f/k/a Y", "X dba Y", ...);
* addresses drop pure admin components (state / region / department / null
  placeholders) and canonicalise street-type abbreviations. Country-specific
  tables only *add* knowledge; an unseen country falls back to the union of
  all tables, so the pipeline never depends on a closed set of countries.
"""
import re
import unicodedata
from collections import Counter, defaultdict

from .translit import Transliterator, has_native, rule_translit_token

# ----------------------------------------------------------------------------
# Static tables (generic domain knowledge, not entity data)
# ----------------------------------------------------------------------------
US_STATES = {
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
    "puerto rico": "pr",
}
INDIA_STATES = {
    "maharashtra": "mh", "delhi": "dl", "uttar pradesh": "up", "karnataka": "ka",
    "tamil nadu": "tn", "tamilnadu": "tn", "gujarat": "gj", "west bengal": "wb",
    "telangana": "tg", "haryana": "hr", "kerala": "kl", "keralam": "kl", "rajasthan": "rj",
    "bihar": "br", "madhya pradesh": "mp", "andhra pradesh": "ap", "orissa": "od",
    "odisha": "od", "punjab": "pb", "goa": "ga", "assam": "as", "jharkhand": "jh",
    "chhattisgarh": "cg", "uttarakhand": "uk", "himachal pradesh": "hp",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
}
FRANCE_ADMIN = {
    "hauts de france", "nouvelle aquitaine", "pays de la loire", "ile de france",
    "auvergne rhone alpes", "occitanie", "grand est", "bretagne", "normandie",
    "bourgogne franche comte", "centre val de loire", "provence alpes cote d azur", "corse",
    "nord", "pas de calais", "gironde", "loire atlantique", "somme", "aisne", "oise",
    "vendee", "maine et loire", "sarthe", "mayenne", "landes", "dordogne",
    "pyrenees atlantiques", "lot et garonne", "charente", "charente maritime",
}
COUNTRY_WORDS = {"us", "usa", "united states", "india", "france", "bharat"}
NULL_WORDS = {"null", "n a", "na", "none", "nil", "nan", "", "-", "null null"}


def _admin_set(states):
    return set(states) | set(states.values())


ADMIN_BY_COUNTRY = {
    "us": _admin_set(US_STATES),
    "india": _admin_set(INDIA_STATES),
    "france": set(FRANCE_ADMIN),
}
ADMIN_ALL = set().union(*ADMIN_BY_COUNTRY.values())
STATE_CANON = {**{k: v for k, v in US_STATES.items()}, **{k: v for k, v in INDIA_STATES.items()}}

# street / address abbreviation canonicalisation (long form -> short form)
ADDR_CANON = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "drive": "dr", "avenue": "ave",
    "av": "ave", "lane": "ln", "court": "ct", "circle": "cir", "place": "pl", "boulevard": "blvd",
    "bd": "blvd", "boul": "blvd", "trail": "trl", "township": "twp", "highway": "hwy",
    "terrace": "ter", "parkway": "pkwy", "fort": "ft", "mount": "mt", "sainte": "ste",
    "suite": "ste", "apartment": "apt", "floor": "fl", "building": "bldg", "north": "n",
    "south": "s", "east": "e", "west": "w", "square": "sq", "number": "no", "nr": "near",
    "opposite": "opp", "sector": "sec", "colony": "col", "nagar": "ngr", "chowk": "chk",
    "extension": "extn", "ext": "extn", "expressway": "expy", "center": "ctr", "centre": "ctr",
    "heights": "hts", "junction": "jct", "point": "pt", "route": "rte", "estate": "est",
    "cross": "crs", "main": "mn", "first": "1st", "second": "2nd", "third": "3rd",
    "fourth": "4th", "fifth": "5th",
}
FR_ADDR_CANON = {
    "r": "rue", "all": "allee", "che": "chemin", "chem": "chemin", "imp": "impasse",
    "rte": "route", "fg": "faubourg", "fbg": "faubourg", "qu": "quai", "crs": "cours",
    "sq": "square", "res": "residence", "lot": "lotissement", "hlm": "hlm", "pl": "place",
}
CITY_ALIAS = {
    "bengaluru": "bangalore", "gurugram": "gurgaon", "calcutta": "kolkata", "bombay": "mumbai",
    "madras": "chennai", "trivandrum": "thiruvananthapuram", "ahmadabad": "ahmedabad",
    "kochi": "cochin", "mysuru": "mysore", "vadodra": "vadodara", "baroda": "vadodara",
}
ADDR_NOISE_TOKENS = {"null", "cdp"}

LEGAL_CANON = {
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd", "private": "pvt", "pvt": "pvt",
    "incorporated": "inc", "inc": "inc", "lnc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cie": "co", "compagnie": "co", "llc": "llc", "llp": "llp",
    "lp": "lp", "pc": "pc", "pllc": "pllc", "plc": "plc", "pa": "pa", "ltee": "ltd",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "ei": "ei", "scop": "scop", "gie": "gie", "selarl": "selarl", "gmbh": "gmbh",
    "ag": "ag", "bv": "bv", "nv": "nv", "opc": "opc", "psc": "psc", "chtd": "chtd",
}
TITLE_TOKENS = {"mr", "mrs", "ms", "miss", "dr", "shri", "sri", "smt", "messrs", "mssrs", "m s", "sh", "kum", "prof"}
ALIAS_CONNECTORS = [
    "formerly known as", "also known as", "doing business as", "trading as", "known as",
    "formerly", "fka", "aka", "dba", "nee", "t a", "d b a", "f k a", "a k a",
]
DOMAIN_RE = re.compile(
    r"\b(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|in|co\.in|co|fr|biz|info|us|io)\b"
)
OCR_FIX = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e", "4": "a", "7": "t", "8": "b"})
_ALIAS_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in ALIAS_CONNECTORS) + r")\b")

_PUNCT_RE = re.compile(r"[^0-9a-z]+")
_DROP_NAME_TOKENS = {"and", "the", "www", "com", "of", "fka", "aka", "dba", "nee", "formerly"}
_DIGITS_RE = re.compile(r"\d+")


def fold(s):
    """Unicode NFKD accent folding + lower-case (keeps only ASCII-ish text)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return (
        s.lower()
        .replace("ß", "ss").replace("æ", "ae").replace("œ", "oe").replace("ø", "o")
        .replace("ł", "l").replace("đ", "d").replace("ı", "i")
    )


def _fix_ocr_token(t):
    """Undo OCR-style digit-for-letter swaps inside alphabetic tokens (5ervices, c0m)."""
    if t.isdigit() or t.isalpha():
        return t
    letters = sum(c.isalpha() for c in t)
    if letters >= len(t) - 1 and letters >= 2:
        return t.translate(OCR_FIX)
    return t


def native_comp_key(comp):
    """Rule-based transliteration key of a whole native-script address component."""
    return " ".join(rule_translit_token(t) for t in comp.split() if t)


def learn_addr_comp_map(pairs, min_count=5, min_share=0.6):
    """Learn native-script address component -> admin name (state) mapping.

    ``pairs`` yields (source1_address, country, other_address) for training
    matches. For every native component of ``other_address`` we count the admin
    components present in the Source-1 address and keep a dominant mapping.
    """
    co = defaultdict(Counter)
    for a1, country, a2 in pairs:
        if not has_native(a2):
            continue
        c = fold(country or "").strip()
        admin = ADMIN_BY_COUNTRY.get(c, ADMIN_ALL)
        s1_admin = [
            " ".join(_PUNCT_RE.sub(" ", fold(x)).split()) for x in a1.split(",")
        ]
        s1_admin = [x for x in s1_admin if x in admin]
        if not s1_admin:
            continue
        for comp in a2.split(","):
            comp = comp.strip()
            if comp and has_native(comp):
                for x in s1_admin:
                    co[native_comp_key(comp)][x] += 1
    out = {}
    for k, cnt in co.items():
        v, n = cnt.most_common(1)[0]
        if n >= min_count and n / sum(cnt.values()) >= min_share:
            out[k] = v
    return out


class Normalizer:
    """Holds the transliteration dictionaries and exposes name/address normalisers."""

    def __init__(self, translit_map=None, addr_comp_map=None):
        self.tr = Transliterator(translit_map)
        self.addr_comp_map = dict(addr_comp_map or {})

    # ------------------------------------------------------------------ names
    def name(self, raw):
        """Return a dict with several normalised views of a business name."""
        s = raw or ""
        native = has_native(s)
        if native:
            s = self.tr.text(s)
        s = fold(s)
        domain = None
        m = DOMAIN_RE.search(s)
        if m:
            domain = m.group(1).replace("-", "")
            s = DOMAIN_RE.sub(" " + domain + " ", s)
        s = s.replace("&", " and ")
        # collapse dotted / slashed initials: l.l.c. -> llc, f/k/a -> fka, m/s -> ms
        s = re.sub(r"\b([a-z])[./](?=[a-z]\b)", r"\1", s)
        s = re.sub(r"\b([a-z])[./](?=[a-z]\b)", r"\1", s)
        s = re.sub(r"(?<=\d)[,.](?=\d)", "", s)
        s = _PUNCT_RE.sub(" ", s)
        toks = [_fix_ocr_token(t) for t in s.split()]
        full = " ".join(toks)

        # alias parts
        parts = [p.strip() for p in _ALIAS_RE.split(full)]
        alias_parts = [p for p in parts if p and p not in ALIAS_CONNECTORS]
        has_alias = len(alias_parts) > 1

        legal, core = [], []
        for t in toks:
            if t in LEGAL_CANON:
                legal.append(LEGAL_CANON[t])
            elif t in TITLE_TOKENS or t in _DROP_NAME_TOKENS:
                continue
            elif t.isdigit() and len(t) >= 4:  # phone / store numbers
                continue
            else:
                core.append(t)
        core_parts = []
        if has_alias:
            for p in alias_parts:
                cp = [
                    t for t in p.split()
                    if t not in LEGAL_CANON and t not in TITLE_TOKENS
                    and t not in _DROP_NAME_TOKENS
                    and not (t.isdigit() and len(t) >= 4)
                ]
                if cp:
                    core_parts.append(" ".join(cp))
        return {
            "n_full": full,
            "n_core": " ".join(core),
            "n_compact": "".join(core),
            "n_legal": " ".join(sorted(set(legal))),
            "n_alias": "|".join(core_parts),
            "n_domain": domain or "",
            "n_native": native,
        }

    # -------------------------------------------------------------- addresses
    def address(self, raw, country):
        """Return a dict with normalised views of an address."""
        s = raw or ""
        native = has_native(s)
        if native:
            comps_raw = []
            for comp in s.split(","):
                if has_native(comp):
                    key = native_comp_key(comp.strip())
                    comp = self.addr_comp_map.get(key) or self.tr.text(comp)
                comps_raw.append(comp)
            s = ",".join(comps_raw)
        s = fold(s)
        c = fold(country or "").strip()
        admin = ADMIN_BY_COUNTRY.get(c, ADMIN_ALL)
        is_fr = c == "france"
        comps, state = [], ""
        for comp in s.split(","):
            cn = " ".join(_PUNCT_RE.sub(" ", comp).split())
            if cn in NULL_WORDS or cn in COUNTRY_WORDS:
                continue
            if cn in admin:
                state = STATE_CANON.get(cn, cn)
                continue
            comps.append(cn)
        toks = []
        for t in " ".join(comps).split():
            if t in ADDR_NOISE_TOKENS:
                continue
            if is_fr and t in FR_ADDR_CANON:
                t = FR_ADDR_CANON[t]
            t = ADDR_CANON.get(t, t)
            t = CITY_ALIAS.get(t, t)
            toks.append(t)
        nums = _DIGITS_RE.findall(" ".join(comps))
        return {
            "a_core": " ".join(toks),
            "a_nums": " ".join(nums),
            "a_state": state,
            "a_native": native,
        }
