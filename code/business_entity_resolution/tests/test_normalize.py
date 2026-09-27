"""Unit tests for name / address normalisation (run: python tests/test_normalize.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber.ids import decode, encode  # noqa: E402
from ber.normalize import Normalizer  # noqa: E402

N = Normalizer({"praivet": "private", "limited": "limited"}, {"karnatak": "karnataka"})


def test_legal_suffix_moved_and_removed_from_core():
    a, b = N.name("Viveros & Echeverry PC"), N.name("PC Viveros & Echeverry")
    assert a["n_core"] == b["n_core"] == "viveros echeverry"
    assert a["n_legal"] == b["n_legal"] == "pc"


def test_accents_and_ocr_digits():
    assert N.name("Modern Construction Énterprises")["n_core"] == "modern construction enterprises"
    assert N.name("Moncada 5ervices lnc")["n_core"] == "moncada services"


def test_dotted_initials_and_alias():
    d = N.name("Rizanex f/k/a Clear Consumer Global L.L.C.")
    assert d["n_legal"] == "llc"
    assert d["n_alias"] == "rizanex|clear consumer global"
    assert "fka" not in d["n_core"].split()


def test_domain_name():
    d = N.name("www.AnchorSmartCommunities.com")
    assert d["n_domain"] == "anchorsmartcommunities" and d["n_core"] == "anchorsmartcommunities"


def test_phone_and_titles_dropped():
    assert N.name("M/s Dynamic Consultants Pvt Ltd - 7648973034")["n_core"] == "dynamic consultants"


def test_native_script_name():
    d = N.name("ಇಂಡೋ ಇಂಡಸ್ಟ್ರೀಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್")
    assert d["n_native"] and d["n_legal"] == "ltd pvt"


def test_address_state_removed_and_abbrev():
    a = N.address("1908 A Casey Court, Barling, AR", "US")
    b = N.address("Arkansas, Barling, 1908 A Casey Ct", "US")
    assert sorted(a["a_core"].split()) == sorted(b["a_core"].split())
    assert a["a_state"] == b["a_state"] == "ar"


def test_address_native_state_and_nulls():
    a = N.address("1/9, HALIYAL, null, ಕರ್ನಾಟಕ", "India")
    assert a["a_state"] == "karnataka" or a["a_state"] == "ka"
    assert "null" not in a["a_core"]


def test_france_abbreviations():
    a = N.address("28 R. PIERRE MAUROY, LILLE, Nord", "France")
    b = N.address("28 Rue Pierre Mauroy, Lille, Hauts-de-France", "France")
    assert a["a_core"] == b["a_core"] == "28 rue pierre mauroy lille"


def test_unseen_country_does_not_crash():
    d = N.address("12 Some Str, Berlin, Brandenburg", "Germany")
    assert d["a_core"].startswith("12 some st")
    assert N.name("")["n_core"] == "" and N.address("", "")["a_core"] == ""


def test_id_codec_roundtrip():
    ids = ["S1-965667", "S2-681193310", "S3-11291185"]
    assert decode(encode(ids)) == ids


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all normalisation tests passed")
