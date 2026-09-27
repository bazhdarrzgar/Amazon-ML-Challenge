"""Unit tests for the official macro-F0.5 metric (run: python -m pytest tests/)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber.metric import f05, macro_f05, detailed_report  # noqa: E402


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_official_example():
    # README: P = 2/3, R = 1  ->  0.714
    pred = ["S2-00047", "S2-00193", "S3-00812"]
    truth = ["S2-00047", "S3-00812"]
    assert round(f05(pred, truth), 3) == 0.714


def test_singleton_correct():
    assert f05([], []) == 1.0


def test_singleton_false_positive():
    assert f05(["S2-1"], []) == 0.0


def test_empty_prediction_nonsingleton():
    assert f05([], ["S2-1"]) == 0.0


def test_perfect_multi_match():
    assert f05(["S2-1", "S3-2", "S3-3"], ["S3-3", "S2-1", "S3-2"]) == 1.0


def test_partial_recall():
    # P = 1, R = 1/2 -> 1.25*0.5/(0.25+0.5) = 0.8333
    assert close(f05(["S2-1"], ["S2-1", "S3-2"]), 1.25 * 0.5 / 0.75)


def test_extra_false_positives():
    # P = 1/3, R = 1 -> 1.25*(1/3)/(0.25/3+1)
    assert close(f05(["S2-1", "S2-2", "S2-3"], ["S2-1"]), 1.25 * (1 / 3) / (0.25 / 3 + 1))


def test_no_overlap():
    assert f05(["S2-9"], ["S2-1"]) == 0.0


def test_duplicates_in_prediction_are_sets():
    assert f05(["S2-1", "S2-1"], ["S2-1"]) == 1.0


def test_macro_average_and_missing_entities():
    truth = {"a": [], "b": ["S2-1"], "c": ["S2-2", "S3-3"]}
    pred = {"a": [], "b": ["S2-1"]}  # c missing -> empty prediction -> 0
    assert close(macro_f05(pred, truth), (1 + 1 + 0) / 3)
    rep = detailed_report(pred, truth)
    assert rep["singleton_acc"] == 1.0
    assert rep["fn"] == 2 and rep["tp"] == 1 and rep["fp"] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all metric tests passed")
