"""
Tests for file-name matching.

A portfolio exported with spaces instead of underscores is the same portfolio.
Before this, the spreadsheets, six tests and half the eval all failed on that
one difference.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.filenames import name_key, resolve, same_name          # noqa: E402


def test_spacing_case_and_punctuation_do_not_matter():
    assert same_name("Microsoft_CFO_transition.pdf", "Microsoft CFO transition.pdf")
    assert same_name("Calder___Rowe_NVDA_Mar_2026.pdf", "Calder & Rowe NVDA Mar 2026.pdf")
    assert same_name("FX_rates.xlsx", "fx rates.XLSX")


def test_accents_fold_away():
    assert name_key("Banque Léman - relevé.pdf") == "banquelemanrelevepdf"


def test_different_documents_stay_different():
    assert not same_name("XOM 1Q26 results.pdf", "XOM 2Q26 results.pdf")
    assert not same_name("report.pdf", "report.xlsx")


def test_resolve_finds_the_file_however_it_was_saved():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "Holdings and trades.xlsx").write_text("x")
        assert resolve(tmp, "Holdings_and_trades.xlsx").name == "Holdings and trades.xlsx"
        assert resolve(tmp, "Holdings and trades.xlsx").name == "Holdings and trades.xlsx"


def test_a_missing_file_still_fails_where_it_is_opened():
    with tempfile.TemporaryDirectory() as tmp:
        path = resolve(tmp, "nothing.pdf")
        assert path == Path(tmp) / "nothing.pdf" and not path.exists()


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  pass  {name}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {name}: {e}")
            except Exception as e:                       # noqa: BLE001
                failed += 1
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'all tests passed' if not failed else f'{failed} failed'}")
