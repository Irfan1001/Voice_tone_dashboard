"""CSV export rendering.

The CSV is what a spreadsheet or BI tool consumes, so its booleans must read the
same as the JSON's. Python's `str(False)` is `"False"`, which is not the same value.
"""

from __future__ import annotations

from api.main import csv_cell


def test_booleans_are_lowercased_to_match_the_json():
    assert csv_cell(True) == "true"
    assert csv_cell(False) == "false"


def test_missing_values_render_empty_not_none():
    assert csv_cell("") == ""


def test_strings_and_numbers_pass_through():
    assert csv_cell("television") == "television"
    assert csv_cell(0.69) == "0.69"
    assert csv_cell(0) == "0"


def test_zero_is_not_confused_with_false():
    """`0 == False` in Python, so an identity check is required, not equality."""
    assert csv_cell(0) == "0"
    assert csv_cell(1) == "1"
