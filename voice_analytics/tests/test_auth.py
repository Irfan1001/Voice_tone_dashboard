"""API-key checking.

Auth is the one place where being wrong is silently expensive, so the checks here
cover the rejection paths rather than the happy one.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

import api.main as main


@pytest.fixture
def keys(monkeypatch):
    """Configure two keys, as a comma-separated VOICE_API_KEYS would."""
    monkeypatch.setattr(main, "API_KEYS", ["alpha-key-123", "beta-key-456"])


def test_a_valid_key_is_accepted(keys):
    assert main.require_key("alpha-key-123") is None
    assert main.require_key("beta-key-456") is None


def test_a_wrong_key_is_rejected(keys):
    with pytest.raises(HTTPException) as e:
        main.require_key("not-a-key")
    assert e.value.status_code == 401


def test_a_missing_key_is_rejected(keys):
    for absent in (None, "", "   "):
        with pytest.raises(HTTPException) as e:
            main.require_key(absent)
        assert e.value.status_code == 401


def test_a_prefix_of_a_valid_key_is_rejected(keys):
    with pytest.raises(HTTPException) as e:
        main.require_key("alpha-key-12")
    assert e.value.status_code == 401


def test_a_non_ascii_key_is_rejected_not_a_server_error(keys):
    """`secrets.compare_digest` raises TypeError on non-ASCII `str` input.

    Unhandled, that surfaced as a 500 with a traceback - so an unauthenticated
    caller could make the service look broken, and monitoring could not tell a
    bad credential from a real fault. It must be an ordinary 401.
    """
    for probe in ("café", "ключ", "🔑", "key with nbsp"):
        with pytest.raises(HTTPException) as e:
            main.require_key(probe)
        assert e.value.status_code == 401, f"{probe!r} should be 401"


def test_no_configured_keys_means_open_mode(monkeypatch):
    """Unset VOICE_API_KEYS starts OPEN deliberately, so local dev works. The
    warning and the /health `auth` field are what stop that shipping unnoticed."""
    monkeypatch.setattr(main, "API_KEYS", [])
    assert main.require_key(None) is None
    assert main.require_key("anything") is None
