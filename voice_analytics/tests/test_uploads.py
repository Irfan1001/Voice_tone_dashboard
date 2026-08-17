"""ZIP upload safety. Each test names the attack it prevents.

A regression here does not produce a wrong number; it writes attacker-chosen bytes
to an attacker-chosen path on the server.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from api.uploads import (
    MAX_ZIP_ENTRIES,
    UploadRejected,
    extract_zip,
    is_audio,
    safe_name,
)

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32   # header-shaped, not decodable


def build(tmp_path: Path, entries: list[tuple[str, bytes]], name="t.zip") -> Path:
    z = tmp_path / name
    with zipfile.ZipFile(z, "w") as f:
        for entry, data in entries:
            f.writestr(entry, data)
    return z


# ------------------------------------------------------------------ safe_name


def test_safe_name_strips_directories():
    assert safe_name("/etc/passwd") == "passwd"
    assert safe_name("../../secret.wav") == "secret.wav"
    assert safe_name("C:\\Windows\\evil.wav") == "evil.wav"


def test_safe_name_never_returns_empty_or_a_dotfile():
    assert safe_name("") == "upload"
    assert safe_name("...") == "upload"
    assert not safe_name(".hidden").startswith(".")


def test_safe_name_drops_shell_and_path_metacharacters():
    assert "/" not in safe_name("a/b;rm -rf $HOME.wav")
    assert ";" not in safe_name("a/b;rm -rf $HOME.wav")


def test_is_audio_rejects_non_audio():
    assert is_audio("a.wav") and is_audio("A.OGG")
    assert not is_audio("a.txt") and not is_audio("a.wav.exe") and not is_audio("a")


# -------------------------------------------------------------- zip handling


def test_zip_slip_entry_is_rejected_and_writes_nothing_outside(tmp_path):
    """`../../../../tmp/escaped.wav` must not land outside the extraction directory."""
    z = build(tmp_path, [("../../../../tmp/escaped.wav", WAV), ("ok.wav", WAV)])
    dest = tmp_path / "out"
    result = extract_zip(z, dest)
    assert [n for n, _ in result.files] == ["ok.wav"]
    assert any("unsafe path" in s for s in result.skipped)
    assert not (tmp_path.parent / "tmp" / "escaped.wav").exists()
    # everything written stays under dest
    for _, p in result.files:
        assert dest.resolve() in p.resolve().parents


def test_absolute_path_entry_is_rejected(tmp_path):
    z = build(tmp_path, [("/etc/cron.d/payload.wav", WAV), ("ok.wav", WAV)])
    result = extract_zip(z, tmp_path / "out")
    assert [n for n, _ in result.files] == ["ok.wav"]


def test_symlink_entry_is_skipped(tmp_path):
    """Anything that follows a symlink from an archive escapes the sandbox."""
    z = tmp_path / "s.zip"
    with zipfile.ZipFile(z, "w") as f:
        info = zipfile.ZipInfo("link.wav")
        info.external_attr = (0o120777 << 16)      # S_IFLNK
        f.writestr(info, "/etc/passwd")
        f.writestr("ok.wav", WAV)
    result = extract_zip(z, tmp_path / "out")
    assert [n for n, _ in result.files] == ["ok.wav"]
    assert any("symlink" in s for s in result.skipped)


def test_non_audio_entries_are_filtered_not_fatal(tmp_path):
    z = build(tmp_path, [("readme.txt", b"hello"), ("a.wav", WAV)])
    result = extract_zip(z, tmp_path / "out")
    assert [n for n, _ in result.files] == ["a.wav"]


def test_zip_with_no_audio_is_rejected_with_a_useful_message(tmp_path):
    z = build(tmp_path, [("a.txt", b"x"), ("b.pdf", b"y")])
    with pytest.raises(UploadRejected, match="no audio"):
        extract_zip(z, tmp_path / "out")


def test_too_many_entries_is_rejected(tmp_path):
    z = build(tmp_path, [(f"f{i}.wav", WAV) for i in range(MAX_ZIP_ENTRIES + 1)])
    with pytest.raises(UploadRejected, match="limit"):
        extract_zip(z, tmp_path / "out")


def test_zip_bomb_is_stopped_by_counting_written_bytes(tmp_path):
    """Counted as written, so a header lying about `file_size` cannot slip past."""
    z = build(tmp_path, [("bomb.wav", b"\0" * (8 * 1024 * 1024))])
    import api.uploads as up
    original = up.MAX_ENTRY_BYTES
    up.MAX_ENTRY_BYTES = 1024                     # 1 KB ceiling vs an 8 MB entry
    try:
        with pytest.raises(UploadRejected):
            extract_zip(z, tmp_path / "out")
    finally:
        up.MAX_ENTRY_BYTES = original


def test_flattened_name_collisions_do_not_overwrite(tmp_path):
    """Two folders can hold the same basename; the second must not replace the first."""
    z = build(tmp_path, [("a/call.wav", WAV), ("b/call.wav", WAV * 2)])
    result = extract_zip(z, tmp_path / "out")
    assert len(result.files) == 2
    assert len({p.name for _, p in result.files}) == 2


def test_extraction_order_is_deterministic(tmp_path):
    z = build(tmp_path, [("c.wav", WAV), ("a.wav", WAV), ("b.wav", WAV)])
    first = [n for n, _ in extract_zip(z, tmp_path / "o1").files]
    second = [n for n, _ in extract_zip(z, tmp_path / "o2").files]
    assert first == second == sorted(first)


def test_a_corrupt_archive_is_rejected_not_crashed(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a zip file at all")
    with pytest.raises(UploadRejected, match="readable ZIP"):
        extract_zip(bad, tmp_path / "out")
