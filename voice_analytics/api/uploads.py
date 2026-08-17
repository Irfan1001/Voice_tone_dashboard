"""Accepting untrusted audio and ZIP uploads safely.

Every guard answers a specific attack, and `tests/test_uploads.py` names each one:

* zip slip - absolute or parent-traversing entry names are rejected, not repaired
* zip bomb - per-entry and total DECOMPRESSED bytes are counted as written, so a
  lying header cannot slip past a header-only check
* entry-count flooding - capped
* symlink entries - skipped, since anything following one escapes the sandbox
* non-audio payloads - filtered by extension; an undecodable file fails its own
  item, not the job

The client-supplied filename is never trusted for anything but display.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXT = {".wav", ".ogg", ".opus", ".mp3", ".flac", ".m4a", ".aac", ".webm"}

MAX_UPLOAD_BYTES = 500 * 1024 * 1024        # 500 MB per request
MAX_ZIP_ENTRIES = 500                        # audio files per batch
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024   # 2 GB expanded, total
MAX_ENTRY_BYTES = 200 * 1024 * 1024         # 200 MB per file
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


class UploadRejected(Exception):
    """The upload is malformed or unsafe. Carries a message safe to return."""


@dataclass
class Extracted:
    files: list[tuple[str, Path]]     # (display name, path on disk)
    skipped: list[str]                # name -> why, for the response


def safe_name(name: str) -> str:
    """Basename only, sanitised. Never used to build a path from client input."""
    base = Path(name.replace("\\", "/")).name
    cleaned = _SAFE.sub("_", base).lstrip(".") or "upload"
    return cleaned[:120]


def save_stream(fileobj, dest: Path, max_bytes: int = MAX_UPLOAD_BYTES) -> int:
    """Stream to disk with a hard ceiling, rather than reading into memory - a
    500 MB `.read()` is 500 MB of RSS on a worker already holding the model weights.
    """
    written = 0
    with dest.open("wb") as out:
        while chunk := fileobj.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise UploadRejected(
                    f"upload exceeds {max_bytes // (1024 * 1024)} MB limit")
            out.write(chunk)
    if written == 0:
        dest.unlink(missing_ok=True)
        raise UploadRejected("uploaded file is empty")
    return written


def is_audio(name: str) -> bool:
    return Path(name).suffix.lower() in AUDIO_EXT


def extract_zip(zip_path: Path, dest_dir: Path) -> Extracted:
    """Extract audio entries only, with every guard above applied."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    files: list[tuple[str, Path]] = []
    skipped: list[str] = []
    total = 0

    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise UploadRejected(f"not a readable ZIP archive: {exc}") from exc

    with archive:
        infos = archive.infolist()
        audio_entries = [i for i in infos
                         if not i.is_dir() and is_audio(i.filename)]
        if not audio_entries:
            kinds = sorted({Path(i.filename).suffix.lower() or "(none)"
                            for i in infos if not i.is_dir()})[:8]
            raise UploadRejected(
                "ZIP contains no audio files. Accepted extensions: "
                f"{', '.join(sorted(AUDIO_EXT))}. Found: {', '.join(kinds) or 'nothing'}")
        if len(audio_entries) > MAX_ZIP_ENTRIES:
            raise UploadRejected(
                f"ZIP holds {len(audio_entries)} audio files, limit is "
                f"{MAX_ZIP_ENTRIES}. Split the batch.")

        for info in audio_entries:
            raw = info.filename
            # zip slip: refuse traversal outright rather than silently repairing it
            if raw.startswith("/") or ".." in Path(raw.replace("\\", "/")).parts:
                skipped.append(f"{raw}: rejected unsafe path")
                continue
            # symlink entries: high bits of external_attr carry the unix mode
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                skipped.append(f"{raw}: rejected symlink entry")
                continue
            if info.file_size > MAX_ENTRY_BYTES:
                skipped.append(
                    f"{raw}: {info.file_size // (1024 * 1024)} MB exceeds the "
                    f"{MAX_ENTRY_BYTES // (1024 * 1024)} MB per-file limit")
                continue

            name = safe_name(raw)
            out = dest_dir / name
            n = 1
            while out.exists():                 # collisions after flattening
                out = dest_dir / f"{Path(name).stem}_{n}{Path(name).suffix}"
                n += 1

            # Copy with a running total: a header can lie about file_size, so the
            # decompressed bytes are counted as they are written.
            written = 0
            try:
                with archive.open(info) as src, out.open("wb") as dst:
                    while chunk := src.read(1024 * 1024):
                        written += len(chunk)
                        total += len(chunk)
                        if written > MAX_ENTRY_BYTES or total > MAX_UNCOMPRESSED_BYTES:
                            dst.close()
                            out.unlink(missing_ok=True)
                            raise UploadRejected(
                                "archive expands beyond the uncompressed size limit "
                                f"({MAX_UNCOMPRESSED_BYTES // (1024**3)} GB) - "
                                "refusing to continue")
                        dst.write(chunk)
            except UploadRejected:
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise
            except Exception as exc:
                out.unlink(missing_ok=True)
                skipped.append(f"{raw}: could not extract ({type(exc).__name__})")
                continue
            files.append((out.name, out))

    if not files:
        raise UploadRejected("no usable audio survived extraction: "
                             + "; ".join(skipped[:5]))
    files.sort(key=lambda t: t[0])       # deterministic processing order
    return Extracted(files=files, skipped=skipped)
