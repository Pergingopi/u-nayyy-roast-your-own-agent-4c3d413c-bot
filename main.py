"""Roast Your Own Agent — make.fun entry.

Writes a single one-line roast of our own agent to
``$MAKEFUN_OUTPUT_DIR/entry.txt`` (default ``/output``).

Gate: UTF-8 text, 1 to 200 characters, not only whitespace, one line.
The entry text is written with NO trailing newline.

Design notes
------------
* Validation is split into small, single-purpose helpers so each gate
  rule can be tested and reasoned about on its own.  Every failure is
  signalled with an explicit ``raise`` — this module never uses
  ``assert`` for input or environment checks, so checks survive
  ``python -O``.
* The output directory comes from ``MAKEFUN_OUTPUT_DIR``; a missing,
  empty, or whitespace-only value falls back to ``/output`` instead of
  crashing on an empty path.
* The write is atomic: a uniquely-named temp file in the same
  directory, fsynced, then ``os.replace()`` onto the destination.
  Stale temp files left behind by earlier crashed runs are swept up
  (best effort, with a retry) before the new write starts, so a
  half-finished run can never leave clutter that a later run trips
  over — but the sweep never touches the live ``entry.txt`` or any
  file that is not one of our own ``.entry-*`` temp files.
* After the rename, the file is read back and compared byte for byte,
  and its mode is checked to be exactly ``0644``.
"""

import os
import stat
import tempfile
import time

ENTRY = (
    "Our agent entered 71 bounties, won $0, and still asks permission "
    "before roasting itself \u2014 even its insults need a preview "
    "and a confirmation."
)

DEFAULT_OUTPUT_DIR = "/output"
TEMP_PREFIX = ".entry-"
# Temp files older than this are treated as stale leftovers from a
# crashed run and swept up; younger ones may belong to a live run.
STALE_AFTER_SECONDS = 3600
MAX_LENGTH = 200


# ---------------------------------------------------------------------------
# Validation — one helper per gate rule.
# ---------------------------------------------------------------------------

def _validate_type(text):
    """Reject anything that is not a ``str`` outright."""
    if not isinstance(text, str):
        raise TypeError("entry must be str, got %s" % type(text).__name__)


def _validate_not_blank(text):
    """Reject empty or whitespace-only entries."""
    if not text.strip():
        raise ValueError("entry must not be only whitespace")


def _validate_length(text):
    """Enforce the 1–200 character gate (characters, not bytes)."""
    if len(text) < 1 or len(text) > MAX_LENGTH:
        raise ValueError(
            "entry must be 1 to %d characters, got %d"
            % (MAX_LENGTH, len(text))
        )


def _validate_single_line(text):
    """Reject embedded newlines or carriage returns."""
    if "\n" in text or "\r" in text:
        raise ValueError("entry must be a single line")


def _validate_no_control_chars(text):
    """Reject C0 control characters and DEL."""
    for ch in text:
        code = ord(ch)
        if code < 0x20 or code == 0x7F:
            raise ValueError(
                "entry contains a control character U+%04X" % code
            )


def _validate_utf8_encodable(text):
    """Round-trip through UTF-8 to guarantee valid encodable text."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("entry is not valid UTF-8 text: %s" % exc)


def _validate(text):
    """Run every gate rule, in a fixed order, raising on the first."""
    _validate_type(text)
    _validate_not_blank(text)
    _validate_length(text)
    _validate_single_line(text)
    _validate_no_control_chars(text)
    _validate_utf8_encodable(text)


# ---------------------------------------------------------------------------
# Environment and filesystem helpers.
# ---------------------------------------------------------------------------

def _resolve_output_dir():
    """Return the output directory.

    ``MAKEFUN_OUTPUT_DIR`` wins when it is set to a non-blank value;
    a missing, empty, or whitespace-only value falls back to the
    default instead of producing an empty or invalid path.
    """
    raw = os.environ.get("MAKEFUN_OUTPUT_DIR")
    if raw is not None and raw.strip():
        return raw.strip()
    return DEFAULT_OUTPUT_DIR


def _fsync_dir(path):
    """Durably persist a directory entry (best effort on odd filesystems)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _is_stale_temp(name, path, now):
    """True only for our own temp files old enough to be crash leftovers."""
    if not name.startswith(TEMP_PREFIX):
        return False
    try:
        age = now - os.stat(path).st_mtime
    except OSError:
        # Vanished (or unreadable) between listing and stat: not ours
        # to worry about any more.
        return False
    return age > STALE_AFTER_SECONDS


def _cleanup_stale_temps(out_dir, attempts=3):
    """Remove stale ``.entry-*`` temp files, retrying transient errors.

    Best effort only: any failure is swallowed, because a leftover
    temp file must never stop a fresh run from writing its entry.
    The live ``entry.txt`` and every non-temp file are never touched.
    """
    try:
        names = os.listdir(out_dir)
    except OSError:
        return
    now = time.time()
    for name in names:
        path = os.path.join(out_dir, name)
        if not _is_stale_temp(name, path, now):
            continue
        for attempt in range(attempts):
            try:
                os.unlink(path)
                break
            except FileNotFoundError:
                break
            except OSError:
                if attempt == attempts - 1:
                    break
                time.sleep(0.05 * (attempt + 1))


def _write_atomic(target, data):
    """Write *data* to *target* atomically with mode 0644.

    A uniquely-named temp file is created in the destination
    directory (``mkstemp`` uses ``O_EXCL``), written, fsynced, and
    renamed over the target with ``os.replace()``.  On any failure
    only our own temp file is removed; nothing else is touched.
    """
    out_dir = os.path.dirname(target)
    fd, tmp_path = tempfile.mkstemp(prefix=TEMP_PREFIX, dir=out_dir)
    try:
        try:
            os.fchmod(fd, 0o644)
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        # Chmod again after the write in case the platform reset bits.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, target)
        # Final mode enforcement on the destination itself.
        os.chmod(target, 0o644)
        _fsync_dir(out_dir)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _verify_output(target, data):
    """Read the written file back; it must be byte-identical, mode 0644."""
    with open(target, "rb") as f:
        written = f.read()
    if written != data:
        raise IOError("read-back verification failed: entry.txt mismatch")
    mode = stat.S_IMODE(os.stat(target).st_mode)
    if mode != 0o644:
        raise IOError("entry.txt mode is not 0644: %s" % oct(mode))


def main():
    _validate(ENTRY)
    data = ENTRY.encode("utf-8")

    out_dir = _resolve_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    _cleanup_stale_temps(out_dir)
    target = os.path.join(out_dir, "entry.txt")
    _write_atomic(target, data)
    _verify_output(target, data)


if __name__ == "__main__":
    main()
