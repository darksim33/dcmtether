"""Derived UIDs: synthetic default, org root and source stem (synthetic values only)."""

import re

import pytest

from dcmtether import uids
from dcmtether.errors import DcmtetherError
from dcmtether.uids import UidError

#: ``derived_uid("2.25.1", "seg", "abc", "series")`` as written by dcmtether before org roots.
SYNTHETIC_UID = "2.25.331224168137694729126586261043108153502"
VALID_UID = re.compile(r"(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*")
#: Longest accepted root (39 characters) and one character more.
LONGEST_ROOT = "1." + "9" * 37
TOO_LONG_ROOT = "1." + "9" * 38


def _uid(role: str = "series", content: str = "abc", root: str | None = None) -> str:
    return uids.derived_uid("2.25.1", "seg", content, role, root)


class TestDerivedUid:
    """``derived_uid`` with and without a root."""

    def test_default_unchanged(self):
        """Without a root the UID equals the one earlier versions wrote."""
        assert _uid() == SYNTHETIC_UID

    def test_root_replaces_synthetic_prefix(self):
        """A root takes the place of ``2.25``; the digits stay the same."""
        assert _uid(root="1.2.3") == "1.2.3." + SYNTHETIC_UID.removeprefix("2.25.")

    @pytest.mark.parametrize(
        "root", [None, "1.2.3", LONGEST_ROOT], ids=["synthetic", "short", "longest"]
    )
    def test_valid_and_within_limit(self, root):
        """Every UID is a valid UID of at most 64 characters."""
        uid = _uid(root=root)
        assert VALID_UID.fullmatch(uid), "not a valid UID"
        assert len(uid) <= uids.UID_MAX

    def test_longest_root_keeps_min_digits(self):
        """A 39-character root leaves exactly 24 hash digits."""
        digits = _uid(root=LONGEST_ROOT).removeprefix(LONGEST_ROOT + ".")
        assert len(digits) == uids.MIN_DIGITS
        assert digits[0] != "0"

    @pytest.mark.parametrize("root", [None, "1.2.3"], ids=["synthetic", "root"])
    def test_deterministic(self, root):
        """The same inputs give the same UID."""
        assert _uid(root=root) == _uid(root=root)

    @pytest.mark.parametrize("root", [None, "1.2.3"], ids=["synthetic", "root"])
    def test_role_and_content_change_uid(self, root):
        """Another role or other content gives another UID."""
        base = _uid(root=root)
        assert _uid("instance", root=root) != base
        assert _uid(content="abd", root=root) != base

    def test_invalid_root_rejected(self):
        """A root that skipped ``check_root`` is still refused."""
        with pytest.raises(UidError):
            _uid(root="1..2")


class TestCheckRoot:
    """``check_root`` accepts valid org roots only."""

    @pytest.mark.parametrize("root", ["1", "1.2.3", "0.9", LONGEST_ROOT])
    def test_accepts_valid(self, root):
        """Valid roots are returned unchanged."""
        assert uids.check_root(root) == root

    def test_strips_blanks(self):
        """Blanks around the root are removed."""
        assert uids.check_root(" 1.2.3 ") == "1.2.3"

    @pytest.mark.parametrize(
        "root",
        ["", "1..2", "1.02", "1.2a", "1.2.", ".1.2", "1,2", "-1.2"],
        ids=[
            "empty",
            "empty-component",
            "leading-zero",
            "letter",
            "trailing-dot",
            "leading-dot",
            "comma",
            "minus",
        ],
    )
    def test_rejects_malformed(self, root):
        """Malformed roots raise ``UidError`` without echoing the value."""
        with pytest.raises(UidError) as info:
            uids.check_root(root)
        assert str(info.value) == "UID root must be digits separated by dots, without leading zeros"

    def test_rejects_too_long(self):
        """A 40-character root is refused."""
        with pytest.raises(UidError) as info:
            uids.check_root(TOO_LONG_ROOT)
        assert str(info.value) == "UID root is longer than 39 characters"

    def test_is_dcmtether_error(self):
        """``UidError`` is an expected error (one-line message in CLI and GUI)."""
        assert issubclass(UidError, DcmtetherError)
        assert issubclass(UidError, ValueError)


class TestStem:
    """``stem`` cuts a UID at its last dot."""

    def test_cuts_last_component(self):
        """The last component is dropped."""
        assert uids.stem("1.2.3.4") == "1.2.3"

    @pytest.mark.parametrize(
        "uid",
        ["1", "", "1..2", "1.02.3", "9" * 40 + ".5"],
        ids=["single-component", "empty", "empty-component", "leading-zero", "long-first"],
    )
    def test_unusable_gives_none(self, uid):
        """A malformed UID, or one whose first component alone is too long, gives ``None``."""
        assert uids.stem(uid) is None

    def test_longest_stem_accepted(self):
        """A stem of 39 characters is kept whole."""
        assert uids.stem(LONGEST_ROOT + ".5") == LONGEST_ROOT

    def test_too_long_stem_is_shortened_at_a_dot(self):
        """A stem over 39 characters loses trailing components until it fits."""
        assert uids.stem(TOO_LONG_ROOT + ".5") == "1"
        uid = ".".join(["1", "2"] + ["12345"] * 10)  # 63 characters, stem of 57
        short = uids.stem(uid)
        assert short == ".".join(["1", "2"] + ["12345"] * 6)  # 39 characters
        assert len(short) <= uids.MAX_ROOT and uids.check_root(short) == short
        assert uid.startswith(short + ".")


class TestUidPrefix:
    """``uid_prefix`` picks the mode."""

    def test_synthetic_default(self):
        """Neither option: synthetic, no fallback."""
        assert uids.uid_prefix("1.2.3.4") == (None, False)

    def test_org_root(self):
        """An org root is checked and used as given."""
        assert uids.uid_prefix("1.2.3.4", uid_root=" 1.5 ") == ("1.5", False)

    def test_bad_org_root(self):
        """A malformed org root raises ``UidError``."""
        with pytest.raises(UidError):
            uids.uid_prefix("1.2.3.4", uid_root="1.02")

    def test_source_stem(self):
        """``uid_source`` uses the source series UID up to its last dot."""
        assert uids.uid_prefix("1.2.3.4", uid_source=True) == ("1.2.3", False)

    def test_source_stem_fallback(self):
        """A malformed source UID falls back to synthetic and says so."""
        assert uids.uid_prefix("1.02.3", uid_source=True) == (None, True)
        assert uids.uid_prefix(TOO_LONG_ROOT + ".5", uid_source=True) == ("1", False)

    def test_options_exclusive(self):
        """Org root and source stem cannot be combined."""
        with pytest.raises(UidError, match="cannot be combined"):
            uids.uid_prefix("1.2.3.4", uid_root="1.5", uid_source=True)
