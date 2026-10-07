import os
import subprocess
import sys

import pytest

from dcmtether import __version__
from dcmtether.cli import EXIT_NOT_IMPLEMENTED, main

COMMANDS = ["scan", "to-nii", "to-dcm", "compare"]


def test_help_lists_all_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in COMMANDS:
        assert command in out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_no_command_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["scan", "src", "out"],
        ["to-nii", "out"],
        ["to-dcm", "x.nii.gz", "out"],
        ["compare", "a", "b"],
    ],
)
def test_stub_commands_report_not_implemented(argv, capsys):
    assert main(argv) == EXIT_NOT_IMPLEMENTED
    assert "not implemented" in capsys.readouterr().err


def test_help_works_on_a_legacy_console_encoding():
    # Windows consoles may default to cp1252; help text must not crash there.
    result = subprocess.run(
        [sys.executable, "-m", "dcmtether", "--help"],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
