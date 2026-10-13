"""The user documentation stays in sync with the code."""

import inspect
import re
from pathlib import Path

import pytest

import dcmtether
from dcmtether import log, pmap, seg
from dcmtether.cli import main

DOCS = Path(__file__).resolve().parent.parent / "docs"
README = DOCS.parent / "README.md"


def test_every_command_is_documented():
    usage = (DOCS / "usage.md").read_text("utf-8")
    for command in main.commands:
        assert f"### `{command} " in usage or f"`{command} " in usage, command


@pytest.mark.parametrize(
    ("name", "parser"),
    [("seg-template.json", seg.load_metadata), ("pmap-template.json", pmap.load_metadata)],
)
def test_example_templates_are_valid(name, parser):
    parser(DOCS / "examples" / name)


def test_documented_log_fields_exist():
    formats = (DOCS / "formats.md").read_text("utf-8")
    section = formats.split("## Log", 1)[1]
    fields = section.split("Fields", 1)[1].split("Events:", 1)[0]  # after the file names
    documented = re.findall(r"`([a-z_]+)`", fields)
    assert len(documented) > 10
    for field in documented:
        assert field in log.ALLOWED_FIELDS, field


def test_api_page_documents_every_public_name():
    text = (DOCS / "api.md").read_text("utf-8")
    documented = set(re.findall(r"`([A-Za-z_]+)[`(]", text))
    assert set(dcmtether.__all__) <= documented, set(dcmtether.__all__) - documented
    # and the signatures shown are the real ones
    for name, shown in re.findall(r"^### `(\w+)\((.*?)\)`", text, flags=re.M):
        parameters = list(inspect.signature(getattr(dcmtether, name)).parameters)
        public = [p for p in parameters if not p.startswith("_")]
        assert re.findall(r"(\w+)(?:=[^,]*)?(?:,|$)", shown.replace("*, ", "")) == public, name


def test_readme_links_exist():
    text = README.read_text("utf-8")
    for target in re.findall(r"\]\(([^)#]+)\)", text):
        if not target.startswith("http"):
            assert (README.parent / target).exists(), target
