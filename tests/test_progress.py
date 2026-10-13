from click.testing import CliRunner
from synthetic import make_4d, make_series, write_series

from dcmtether.cli import main
from dcmtether.forward import convert_dataset
from dcmtether.manifest import write_manifest
from dcmtether.scanning import scan


class Recorder:
    """A tqdm-like hook that records each loop and yields every item."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, iterable, **kwargs):
        record = dict(kwargs, items=0)
        self.calls.append(record)
        for item in iterable:
            record["items"] += 1
            yield item


def _source(tmp_path):
    write_series(make_series(), tmp_path / "src" / "a")
    write_series(make_4d(n_volumes=2), tmp_path / "src" / "b")
    return tmp_path / "src"


def test_scan_reports_files_and_series(tmp_path):
    source = _source(tmp_path)
    recorder = Recorder()
    result = scan(source, progress=recorder)
    assert [(c["desc"], c["unit"]) for c in recorder.calls] == [
        ("Reading headers", "file"),
        ("Classifying", "series"),
    ]
    files, series = recorder.calls
    assert files["total"] == files["items"] == result.n_files == 15
    assert series["total"] == series["items"] == len(result.series) == 2
    assert [s.status for s in result.series] == [s.status for s in scan(source).series]


def test_convert_reports_only_series_to_convert(tmp_path):
    dataset = tmp_path / "dataset"
    write_manifest(dataset, scan(_source(tmp_path)))
    recorder = Recorder()
    assert all(o.ok for o in convert_dataset(dataset, progress=recorder))
    [call] = recorder.calls
    assert (call["desc"], call["unit"], call["total"], call["items"]) == (
        "Converting",
        "series",
        2,
        2,
    )
    recorder = Recorder()
    assert convert_dataset(dataset, progress=recorder) == []  # nothing left to do
    assert recorder.calls[0]["total"] == 0


def test_no_progress_bar_without_terminal(tmp_path):
    result = CliRunner().invoke(
        main, ["run", str(_source(tmp_path)), "-o", str(tmp_path / "dataset")]
    )
    assert result.exit_code == 0, result.output
    assert "Converted 2 series, 0 failed." in result.output
    assert "Converting" not in result.output and "it/s" not in result.output
