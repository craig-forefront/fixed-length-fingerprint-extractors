"""
flx/preparation/sample_table.py where pyarrow is not installed: the module is
imported, what needs no Parquet file works, and reading or writing a table
says how to install pyarrow. The loaders of flx/data/sample_table_loader.py
are imported and used without pyarrow as well (that test needs PyTorch).

No test of this file needs pyarrow and none is skipped without it, because
the environment without pyarrow is the one these tests are about. Where
pyarrow is installed, its absence is simulated: the name is blocked in
sys.modules, which makes every later import of it fail.
"""

import dataclasses
import json
import subprocess
import sys
import tomllib
from os.path import dirname, join

import numpy as np
import pytest

from flx.preparation import sample_table
from flx.preparation.sample_table import (
    SampleRow,
    SampleTableError,
    normalize_image_path,
    read_samples,
    samples_checksum,
    samples_from_rows,
    validate_samples,
    write_part,
)

REPOSITORY = dirname(dirname(__file__))

# First lines of a child interpreter in which these packages cannot be imported
BLOCKED = (
    "import sys\n"
    "for name in ('torch', 'torchvision', 'pandas', 'pyarrow'):\n"
    "    sys.modules[name] = None\n"
)


def run_blocked(code: str, given: str = "") -> subprocess.CompletedProcess:
    """
    Runs the code in a new interpreter, in the repository, with the packages
    blocked. `given` is what the code reads from its standard input.
    """
    return subprocess.run(
        [sys.executable, "-c", BLOCKED + code],
        cwd=REPOSITORY,
        input=given,
        capture_output=True,
        text=True,
    )


def rows() -> list:
    """Three valid rows: a 1000 ppi file with minutiae, one without minutiae, one not known."""
    common = dict(ppi_origin="manifest", ridges="dark", ridges_origin="argument")
    return [
        SampleRow(
            subject=16,
            impression=2,
            image_path="sensor_a/17_3.png",
            file_size=51_234,
            ppi=1000.0,
            source_width=826,
            source_height=1274,
            width=413,
            height=637,
            minutiae=np.array([[175.0, 320.5, 1.25], [12.0, 600.0, -0.5]], dtype=np.float32),
            **common,
        ),
        SampleRow(
            subject=3,
            impression=1,
            image_path="sensor_b/4_2.png",
            file_size=40_000,
            ppi=500.0,
            source_width=400,
            source_height=500,
            width=400,
            height=500,
            minutiae=np.zeros((0, 3), dtype=np.float32),
            **common,
        ),
        SampleRow(
            subject=3,
            impression=0,
            image_path="sensor_b/4_1.png",
            file_size=40_100,
            ppi=500.0,
            source_width=400,
            source_height=500,
            width=400,
            height=500,
            minutiae=None,
            **common,
        ),
    ]


def test_the_sample_table_module_does_not_need_torch_pandas_or_pyarrow():
    """The preparation runs in environments that have none of them."""
    result = run_blocked("import flx.preparation.sample_table\n")
    assert result.returncode == 0, result.stderr
    # Control: an import of a blocked package does fail in such an interpreter
    for package in ("pyarrow", "pyarrow.parquet", "pandas", "torch"):
        result = run_blocked(f"import {package}\n")
        assert result.returncode != 0 and "ModuleNotFoundError" in result.stderr, package


def test_what_needs_no_parquet_file_works_without_pyarrow():
    """Rows in memory are made, validated and summed, and paths normalised."""
    code = (
        "import json\n"
        "import numpy as np\n"
        "from flx.preparation.sample_table import (\n"
        "    SampleRow, SampleTableError, normalize_image_path, samples_checksum,\n"
        "    samples_from_rows, validate_samples,\n"
        ")\n"
        "rows = []\n"
        "for values in json.load(sys.stdin):\n"
        "    if values['minutiae'] is not None:\n"
        "        values['minutiae'] = np.array(values['minutiae'], np.float32).reshape(-1, 3)\n"
        "    rows.append(SampleRow(**values))\n"
        "samples = samples_from_rows(rows)\n"
        "validate_samples(samples)\n"
        "print(samples_checksum(samples))\n"
        "print(samples.find(16, 2), samples.image_path(0), len(samples.minutiae_of(2)))\n"
        "print(normalize_image_path('./sensor_a//17_3.png'))\n"
        "try:\n"
        "    validate_samples(samples_from_rows(rows + rows[:1]))\n"
        "except SampleTableError as error:\n"
        "    print(str(error)[:31])\n"
        "print(sys.modules['pyarrow'])\n"
    )
    given = []
    for row in rows():
        minutiae = None if row.minutiae is None else row.minutiae.tolist()
        given.append(dict(dataclasses.asdict(row), minutiae=minutiae))
    result = run_blocked(code, json.dumps(given))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        samples_checksum(samples_from_rows(rows())),
        "2 sensor_b/4_1.png 2",
        "sensor_a/17_3.png",
        "The same subject and impression",
        "None",  # pyarrow was blocked to the end
    ]


def test_reading_and_writing_a_table_without_pyarrow_say_how_to_install_it(tmp_path, monkeypatch):
    with open(join(REPOSITORY, "pyproject.toml"), "rb") as file:
        (requirement,) = tomllib.load(file)["project"]["optional-dependencies"]["table"]
    samples = samples_from_rows(rows())
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    for use in (
        lambda: read_samples(str(tmp_path)),
        lambda: write_part(str(tmp_path / "table"), 0, 1, samples, {"id": "a"}),
    ):
        with pytest.raises(SampleTableError) as error:
            use()
        message = str(error.value)
        assert "needs the package pyarrow" in message
        assert 'pip install -e ".[table]"' in message
        # With the quotes: without them a shell takes > as a redirection
        assert f'pip install "{requirement}"' in message
    assert not (tmp_path / "table").exists()
    # What needs no Parquet file is not touched by the missing package
    validate_samples(samples)
    assert normalize_image_path("sensor_a/../sensor_b/4_1.png") == samples.image_path(0)


def test_the_loaders_of_a_table_do_not_need_pyarrow():
    """
    flx/data/sample_table_loader.py serves a table that is in memory without
    pyarrow; only reading one from its files needs it. The loaders need
    torch, like everything in flx/data, so only pyarrow is blocked here, and
    this one test is skipped where torch is not installed, as in the
    environment of the minutiae extraction.
    """
    pytest.importorskip("torch")
    code = (
        "import sys\n"
        "sys.modules['pyarrow'] = None\n"
        "import json\n"
        "import numpy as np\n"
        "from flx.data.dataset import Identifier\n"
        "from flx.data.sample_table_loader import (\n"
        "    SampleTable, SampleTableImageLoader, SampleTableMinutiaMapLoader,\n"
        ")\n"
        "from flx.preparation.sample_table import SampleRow, SampleTableError, samples_from_rows\n"
        "rows = []\n"
        "for values in json.load(sys.stdin):\n"
        "    if values['minutiae'] is not None:\n"
        "        values['minutiae'] = np.array(values['minutiae'], np.float32).reshape(-1, 3)\n"
        "    rows.append(SampleRow(**values))\n"
        "table = SampleTable(samples_from_rows(rows), 'images')\n"
        "images, minutiae = SampleTableImageLoader(table), SampleTableMinutiaMapLoader(table)\n"
        "print(table.num_rows, table.subjects.tolist(), table.row(Identifier(16, 2)))\n"
        "print(table.checksum)\n"
        "locations, angles = minutiae.get_minutiae(Identifier(16, 2))\n"
        "print(locations.shape, angles.tolist(), minutiae.get_frame_size(Identifier(16, 2)))\n"
        "print([minutiae.get(Identifier(s, i))[1] for s, i in ((3, 0), (3, 1), (16, 2))])\n"
        "try:\n"
        "    images.get(Identifier(3, 1))\n"
        "except FileNotFoundError as error:\n"
        "    print(str(error)[:14], 'sensor_b/4_2.png' in str(error))\n"
        "try:\n"
        "    SampleTable.read('table', 'images')\n"
        "except SampleTableError as error:\n"
        "    print('needs the package pyarrow' in str(error))\n"
        "print(sys.modules['pyarrow'])\n"
    )
    given = []
    for row in rows():
        minutiae = None if row.minutiae is None else row.minutiae.tolist()
        given.append(dict(dataclasses.asdict(row), minutiae=minutiae))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY,
        input=json.dumps(given),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "3 [3, 16] 2",
        samples_checksum(samples_from_rows(rows())),
        "(2, 2) [1.25, -0.5] (413, 637)",
        "[0.0, 0.0, 1.0]",  # minutiae not known, an image without minutiae, two minutiae
        "The image file True",
        "True",
        "None",  # pyarrow was blocked to the end
    ]


def test_the_requirement_of_pyarrow_is_the_same_wherever_it_is_stated():
    """pyproject.toml, requirements.txt and the message that says how to install it."""
    with open(join(REPOSITORY, "pyproject.toml"), "rb") as file:
        project = tomllib.load(file)["project"]
    assert project["optional-dependencies"]["table"] == [sample_table.PYARROW_REQUIREMENT]
    # A sample table is an option: the package itself does not depend on pyarrow
    assert not any("pyarrow" in requirement for requirement in project["dependencies"])
    with open(join(REPOSITORY, "requirements.txt")) as file:
        assert sample_table.PYARROW_REQUIREMENT in file.read().splitlines()


def test_the_requirement_of_pyarrow_does_not_go_below_17():
    """
    With pyarrow 16.1 some damaged parts end the interpreter with a
    segmentation fault inside of pyarrow when their rows are read, so that
    read_samples cannot refuse them. pyarrow 17.0 raises an error for the
    same files. A requirement that allows 16.1 would allow that again, and
    the other tests would not notice it: they pass with 16.1.
    """
    name, _, version = sample_table.PYARROW_REQUIREMENT.partition(">=")
    assert name == "pyarrow"
    assert int(version.split(".")[0]) >= 17
