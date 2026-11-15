import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "flx.image_processing",
        "flx.image_processing.slap_segmentation",
        "flx.image_processing.augmentation",
        "flx.data",
    ],
)
def test_import_in_new_interpreter(module: str):
    # flx.data and flx.image_processing import each other. Each of them must
    # be importable first.
    subprocess.run([sys.executable, "-c", f"import {module}"], check=True)
