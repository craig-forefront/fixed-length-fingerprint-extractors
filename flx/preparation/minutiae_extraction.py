"""
Minutiae extraction for a very large list of image files: resumable, in
shards, with a log of the failures. Used by scripts/extract_minutiae.py and
scripts/run_minutiae_extractor.py.

This module needs no torch (see flx/preparation/__init__.py). The minutiae
extractor (package `fingerflow` of the minutiae_extractor repository) is
imported when the first extractor is made.

.mnt file format:
    Line 1: <num_minutiae>
    Line 2: <image_width> <image_height>
    Lines 3+: <x> <y> <angle_radians> <class_index> <direction_confidence>
    Last line: # <how the image was prepared>

    Width, height, x and y refer to the prepared image: the image file
    resampled to 500 pixels per inch (see sensor_normalization.py). x is the
    column and y the row (y pointing down). The angle is the direction of the
    minutia, the direction vector is (cos(angle), sin(angle)) in these axes.
    For a ridge ending it points back along the ridge, for a bifurcation into
    the valley between the two branches (as in ISO 19794-2); it is between 0
    and 2 pi.
    The class index is the minutia type of ClassifyNet: 0 ending,
    1 bifurcation, 2 fragment, 3 enclosure, 4 crossbar, 5 other; -1 if the
    minutiae were not classified. DeepPrint does not read it.
    direction_confidence (0 to 1) rates the direction only. It is NOT the
    extractor's score: the score (how sure the extractor is that the point is
    a minutia) is not written. Every minutia that the extractor returns with
    its default settings is in the file, and the minutiae cannot be filtered
    by score afterwards.
    Readers take the number of minutiae from line 1 and ignore what follows
    the minutiae.

    A .mnt file is written under a temporary name and renamed when it is
    complete, so a file with the name .mnt is always a finished one.
"""

import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import numpy as np

from flx.preparation.sensor_normalization import SensorNormalizer

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

SETTINGS_FILE = "extraction_settings.json"

# Class index in the .mnt file of a minutia that was not classified
NOT_CLASSIFIED = -1


def write_mnt_file(
    path: str, minutiae, width: int, height: int, comment: Optional[str] = None
) -> None:
    """
    Writes the minutiae table of the extractor (a pandas DataFrame with the
    columns x, y, angle and optionally class and direction_confidence) as a
    .mnt file. The file appears under its name only when it is complete.
    """
    count = len(minutiae)
    columns = list(minutiae.columns)

    def column(name: str, default: float) -> np.ndarray:
        if name in columns:
            return minutiae[name].to_numpy(dtype=float)
        return np.full(count, default)

    x, y, angle = column("x", 0.0), column("y", 0.0), column("angle", 0.0)
    classes = column("class", np.nan)
    confidence = column("direction_confidence", np.nan)

    lines = [f"{count}", f"{int(width)} {int(height)}"]
    for i in range(count):
        class_index = NOT_CLASSIFIED if np.isnan(classes[i]) else int(classes[i])
        lines.append(
            "%.4f %.4f %.6f %d %.4f" % (x[i], y[i], angle[i], class_index, confidence[i])
        )
    if comment:
        lines.append("# " + comment.replace("\n", " "))

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp{os.getpid()}"
    with open(temporary, "w") as file:
        file.write("\n".join(lines) + "\n")
    os.replace(temporary, path)


# What scripts/extract_minutiae.py needs of the minutiae extractor
EXTRACTOR_REQUIREMENT = (
    "the fingerflow package of the minutiae_extractor repository "
    "(https://github.com/craig-forefront/minutiae_extractor), whose Extractor "
    "has the image_dpi option and accepts None for the ClassifyNet and CoreNet "
    "paths: its main branch, commit 420b6e5 (\"Merge pull request #3\") or "
    "later. Older clones, the upstream repository and the fingerflow release "
    "on PyPI do not work. See docs/installation.md."
)


class ExtractorTooOldError(TypeError):
    """The installed minutiae extractor lacks what the extraction needs."""


def make_extractor(model_paths: dict, image_dpi: float = 500):
    """
    The extractor of the minutiae_extractor repository.

    model_paths: 'coarse_net' and 'fine_net' are needed. 'classify_net' is
    optional: without it (missing or None) the minutiae are not classified,
    which saves one network call per minutia; DeepPrint does not use the
    class. CoreNet is never loaded, because the core points are not written.
    """
    from fingerflow.extractor import Extractor

    try:
        return Extractor(
            model_paths["coarse_net"],
            model_paths["fine_net"],
            model_paths.get("classify_net"),
            None,
            image_dpi=image_dpi,
        )
    except TypeError as error:
        # An Extractor without image_dpi rejects the keyword; one that needs
        # all four networks fails on the path None
        import fingerflow

        location = getattr(fingerflow, "__file__", None) or "location unknown"
        raise ExtractorTooOldError(
            "The Extractor of the installed fingerflow package could not be "
            f"set up ({type(error).__name__}: {error}). Most likely the package "
            f"is too old (it was imported from {location}). The extraction "
            f"needs {EXTRACTOR_REQUIREMENT}"
        ) from error


def extract_to_file(extractor, normalizer: SensorNormalizer, image_path: str, output_path: str):
    """
    Prepares one image (see SensorNormalizer.load), extracts its minutiae and
    writes them to output_path.

    Returns (minutiae table of the extractor, PreparedImageInfo).
    """
    image, info = normalizer.load(image_path, check_polarity=True)
    minutiae = extractor.extract_minutiae(image)["minutiae"]
    # The size of the prepared image: the minutiae coordinates refer to it
    write_mnt_file(output_path, minutiae, info.width, info.height, info.describe())
    return minutiae, info


def iter_image_paths(input_dir: str, extensions: tuple = IMAGE_EXTENSIONS) -> Iterator[str]:
    """
    Yields the paths, relative to input_dir, of all image files below it. The
    order is the same in every call as long as the files do not change:
    directories and files are visited in sorted order. Nothing is collected in
    memory beyond one directory.
    """
    extensions = tuple(e.lower() for e in extensions)
    for directory, subdirectories, files in os.walk(input_dir):
        subdirectories.sort()
        relative_directory = os.path.relpath(directory, input_dir)
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in extensions:
                yield name if relative_directory == "." else os.path.join(relative_directory, name)


def iter_file_list(file_list: str) -> Iterator[str]:
    """
    Yields the image paths of a file list: a text file with one path per line,
    relative to the input directory. Empty lines and lines that start with #
    are left out.
    """
    with open(file_list) as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def select_shard(paths: Iterable[str], num_shards: int, shard_index: int) -> Iterator[str]:
    """
    Yields the paths of one shard: those whose position in the list, counted
    from 0, leaves the remainder shard_index when divided by num_shards. The
    shards 0 .. num_shards - 1 together hold every path exactly once.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be at least 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be from 0 to num_shards - 1 = {num_shards - 1}, got {shard_index}"
        )
    for index, path in enumerate(paths):
        if index % num_shards == shard_index:
            yield path


def shard_suffix(num_shards: int, shard_index: int) -> str:
    """Part of the log file names of a shard; empty without sharding."""
    if num_shards == 1:
        return ""
    return f".shard-{shard_index:05d}-of-{num_shards:05d}"


class SettingsChangedError(RuntimeError):
    """The output directory holds files that were extracted with other settings."""


def check_settings(output_dir: str, settings: dict, allow_change: bool = False) -> None:
    """
    Stores the settings of the extraction in the output directory, or compares
    them with the stored ones. All .mnt files of a directory must come from the
    same settings: a run with other settings raises SettingsChangedError,
    unless allow_change is set, which stores the new settings.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, SETTINGS_FILE)
    text = json.dumps(settings, indent=2, sort_keys=True) + "\n"
    if not allow_change:
        try:
            # Fails if the file exists, also when another shard is just creating it
            with open(path, "x") as file:
                file.write(text)
            return
        except FileExistsError:
            pass
        with open(path) as file:
            stored_text = file.read()
        try:
            stored = json.loads(stored_text)
        except ValueError:
            stored = None
        if stored == settings:
            return
        differences = sorted(
            key
            for key in set(settings) | set(stored or {})
            if (stored or {}).get(key) != settings.get(key)
        )
        raise SettingsChangedError(
            f"{path} was written by a run with other settings (different: "
            f"{', '.join(differences)}). All .mnt files of one directory must be "
            "extracted with the same settings. Use the former settings, a new "
            "output directory, or --allow_settings_change together with a run "
            "without --skip_existing that extracts every file again."
        )
    temporary = f"{path}.tmp{os.getpid()}"
    with open(temporary, "w") as file:
        file.write(text)
    os.replace(temporary, path)


@dataclass
class ExtractionConfig:
    """What a worker process needs to extract the minutiae of image files."""

    model_paths: dict
    normalizer: SensorNormalizer
    image_dpi: float = 500


@dataclass
class ExtractionSummary:
    extracted: int = 0
    skipped: int = 0
    failed: int = 0
    warnings: int = 0
    without_minutiae: int = 0
    first_failures: list = field(default_factory=list)
    # Extracted files per resolution source and value, e.g.
    # {"metadata, 300 ppi": 12}; see PreparedImageInfo.resolution_source
    resolution_sources: dict = field(default_factory=dict)


# The configuration and the extractor of this process. The models are loaded
# once per process, when the first image is extracted.
_config: Optional[ExtractionConfig] = None
_extractor = None


def init_worker(config: ExtractionConfig) -> None:
    global _config, _extractor
    _config = config
    _extractor = None


def _get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = make_extractor(_config.model_paths, _config.image_dpi)
    return _extractor


def extract_one(
    job: tuple[str, str],
) -> tuple[str, bool, Optional[str], Optional[str], int, Optional[str]]:
    """
    Extracts the minutiae of one image in a worker process.

    job: (path of the image, path of the .mnt file)

    Returns (path of the image, success, error message, warning, number of
    minutiae, where the resolution came from and its value). An error of the image (unreadable, resolution unknown, failure
    inside the extractor) is returned, not raised, so that the run goes on.
    An extractor that can not be set up (missing package or model files, a
    version without the options used here) is not an error of the image: that
    exception is raised.
    """
    image_path, output_path = job
    extractor = _get_extractor()
    try:
        minutiae, info = extract_to_file(extractor, _config.normalizer, image_path, output_path)
        return (
            image_path,
            True,
            None,
            info.polarity_warning,
            len(minutiae),
            info.resolution_source,
        )
    except Exception as error:  # one bad file must not stop the run
        return (image_path, False, f"{type(error).__name__}: {error}", None, 0, None)


class _Log:
    """A log file that is written line by line and removed if it stays empty."""

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "w")
        self.lines = 0

    def add(self, image_path: str, message: str) -> None:
        message = " ".join(str(message).split())
        self._file.write(f"{image_path}\t{message}\n")
        self._file.flush()
        self.lines += 1

    def close(self) -> None:
        self._file.close()
        if self.lines == 0:
            os.remove(self.path)


def run_extraction(
    config: ExtractionConfig,
    jobs: Iterable[tuple[str, str]],
    error_log_path: str,
    warning_log_path: str,
    workers: int = 1,
    skip_existing: bool = False,
    progress=None,
) -> ExtractionSummary:
    """
    Extracts the minutiae for all jobs (image path, .mnt path).

    - With skip_existing, jobs whose .mnt file exists are left out; this is how
      an interrupted run is continued.
    - A file that fails is written to the error log at once and the run goes
      on. The log is started anew by every run. With skip_existing it then
      holds all files of these jobs that still have no .mnt file.
    - Warnings (for example an image that looks inverted) go to the warning
      log; such files are extracted.
    - With more than one worker, the jobs are handed to worker processes a few
      at a time, so a list of any length needs little memory. If a worker
      process dies, the jobs that were running are logged as failed and the
      run goes on with new processes.

    An exception from setting up the extractor stops the run.
    """
    summary = ExtractionSummary()
    for path in (error_log_path, warning_log_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    errors, warnings = _Log(error_log_path), _Log(warning_log_path)

    def pending_jobs() -> Iterator[tuple[str, str]]:
        for image_path, output_path in jobs:
            if skip_existing and os.path.exists(output_path):
                summary.skipped += 1
                if progress is not None:
                    progress.update(1)
                continue
            yield (image_path, output_path)

    def record(result) -> None:
        image_path, success, error, warning, num_minutiae, source = result
        if success:
            summary.extracted += 1
            if source is not None:
                summary.resolution_sources[source] = (
                    summary.resolution_sources.get(source, 0) + 1
                )
            if num_minutiae == 0:
                summary.without_minutiae += 1
            if warning:
                summary.warnings += 1
                warnings.add(image_path, warning)
        else:
            summary.failed += 1
            errors.add(image_path, error)
            if len(summary.first_failures) < 10:
                summary.first_failures.append((image_path, error))
        if progress is not None:
            progress.update(1)

    try:
        if workers <= 1:
            init_worker(config)
            for job in pending_jobs():
                record(extract_one(job))
        else:
            _run_in_processes(config, pending_jobs(), workers, record)
    finally:
        errors.close()
        warnings.close()
    return summary


_WORKER_DIED = (
    "a worker process died while this file or another one was being extracted "
    "(out of memory?); run again with --skip_existing to try it again"
)


def _run_in_processes(config, jobs: Iterator[tuple[str, str]], workers: int, record) -> None:
    window = 4 * workers  # jobs that are handed out at one time

    def new_executor() -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=workers, initializer=init_worker, initargs=(config,)
        )

    executor = new_executor()
    running = {}  # future -> job

    def collect(future) -> bool:
        """Records the result of a finished job. False if its process pool is broken."""
        job = running.pop(future)
        try:
            record(future.result())
            return True
        except BrokenProcessPool:
            record((job[0], False, _WORKER_DIED, None, 0, None))
            return False

    def replace_broken_executor() -> ProcessPoolExecutor:
        # A broken pool fails all its jobs at once; none of them is left running
        wait(list(running))
        for future in list(running):
            collect(future)
        executor.shutdown(wait=False, cancel_futures=True)
        return new_executor()

    exhausted = False
    try:
        while True:
            while not exhausted and len(running) < window:
                job = next(jobs, None)
                if job is None:
                    exhausted = True
                    break
                try:
                    future = executor.submit(extract_one, job)
                except BrokenProcessPool:
                    executor = replace_broken_executor()
                    future = executor.submit(extract_one, job)
                running[future] = job
            if not running:
                return
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            # A list, so that every finished job is recorded before the pool is replaced
            if not all([collect(future) for future in done]):
                executor = replace_broken_executor()
    except BaseException:
        # E.g. the extractor could not be set up: do not try the other images
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=True)
