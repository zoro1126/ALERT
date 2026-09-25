"""
dataset.py
==========
UTA Real-Life Drowsiness Dataset (UTA-RLDD) — video scanner and label parser.

Dataset layout (on disk)
------------------------
<root>/
  FoldN_partM/
    FoldN_partM/          ← inner duplicate dir (quirk of the dataset)
      <subject_id>/       ← e.g. "01", "02", …
        0.<ext>           → label 0  (Alert)
        5.<ext>           → label 1  (Low Vigilant)
        10.<ext>          → label 2  (Drowsy)

Extensions observed: .mov .MOV .mp4   (case-insensitive match required)

Usage
-----
from training.dataset import scan_dataset, fold_split

records  = scan_dataset()          # list[VideoRecord]
train, val = fold_split(records, val_folds=[4])
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from utils.config import Config

# ---------------------------------------------------------------------------
# Label mapping: filename stem → integer class index
# ---------------------------------------------------------------------------

# Config.LABEL_MAP == {"0": 0, "5": 1, "10": 2} — stems map directly to labels
_STEM_TO_LABEL: dict[str, int] = dict(Config.LABEL_MAP)  # {"0": 0, "5": 1, "10": 2}

_VALID_EXTENSIONS: set[str] = {".mov", ".mp4"}   # matched case-insensitively

# Regex that pulls fold number and part number from a directory name like
# "Fold3_part2"
_FOLD_RE = re.compile(r"fold(\d+)_part(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoRecord:
    """One labelled video file from the UTA-RLDD dataset.

    Attributes
    ----------
    path : Path
        Absolute path to the video file.
    label : int
        0 = Alert, 1 = Low Vigilant, 2 = Drowsy
    fold : int
        Fold number (1–4) extracted from the parent directory name.
    subject : str
        Subject ID string (e.g. "01", "27").
    """
    path: Path
    label: int
    fold: int
    subject: str

    @property
    def label_name(self) -> str:
        """Human-readable label string."""
        return {0: "alert", 1: "low_vigilant", 2: "drowsy"}[self.label]


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_dataset(root: Path | str | None = None) -> list[VideoRecord]:
    """
    Walk the UTA-RLDD directory tree and return every labelled video file.

    Parameters
    ----------
    root : Path or str, optional
        Dataset root directory.  Defaults to ``Config.DATASET_ROOT``.

    Returns
    -------
    list[VideoRecord]
        One entry per valid video file found, sorted by (fold, subject, label).

    Raises
    ------
    FileNotFoundError
        If ``root`` does not exist on disk.
    """
    root = Path(root) if root is not None else Path(Config.DATASET_ROOT)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            "Check Config.DATASET_ROOT or pass an explicit path."
        )

    records: list[VideoRecord] = []
    skipped: list[str] = []

    # Depth layout:  root / FoldN_partM / FoldN_partM / <subject> / <video>
    # We iterate subject dirs by looking for directories whose name is numeric.
    for fold_outer in sorted(root.iterdir()):
        if not fold_outer.is_dir():
            continue

        m = _FOLD_RE.match(fold_outer.name)
        if m is None:
            continue                          # skip non-fold dirs
        fold_num = int(m.group(1))

        # Handle the inner duplicate directory (quirk of UTA-RLDD packaging)
        fold_inner = fold_outer / fold_outer.name
        search_root = fold_inner if fold_inner.is_dir() else fold_outer

        for subject_dir in sorted(search_root.iterdir()):
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            for video_file in sorted(subject_dir.iterdir()):
                if not video_file.is_file():
                    continue
                if video_file.suffix.lower() not in _VALID_EXTENSIONS:
                    continue

                stem = video_file.stem          # "0", "5", or "10"
                if stem not in _STEM_TO_LABEL:
                    skipped.append(str(video_file))
                    continue

                records.append(VideoRecord(
                    path=video_file.resolve(),
                    label=_STEM_TO_LABEL[stem],
                    fold=fold_num,
                    subject=subject_id,
                ))

    if skipped:
        import warnings
        warnings.warn(
            f"scan_dataset: skipped {len(skipped)} unrecognised files.\n"
            + "\n".join(f"  {s}" for s in skipped[:5])
            + ("\n  …" if len(skipped) > 5 else ""),
            stacklevel=2,
        )

    return sorted(records, key=lambda r: (r.fold, r.subject, r.label))


# ---------------------------------------------------------------------------
# Train / validation split by fold
# ---------------------------------------------------------------------------

def fold_split(
    records: Sequence[VideoRecord],
    val_folds: Sequence[int],
) -> tuple[list[VideoRecord], list[VideoRecord]]:
    """
    Split records into train and validation sets by fold number.

    Parameters
    ----------
    records : sequence of VideoRecord
    val_folds : sequence of int
        Fold numbers to use as validation (e.g. ``[4]``).

    Returns
    -------
    (train_records, val_records) : tuple[list, list]
    """
    val_set = set(val_folds)
    train = [r for r in records if r.fold not in val_set]
    val   = [r for r in records if r.fold     in val_set]
    return train, val


# ---------------------------------------------------------------------------
# Quick summary helper (useful for debugging)
# ---------------------------------------------------------------------------

def dataset_summary(records: Sequence[VideoRecord]) -> str:
    """Return a human-readable summary string for a list of VideoRecords."""
    from collections import Counter
    label_counts = Counter(r.label_name for r in records)
    fold_counts  = Counter(r.fold for r in records)
    lines = [
        f"Total videos : {len(records)}",
        f"By label     : {dict(label_counts)}",
        f"By fold      : {dict(sorted(fold_counts.items()))}",
        f"Subjects     : {len({r.subject for r in records})}",
    ]
    return "\n".join(lines)
