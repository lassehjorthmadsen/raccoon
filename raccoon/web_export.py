"""Where the raccoon-website checkout is, and how not to guess wrong about it.

The browser engine lives in a separate repo (raccoonbg.com). Its position
relative to this one is a per-machine fact, not a property of either project:
siblings on the Linux dev box, opposite sides of the WSL boundary on the Windows
ones. So the export scripts default to the sibling layout — the common case —
and let ``$RACCOON_WEBSITE`` override it.

The important part is :func:`ensure_out_dir`. Both exporters used to
``mkdir(parents=True)`` their output path, which meant a default that was wrong
on this machine did not fail: it manufactured a plausible-looking empty tree
somewhere harmless, reported success, and left the site shipping the previous
weights with nothing anywhere to indicate the export had missed. Creating the
leaf directory is fine; creating its whole ancestry is the part that hides the
mistake.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Environment variable holding the raccoon-website checkout root.
WEBSITE_ENV = "RACCOON_WEBSITE"

#: Fallback when the variable is unset: the sibling layout.
DEFAULT_WEBSITE_ROOT = "../raccoon-website"


def website_path(subpath: str) -> str:
    """Default location of ``subpath`` inside the raccoon-website checkout.

    Returned as a string so argparse can show it verbatim in ``--help``.
    """
    root = os.environ.get(WEBSITE_ENV) or DEFAULT_WEBSITE_ROOT
    return str(Path(root) / subpath)


def ensure_out_dir(out_dir: str | Path) -> Path:
    """Return ``out_dir``, creating the leaf but never inventing its ancestry.

    Raises ``SystemExit`` when the parent is missing — which is what a wrong
    default, a stale checkout, or a typo'd ``--out-dir`` all look like from
    here.
    """
    path = Path(out_dir)
    if not path.parent.is_dir():
        raise SystemExit(
            f"--out-dir parent does not exist: {path.parent.resolve()}\n"
            f"The raccoon-website checkout is not where this expected. Pass "
            f"--out-dir explicitly, or set ${WEBSITE_ENV} to its root "
            f"(default assumes the sibling layout, {DEFAULT_WEBSITE_ROOT})."
        )
    path.mkdir(exist_ok=True)
    return path
