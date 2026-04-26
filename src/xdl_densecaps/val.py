"""Validate the configured model checkpoint."""

from __future__ import annotations

from typing import Sequence

from xdl_densecaps.evaluation import run_evaluation_script


def main(argv: Sequence[str] | None = None) -> int:
    return run_evaluation_script(
        argv,
        split_name="val",
        description="Validate the configured normal/lesion model checkpoint.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
