"""Command-line interface for vlmscope."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from vlmscope.__about__ import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlmscope",
        description="Evaluate vision-language models on VQA, captioning and retrieval.",
    )
    parser.add_argument(
        "--version", action="version", version=f"vlmscope {__version__}"
    )
    parser.set_defaults(func=None)
    parser.add_subparsers(dest="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
