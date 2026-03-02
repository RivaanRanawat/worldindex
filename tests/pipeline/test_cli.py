from __future__ import annotations

import pytest

from pipeline.cli import build_parser


@pytest.mark.parametrize(
    "command",
    ["run", "extract", "compress", "build-index", "serve", "status", "validate"],
)
def test_subcommand_help_exits_cleanly(command: str) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([command, "--help"])

    assert exc_info.value.code == 0
