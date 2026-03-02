from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from api.config import ServingConfig
from api.server import create_app


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: poetry run python scripts/serve.py <config.yaml>")

    config = ServingConfig.from_yaml(Path(args[0]))
    uvicorn.run(
        create_app(config=config),
        host=config.host,
        port=config.port,
    )


if __name__ == "__main__":
    main()
