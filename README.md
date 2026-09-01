# ames-price

## Development setup

```shell
uv sync
uv run pre-commit install --install-hooks \
  --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
```

## Verify setup

```shell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```
