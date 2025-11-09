# Development

## Setup

Install PDM as described in [README.md](README.md).
Then install with `--dev` dependencies rather than just `--prod`.

```shell
pdm install --dev
```

## Lint

This project uses `ruff` and `pytype` for linting.
To run them

```bash
pdm run lint
```

**Note:** The `pytype` linter requires Python development headers to be installed on your system. If you get an error like `fatal error: Python.h: No such file or directory`, you will need to install the appropriate package for your operating system (e.g., `python3-dev` on Debian/Ubuntu, `python3-devel` on Fedora/CentOS). If you are unable to install the development headers, you can still run the `ruff` linter separately:

```bash
.venv/bin/pdm run ruff check
```
