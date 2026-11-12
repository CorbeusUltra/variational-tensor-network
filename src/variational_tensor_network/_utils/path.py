from pathlib import Path


def find_project_root(script_file: str | Path) -> Path:
    """Find the nearest parent directory containing pyproject.toml."""
    script = Path(script_file).resolve()
    for root in script.parents:
        if (root / "pyproject.toml").is_file():
            return root
    raise ValueError("Cannot find a project root containing 'pyproject.toml'.")
