from os import environ
from pathlib import Path


def find_project_root(script_file: str | Path) -> Path:
    """Find the nearest parent directory containing pyproject.toml."""
    script = Path(script_file).resolve()
    for root in script.parents:
        if (root / "pyproject.toml").is_file():
            return root
    raise ValueError("Cannot find a project root containing 'pyproject.toml'.")


def campaign_start_D() -> int:
    start_D = int(environ.get("VTN_START_D", "2"))
    if start_D not in (2, 3, 4):
        raise ValueError("VTN_START_D must be 2, 3, or 4.")
    return start_D


def campaign_data_dir(project_root: Path) -> Path:
    data_dir = project_root / "data"
    start_D = campaign_start_D()
    return data_dir if start_D == 2 else data_dir / "campaigns" / f"from-D{start_D}"
