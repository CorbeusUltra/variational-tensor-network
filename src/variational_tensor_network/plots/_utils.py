"""Shared helpers for displaying and saving figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def finish_plot(fig: Figure, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"\nFigure saved under '{output_path}'.\n")
    if show:
        plt.show()
    plt.close(fig)


def plot_output_path(script_file: str, filename: str) -> Path:
    """Build a figure path under the project's outputs directory."""
    script = Path(script_file).resolve()
    for root in script.parents:
        if (root / "pyproject.toml").is_file():
            relative_script = script.relative_to(root).with_suffix("")
            if relative_script.parts[0] == "scripts":
                relative_script = Path(*relative_script.parts[1:])
            return root / "outputs" / relative_script / filename
    raise ValueError("Cannot find a project root containing 'pyproject.toml'.")
