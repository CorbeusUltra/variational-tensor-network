"""Shared helpers for displaying and saving figures."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def finish_plot(fig: Figure, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if output_path is not None:
        fig.canvas.manager.set_window_title(output_path.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pkl_dir = output_path.parent / ".pkl"
        pkl_dir.mkdir(parents=True, exist_ok=True)
        with (pkl_dir / f"{output_path.name}.pkl").open("wb") as f:
            pickle.dump(fig, f)
        pdf_path = Path(f"{output_path}.pdf")
        fig.savefig(pdf_path)
        print(f"\nFigure saved under '{pdf_path}'.\n")
    if show:
        plt.show()
    plt.close(fig)


# Legacy

# def plot_output_path(script_file: str, filename: str) -> Path:
#     """Build a figure path under the project's outputs directory."""
#     script = Path(script_file).resolve()
#     root = find_project_root(script)
#     relative_script = script.relative_to(root).with_suffix("")
#     if relative_script.parts[0] == "scripts":
#         relative_script = Path(*relative_script.parts[1:])
#     return root / "outputs" / relative_script / filename
