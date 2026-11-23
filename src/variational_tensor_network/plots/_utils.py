"""Shared helpers for displaying and saving figures."""

import pickle
from math import isfinite
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
        pkl_dir = output_path.parent / "pkl"
        pkl_dir.mkdir(parents=True, exist_ok=True)
        with (pkl_dir / f"{output_path.name}.pkl").open("wb") as f:
            pickle.dump(fig, f)
        pdf_path = Path(f"{output_path}.pdf")
        fig.savefig(pdf_path)
        print(f"\nFigure saved under '{pdf_path}'.\n")
    if show:
        plt.show()
    plt.close(fig)


def field_plot_name(model_name, tiling_name, couplings, D_values, curves, *, observable, continuation=False) -> str:
    fields = [h for points in curves for h, y in points if isfinite(y)]
    if not fields:
        raise ValueError("A field plot needs at least one finite point.")
    coupling_tag = "_".join(f"{key}={value:.12g}" for key, value in sorted(couplings.items()))
    D_tag = "-".join(str(D) for D in sorted(D_values))
    mode_tag = "_up-down" if continuation else ""
    return f"{model_name}_{tiling_name}_{coupling_tag}_D={D_tag}_h={min(fields):.12g}-to-{max(fields):.12g}_points={len(fields)}{mode_tag}_{observable}"


# Legacy

# def plot_output_path(script_file: str, filename: str) -> Path:
#     """Build a figure path under the project's outputs directory."""
#     script = Path(script_file).resolve()
#     root = find_project_root(script)
#     relative_script = script.relative_to(root).with_suffix("")
#     if relative_script.parts[0] == "scripts":
#         relative_script = Path(*relative_script.parts[1:])
#     return root / "outputs" / relative_script / filename
