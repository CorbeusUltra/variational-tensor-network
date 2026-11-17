"""Plot Simple Update diagnostics from a saved state."""

import torch  # noqa: I001

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.plots.simple_update import plot_simple_update_diagnostics 


def main() -> None:
    root = find_project_root(__file__)
    state_path = root / "data" / "simple_update" / "1c2122a573811250a67ab495953f7aa25c1cc5d140a45d6e2ce297e9641181a0.pt"
    show_fig = True
    save_fig = False

    if not (show_fig or save_fig):
        return
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    output_dir = root / "outputs" / "figures" / "analysis" / "simple_update" if save_fig else None
    plot_simple_update_diagnostics(state["info_su"], state["metadata"], show=show_fig, output_dir=output_dir)


if __name__ == "__main__":
    main()
