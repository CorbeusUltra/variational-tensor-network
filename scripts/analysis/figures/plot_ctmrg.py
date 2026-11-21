"""Plot CTMRG diagnostics from a saved scan."""

import torch

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.plots.ctmrg import plot_ctmrg_convergence


def main() -> None:
    root = find_project_root(__file__)
    scan_id = "d1a5f6bd94012508a4c48f0365ec4b2848392fef343f89d0e074fed736cc806a"
    result_path = root / "data" / "ctmrg" / scan_id / "chi=20.pt"
    show_fig = True
    save_fig = False
    by_site = True

    if not (show_fig or save_fig):
        return
    if not result_path.is_file():
        print(f"Missing CTMRG result: {result_path}")
        return

    result = torch.load(result_path, map_location="cpu", weights_only=True, mmap=True)
    output_dir = root / "outputs" / "figures" / "ctmrg" / result_path.parent.name
    plot_ctmrg_convergence(result, by_site=by_site, show=show_fig, output_path=output_dir / f"cycles_{result_path.stem}" if save_fig else None)


if __name__ == "__main__":
    main()
