"""Plot magnetization curves from saved CTMRG scan summaries."""

import json
from math import isfinite, nan

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.plots.magnetization import plot_magnetization


def main() -> None:
    model_name = "ffb"
    tiling_name = "abcd"
    JD = 1.0
    J = 0.3 * JD
    D_values = (2,3)  # Use (2, 3, 4) when those scans are available.
    h_min, h_max = 0, 3
    show_fig = True
    save_fig = True

    if not (show_fig or save_fig):
        return
    root = find_project_root(__file__)
    curves: dict[int, dict[float, float]] = {D: {} for D in D_values}
    for path in sorted((root / "data" / "ctmrg").glob("*/scan.json")):
        scan = json.loads(path.read_text(encoding="utf-8"))
        point = scan["point"]
        D = point["D"]
        params = point["parameters"]
        if point["model"] != model_name or point["tiling"] != tiling_name:
            continue
        if D not in curves or params["JD"] != JD or params["J"] != J:
            continue
        h = params["h"]
        if not(h_min <= h <= h_max):
            continue
        curve = curves[D]
        m = scan["m_z_over_m_sat"]
        if scan["status"] != "converged" or m is None or not isfinite(m):
            curve.setdefault(h, nan)
            continue
        if h in curve and isfinite(curve[h]):
            raise ValueError(f"Multiple converged scans for D={D}, h={h:g} ({path})")
        curve[h] = m
    curves_by_D: dict[int, list[tuple[float, float]]] = {}
    for D, curve in curves.items():
        if not any(isfinite(m) for m in curve.values()):
            print(f"D={D}: no converged scan for these parameters.")
            continue
        curves_by_D[D] = sorted(curve.items())
    if not curves_by_D:
        raise SystemExit("No magnetization curve to plot. Check the filters and data/ctmrg/.")
    D_tag = "-".join(str(D) for D in D_values)
    name = f"magnetization_{model_name}_{tiling_name}_JD={JD:g}_J={J:g}_D={D_tag}"
    output_path = root / "outputs" / "figures" / "analysis" / "magnetization" / name if save_fig else None
    plot_magnetization(curves_by_D, title=f"{model_name} | {tiling_name} | $J/J_D={J / JD:g}$", show=show_fig, output_path=output_path)


if __name__ == "__main__":
    main()
