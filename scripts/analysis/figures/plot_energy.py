"""Plot magnetization curves from saved CTMRG scan summaries."""

import json
from math import isfinite, nan

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.plots.observables import plot_energy


def main() -> None:
    model_name = "x_shastry_sutherland" # "heisenberg", "shastry_sutherland", "ffb", "x_shastry_sutherland"
    JD = 1.0
    J = JD *.3
    J1 = JD * .35
    J2 = JD * .25
    parameters_by_model = {
        "heisenberg":           {"J": J},  # Expected energy per spin : −0.669441857(7) * J
        "shastry_sutherland":   {"JD": JD, "J": J},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0.429(6)
        "ffb":                  {"JD": JD, "J": J},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0,675(2)
        "x_shastry_sutherland": {"JD": JD, "J1": J1, "J2": J2}, 
    }
    tiling_name = "abcd"
    couplings = parameters_by_model[model_name]
    D_values = (2,)  # Use (2, 3, 4) when those scans are available.
    h_min, h_max = 1.9, 2.7
    show_fig = True
    save_fig = True

    if not (show_fig or save_fig):
        return
    root = find_project_root(__file__)
    curves: dict[int, dict[float, float]] = {D: {} for D in D_values}
    for path in sorted((root / "data" / "ctmrg").glob("*/scan.json")):
        scan = json.loads(path.read_text(encoding="utf-8"))
        if scan.get("sweep_direction") is not None:
            continue
        point = scan["point"]
        D = point["D"]
        params = point["parameters"]
        if point["model"] != model_name or point["tiling"] != tiling_name:
            continue
        scan_couplings = {key: value for key, value in params.items() if key != "h"}
        if D not in curves or scan_couplings != couplings:
            continue
        if couplings is None:
            couplings = scan_couplings
        elif couplings != scan_couplings:
            raise SystemExit(f"Multiple coupling sets for {model_name}: {couplings} and {scan_couplings}.")
        h = params["h"]
        if not(h_min <= h <= h_max):
            continue
        curve = curves[D]
        E = scan.get("energy_per_spin")
        if scan["status"] != "converged" or E is None or not isfinite(E):
            print(f"D={D} | h={h:.10g} | status={scan['status']} | E={E}")
            for attempt in scan["attempts"]:
                print(f"  chi={attempt['chi']} | CTMRG={attempt['ctmrg_status']}")
            print(f"  Summary: {path}")
            curve.setdefault(h, nan)
            continue
        if h in curve and isfinite(curve[h]):
            raise ValueError(f"Multiple converged scans for D={D}, h={h:g} ({path})")
        curve[h] = E
    curves_by_D: dict[int, list[tuple[float, float]]] = {}
    for D, curve in curves.items():
        if not any(isfinite(m) for m in curve.values()):
            print(f"D={D}: no converged scan for these parameters.")
            continue
        curves_by_D[D] = sorted((h, m) for h, m in curve.items() if isfinite(m))
    if not curves_by_D:
        raise SystemExit("No magnetization curve to plot. Check the filters and data/ctmrg/.")

    coupling_labels = [f"{key}={value}" for key, value in sorted(couplings.items())]
    coupling_tag = "_".join(coupling_labels)
    D_tag = "-".join(str(D) for D in D_values)
    name = f"energy_{model_name}_{tiling_name}_{coupling_tag}_D={D_tag}"
    title = f"{model_name} | {tiling_name} | {', '.join(coupling_labels)}"
    output_path = root / "outputs" / "figures" / "energy" / name if save_fig else None
    plot_energy(curves_by_D, title=title, show=show_fig, output_path=output_path)


if __name__ == "__main__":
    main()
