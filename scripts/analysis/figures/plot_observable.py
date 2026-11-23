"""Plot magnetization curves from saved CTMRG scan summaries."""

import json  # noqa: I001
import sys
from math import isfinite, nan

from variational_tensor_network._utils.path import campaign_data_dir, find_project_root
from variational_tensor_network.plots._utils import field_plot_name
from variational_tensor_network.plots.observables import plot_observable, plot_field_sweep


def main() -> None:
    OBSERVABLES = {
        "energy": ("energy_per_spin", r"$E/N_{\mathrm{spins}}$"),
        "magnetization": ("m_z_over_m_sat", r"$m_z/m_{\mathrm{sat}}$"),
    }
    selected_observables = ["energy", "magnetization"] # "energy", "magnetization"
    for observable in selected_observables:
        value_key, ylabel = OBSERVABLES[observable]
        model_name = "x_shastry_sutherland"  # "heisenberg", "shastry_sutherland", "ffb", "x_shastry_sutherland"
        tiling_name = "abcd"
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
        couplings = parameters_by_model[model_name]
        continuation = (sys.argv[1:] == ["--continuation"])
        D_values = (2,) if continuation else (2,3,4,5,6,7)
        h_min, h_max = 1.9, 2.7
        show_fig = True
        save_fig = True

        if not (show_fig or save_fig):
            return
        root = find_project_root(__file__)
        curves: dict[int, dict[float, float]] = {D: {} for D in D_values}
        curves_by_direction: dict[str, dict[float, float]] = {"up": {}, "down": {}}
        # for path in sorted((root / "data" / "ctmrg").glob("*/scan.json")):
        for path in sorted((campaign_data_dir(root) / "ctmrg").glob("*/scan.json")):
            scan = json.loads(path.read_text(encoding="utf-8"))
            direction = scan.get("sweep_direction")
            if continuation:
                if direction not in curves_by_direction:
                    continue
            elif direction is not None:
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
                raise SystemExit( f"Multiple coupling sets for {model_name}: {couplings} and {scan_couplings}.")
            h = params["h"]
            if not(h_min <= h <= h_max):
                continue
            curve = curves_by_direction[direction] if continuation else curves[D]
            value = scan.get(value_key)
            if scan["status"] != "converged" or value is None or not isfinite(value):
                print(f"D={D} | h={h:.10g} | status={scan['status']} | {value_key}={value}")
                for attempt in scan["attempts"]:
                    print(f"  chi={attempt['chi']} | CTMRG={attempt['ctmrg_status']}")
                print(f"  Summary: {path}")
                curve.setdefault(h, nan)
                continue
            if h in curve and isfinite(curve[h]):
                raise ValueError(f"Multiple converged scans for D={D}, h={h:g} ({path})")
            curve[h] = value
        if continuation:
            sweep_curves = {direction: sorted(curve.items(), reverse=(direction == "down")) for direction, curve in curves_by_direction.items()}
            if not any(isfinite(m) for points in sweep_curves.values() for _, m in points):
                raise SystemExit("No CTMRG measure converged for the sweep at D=2.")
            name = field_plot_name(model_name, tiling_name, couplings, D_values, sweep_curves.values(),observable=observable, continuation=True)
            output_path = (root / "outputs" / "figures" / "magnetization" / name if save_fig else None)
            plot_field_sweep(sweep_curves, ylabel=ylabel, show=show_fig, output_path=output_path)
            continue
        curves_by_D: dict[int, list[tuple[float, float]]] = {}
        for D, curve in curves.items():
            if not any(isfinite(m) for m in curve.values()):
                print(f"D={D}: no converged scan for these parameters.")
                continue
            curves_by_D[D] = sorted((h, m) for h, m in curve.items() if isfinite(m))
        if not curves_by_D:
            raise SystemExit("No magnetization curve to plot. Check the filters and data/ctmrg/.")
        coupling_labels = [f"{key}={value}" for key, value in sorted(couplings.items())]
        name = field_plot_name(model_name, tiling_name, couplings, curves_by_D.keys(), curves_by_D.values(), observable=observable)
        title = f"{model_name} | {tiling_name} | {', '.join(coupling_labels)}"
        output_path = (root / "outputs" / "figures" / observable / name if save_fig else None)
        plot_observable(curves_by_D, ylabel=ylabel, title=title,show=show_fig, output_path=output_path)


if __name__ == "__main__":
    main()
