import os
import sys
from pathlib import Path
from pprint import pp
from typing import Any, Literal

from variational_tensor_network._utils.path import campaign_start_D


def run_magnetization_point(D: int = 2, h: float = 3, *, source_path: Path | None = None, sweep_direction: Literal["up", "down"] | None = None) -> dict[str, Any]:
    from variational_tensor_network.simulation.chi_scan import run_chi_scan
    model_name = "x_shastry_sutherland"  # "heisenberg", "shastry_sutherland", "ffb", "x_shastry_sutherland"
    JD = 1.0
    J = JD *.25
    J1 = JD * .35
    J2 = JD * .25
    parameters_by_model = {
        "heisenberg":           {"J": J, "h": h},  # Expected energy per spin : −0.669441857(7) * J
        "shastry_sutherland":   {"JD": JD, "J": J, "h": h},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0.429(6)
        "ffb":                  {"JD": JD, "J": J, "h": h},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0,675(2)
        "x_shastry_sutherland": {"JD": JD, "J1": J1, "J2": J2, "h": h}, 
    }
    tiling_name = "abcd"  # "checkerboard", "abcd", "rectangle_2x3"
    seed = 2 if campaign_start_D() == 4 else 1

    conv_tol_chi = 1e-6
    max_chi_iterations = 20
    required_stable_chi_iteration = 2
    conv_tol_ctmrg = 1e-7
    max_ctmrg_cycles = 2000

    verbosity = 2
    show_fig = False
    save_fig = False

    initialization = None
    if os.environ.get("VTN_CAMPAIGN") == "D3-lowest-energy":
        initialization = "lower_D_svd"
    elif sweep_direction is not None and source_path is not None:
        initialization = "nearby_h"
    su_state_path = None
    if initialization is not None or sweep_direction is not None:
        from variational_tensor_network.simulation.simple_update import is_usable_su_state, run_simple_update  # noqa: I001
        su_state_path, state = run_simple_update(model_name, parameters_by_model, tiling_name, D, seed, initialization=initialization, source_path=source_path, sweep_direction=sweep_direction, verbosity=verbosity, show_fig=False, save_fig=False)
        if not is_usable_su_state(state):
            return {"status": "su_failed", "su_state_path": su_state_path}
    result = run_chi_scan(model_name, parameters_by_model, tiling_name, D, seed, max_chi_iterations=max_chi_iterations , conv_tol_chi=conv_tol_chi, required_stable_chi_iteration=required_stable_chi_iteration, max_ctmrg_cycles=max_ctmrg_cycles, conv_tol_ctmrg=conv_tol_ctmrg, su_state_path=su_state_path, verbosity=verbosity, show_fig=show_fig, save_fig=save_fig)
    result["su_state_path"] = su_state_path
    return result


def run_field_sweep(fields: list[float], direction: Literal["up", "down"]) -> None:
    source_path = None
    for h in fields:
        print(f"D=2 | {direction} | h={h:g}", flush=True)
        result = run_magnetization_point(D=2, h=h, source_path=source_path, sweep_direction=direction)
        if result["status"] == "su_failed":
            print(f"Branche {direction} arrêtée à h={h:g}", flush=True)
            break
        source_path = result["su_state_path"]


def linspace(start: float, stop: float, n: int):
    if n == 1:
        return [start]
    return [round(start + i * (stop - start) / (n - 1),10) for i in range(n)]


def arange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    if start >= stop:
        raise ValueError(f"must have start < stop, got start={start} >= stop={stop}")
    values = []
    h = start
    while round(h, 10) < stop:
        values.append(round(h, 10))
        h += step
    return values


def main() -> None:
    start_D = campaign_start_D()
    selected_campaign = (os.environ.get("VTN_CAMPAIGN") == "D3-lowest-energy")
    first_D = 4 if selected_campaign else start_D
    D_VALUES = range(first_D, 8)
    start, stop, step = 1.9, 2.7, .025
    h_VALUES = arange(start, stop, step)
    extra_h = arange(2.100, 2.185, 0.005)
    h_VALUES = sorted(set(h_VALUES + extra_h))
    points = [(D, h) for D in D_VALUES for h in h_VALUES]
    if sys.argv[1:] == ["--continuation"]:
        if start_D != 2:
            raise ValueError("--continuation requires VTN_START_D=2.")
        run_field_sweep(h_VALUES, "up")
        run_field_sweep(h_VALUES[::-1], "down")
        return
    if not points or len(points) != len(set(points)):
        raise ValueError("The scan grid must contain at least one point and no duplicates.")
    if sys.argv[1:] == ["--points"]:
        pp(points)
        return
    if sys.argv[1:] == ["--count"]:
        print(len(points))
        return
    if sys.argv[1:] == ["--array-groups"]:
        groups: dict[int, list[str]] = {}
        for index, (D, _) in enumerate(points):
            groups.setdefault(D, []).append(str(index))
        for D, indices in groups.items():
            print(D, ",".join(indices))
        return
    print(f"\n{len(h_VALUES)} values of h from {start} to {stop}, step = {step}\n")
    slurm_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_index is None:
        for D, h in points:
            print(f"Local | D={D} | h={h}", flush=True)
            run_magnetization_point(D, h)
        return
    task_id = int(slurm_index)
    if not 0 <= task_id < len(points):
        raise ValueError(f"Task index {task_id} is outside the scan grid.")
    D, h = points[task_id]
    print(f"Slurm task {task_id} | D={D} | h={h}", flush=True)
    result = run_magnetization_point(D, h)
    if selected_campaign and result["status"] == "su_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
