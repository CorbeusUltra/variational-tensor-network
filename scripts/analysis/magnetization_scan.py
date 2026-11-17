import os
import sys
from pprint import pp
from typing import Any


def run_magnetization_point(D: int = 2, h: float = 3) -> dict[str, Any]:
    from variational_tensor_network.simulation.chi_scan import run_chi_scan
    model_name = "x_shastry_sutherland"  # "heisenberg", "shastry_sutherland", "ffb"
    JD = 1.0
    J = JD *.3
    J1 = JD * .35
    J2 = JD * .25
    parameters_by_model = {
        "heisenberg":           {"J": J, "h": h},  # Expected energy per spin : −0.669441857(7) * J
        "shastry_sutherland":   {"JD": JD, "J": J, "h": h},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0.429(6)
        "ffb":                  {"JD": JD, "J": J, "h": h},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0,675(2)
        "x_shastry_sutherland": {"JD": JD, "J1": J1, "J2": J2, "h": h}, 
    }
    tiling_name = "abcd"  # "checkerboard", "abcd", "rectangle_2x3"
    seed = 0

    conv_tol_chi = 1e-6
    max_chi_iterations = 50
    required_stable_chi_iteration = 3
    conv_tol_ctmrg = 1e-7
    max_ctmrg_cycles = 2000

    verbosity = 2
    show_fig = False
    save_fig = False

    return run_chi_scan(model_name, parameters_by_model, tiling_name, D, seed, max_chi_iterations=max_chi_iterations , conv_tol_chi=conv_tol_chi, required_stable_chi_iteration=required_stable_chi_iteration, max_ctmrg_cycles=max_ctmrg_cycles, conv_tol_ctmrg=conv_tol_ctmrg, verbosity=verbosity, show_fig=show_fig, save_fig=save_fig)



def linspace(start: float, stop: float, n: int):
    if n == 1:
        return [start]
    return [round(start + i * (stop - start) / (n - 1),10) for i in range(n)]


def main() -> None:
    D_VALUES = [2,3]
    start, stop, n = 0, 4, 5
    points = [(D, h) for D in D_VALUES for h in linspace(start, stop, n)]
    if not points or len(points) != len(set(points)):
        raise ValueError("The scan grid must contain at least one point and no duplicates.")
    if sys.argv[1:] == ["--points"]:
        pp(points)
        return
    if sys.argv[1:] == ["--count"]:
        print(len(points))
        return
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
    run_magnetization_point(D, h)


if __name__ == "__main__":
    main()
