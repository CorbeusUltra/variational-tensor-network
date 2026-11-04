"""Apply simple update and CTMRG to a model. One PEPS site can either represents one dimer of two spins-1/2 or a single spin-1/2."""

from time import perf_counter  # noqa: I001

import torch

from variational_tensor_network.models import MODELS
from variational_tensor_network.objects import Config
from variational_tensor_network.plots import plot_energy_by_D_vs_chi, plot_energy_vs_chi, plot_output_path # noqa: F401
from variational_tensor_network.reporting import format_duration, print_scan_summary
from variational_tensor_network.simulation import run_D_scan


def main() -> None:
    verbosity = 1
    model_name = "heisenberg" # "heisenberg", "ffb", ""shastry_sutherland""
    tiling_names = ["checkerboard"] # "checkerboard", "abcd"
    model = MODELS[model_name]
    config = Config(d=model["d"])
    # D_values = [2]
    # D_values = [3]
    D_values = [2, 3]
    # D_values = [2, 3, 4]
    # D_values = [4]
    # chi_values_by_D = {3: [9]}
    # chi_values_by_D = {D: [16 + i for i in range (8)] for D in D_values} # Small grid for quick tests.
    # chi_values_by_D = {D: [D**2 + i for i in range(8)] for D in D_values} # Small grid greater than D**2
    chi_values_by_D = {D: [16 + 2 * D * i for i in range(8)] for D in D_values} # Previous advised range: D + 1 points, with spacing 2D.
    # chi_values_by_D = {D: [16 + 2 * D**2 * i for i in range(8)] for D in D_values} # Previous advised range: D + 1 points, with spacing 2D.
    schedule = ((.1, 1e-5), (.01, 1e-6), (.001, 1e-7)) # less precision required
    # schedule = ((.01, 1e-6), (.001, 1e-7))
    # schedule = ((.001, 1e-7),)
    # schedule = ((.1, 1e-5), (.01, 1e-7), (.001, 1e-8)) # more precision required
    parameters_by_model = {"heisenberg": {"J": 1.0, "h": 0}, "ffb": {"JD": 2.0, "J": 1.0, "h": 0}, "shastry_sutherland": {"JD": 2.0, "J": 1.0, "h": 0}}
    H = model["bond"](**parameters_by_model[model_name])
    start = perf_counter()
    torch.manual_seed(0)
    simulation_time = 0.0
    for tiling_name in tiling_names:
        if verbosity >= 1:
            print(f"\n\n{model['label']} | {tiling_name}\n")
        scan_start = perf_counter()
        results_by_D = run_D_scan(D_values, tiling_name, config, H, chi_values_by_D, schedule, verbosity=verbosity)
        simulation_time += perf_counter() - scan_start
        if verbosity == 0:
            print_scan_summary(results_by_D, tiling_name, schedule)
        for D, result in results_by_D.items():
            # output_path = plot_output_path(__file__, f"{model_name}/{tiling_name}/energy_vs_chi_{tiling_name}_D={D}.pdf")
            plot_energy_vs_chi(result["chi_values"], result["energies_per_peps_site_ctmrg"], D, title=f"{model['label']} — {tiling_name}", show=True) # output_path=output_path
    print(f"\nTotal simulation time : {format_duration(simulation_time)}")
    print(f"Total runtime : {format_duration(perf_counter() - start)}\n")


if __name__ == "__main__":
    main()
