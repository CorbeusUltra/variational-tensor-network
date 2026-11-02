"""Apply simple update and CTMRG to a model. One PEPS site can either represents one dimer of two spins-1/2 or a single spin-1/2."""

from time import perf_counter  # noqa: I001

import torch

from variational_tensor_network.models import MODELS
from variational_tensor_network.objects import Config
from variational_tensor_network.plots import plot_energy_by_D_vs_chi, plot_energy_vs_chi, plot_output_path # noqa: F401
from variational_tensor_network.simulation import run_D_scan, format_duration, print_scan_summary


def main() -> None:
    model_name = "shastry_sutherland" # "heisenberg", "ffb", ""shastry_sutherland""
    tiling_names = ["checkerboard"] # "checkerboard", "abcd"
    model = MODELS[model_name]
    config = Config(d=model["d"], D=3, chi=5)
    D = config.D
    # D_values = [2]
    D_values = [2, 3]
    # D_values = [2, 3, 4]
    chi_values_by_D = {D: [i + 1 for i in range(16)] for D in D_values} # Independent of D, useful for checkerboard curves. Small chi can be unreliable for abcd.
    # chi_values_by_D = {D: [4, 6, 8] for D in D_values} # Small grid for quick tests.
    # chi_values_by_D = {D: [16 + 2 * D * i for i in range(D + 1)] for D in D_values} # Previous advised range: D + 1 points, with spacing 2D.
    schedule = ((.1, 1e-5), (.01, 1e-6), (.001, 1e-7)) # less precision required
    # schedule = ((.01, 1e-6), (.001, 1e-10))
    # schedule = ((.1, 1e-5), (.01, 1e-7), (.001, 1e-8)) # more precision required
    # parameters_by_model = {"heisenberg": {"J": 1.0}, "ffb": {"JD": 2.0, "J": 1.0, "h": 0.0}}
    H = model["bond"]()
    start = perf_counter()
    torch.manual_seed(0)
    for tiling_name in tiling_names:
        print(f"\n\n{model['label']} | {tiling_name}\n")
        results_by_D = run_D_scan(D_values, tiling_name, config, H, chi_values_by_D, schedule)
        print_scan_summary(results_by_D, tiling_name, schedule)
        for D, result in results_by_D.items():
            output_path = plot_output_path(__file__, f"{model_name}/{tiling_name}/energy_vs_chi_{tiling_name}_D={D}.pdf")
            plot_energy_vs_chi(result["chi_values"], result["energies_per_peps_site_ctmrg"], D, title=f"{model['label']} — {tiling_name}", show=True, output_path=output_path)
        # energies_by_D = {D: (result["chi_values"], result["energies_per_peps_site_ctmrg"]) for D, result in results_by_D.items()}
        # output_path = plot_output_path(__file__, f"energy_by_D_vs_chi_{tiling_name}.pdf")
        # plot_energy_by_D_vs_chi(energies_by_D, title=f"FFB — {tiling_name}",show=True, output_path=output_path)

    print(f"\nTotal runtime: {format_duration(perf_counter() - start)}\n")


if __name__ == "__main__":
    main()
