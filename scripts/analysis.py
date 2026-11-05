"""Apply simple update and CTMRG to a model. One PEPS site can either represents one dimer of two spins-1/2 or a single spin-1/2."""

from dataclasses import replace # noqa: I001
from functools import partial
from time import perf_counter

import torch

from variational_tensor_network.models import MODELS
from variational_tensor_network.objects import Config
from variational_tensor_network.plots import plot_energy_vs_chi, plot_output_path
from variational_tensor_network.reporting import format_duration, print_boxed
from variational_tensor_network.simulation import run_simulation
from variational_tensor_network.plots import plot_simple_update

def main() -> None:
    verbosity = 1
    model_name = "heisenberg" # "heisenberg", "ffb", ""shastry_sutherland""
    tiling_names = ["checkerboard"] # "checkerboard", "abcd"
    D_values = [2, 3] # [2] ; [3] ; [4] ; [2, 3] ; [2, 4] ; [3, 4] ; [2, 3, 4]
    n_chi = 6 # 10, 16, ...
    chi_values_by_D = {D: [D**2 + D * i for i in range(n_chi)] for D in D_values} # {D: [16 + 2 * D**2 * i for i in range(n_chi)] for D in D_values}
    schedule = ((.1, 1e-5), (.01, 1e-6), (.001, 1e-7)) # schedule = ((.01, 1e-6), (.001, 1e-7)) ; schedule = ((.001, 1e-7),) ; schedule = ((.1, 1e-5), (.01, 1e-7), (.001, 1e-8))

    model = MODELS[model_name]
    d = model["d"]
    config = Config(d=d)
    parameters_by_model = {
        "heisenberg":                    {"J": 1.0, "h": 0}, # Exact GS energy = -0.669441857（7）(per spin)
        "ffb":                {"JD": 4.0, "J": 1.0, "h": 0}, # Exact GS energy = -3 (per dimer, so -1.5 per spin)
        "shastry_sutherland": {"JD": 4.0, "J": 1.0, "h": 0}, # Exact GS energy = -3 (per dimer, so -1.5 per spin)
        }
    parameters = parameters_by_model[model_name]
    parameters_text = ", ".join(f"{name}={value:g}" for name, value in parameters.items())
    H = model["bond"](**parameters)
    torch.manual_seed(0)
    simulation_time = 0.0
    plots_to_show = []
    for tiling_name in tiling_names:
        for D in D_values:
            config_D = replace(config, D=D)
            chi_values = chi_values_by_D[D]
            heading = f"{model['label']} (d={d}) ({parameters_text}) | {tiling_name} | D={D}"
            if verbosity >= 1:
                print_boxed(heading)
            start_D = perf_counter()
            su_info, ctmrg_info_by_chi = run_simulation(tiling_name, config_D, H, chi_values, schedule, verbosity=verbosity)
            duration_D = perf_counter() - start_D
            simulation_time += duration_D
            if verbosity >= 1:
                print(f"\nSimulation time for D={D}: {format_duration(duration_D)}\n")
            energies = [ctmrg_info_by_chi[chi]["energy_per_peps_site"] for chi in chi_values]
            title = f"su_{model_name}_{tiling_name}_D={D}"
            output_path = plot_output_path(__file__, title + ".pdf")
            plots_to_show.append(partial(plot_simple_update, info=su_info, title=title, show=True, output_path=output_path))
            title = f"ctmrg_{model_name}_{tiling_name}_D={D}_energy_vs_chi"
            output_path = plot_output_path(__file__, title + ".pdf")
            plots_to_show.append(partial(plot_energy_vs_chi, chi_values, energies, D, title=title, show=True, output_path=output_path))
    if verbosity >= 1:
        print(f"\nTotal simulation time : {format_duration(simulation_time)}\n")
    for show_plot in plots_to_show:
        show_plot()


if __name__ == "__main__":
    main()
