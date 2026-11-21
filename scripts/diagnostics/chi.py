"""Run a chi convergence scan for one PEPS state."""

from variational_tensor_network.simulation.chi_scan import run_chi_scan


def main() -> None:
    model_name = "x_shastry_sutherland" # "heisenberg", "shastry_sutherland", "ffb", "x_shastry_sutherland"
    JD = 1.0
    J = JD *.25
    J1 = JD * .35
    J2 = JD * .25
    h = 1.0
    parameters_by_model = {
        "heisenberg":           {"J": J, "h": h},  # Expected energy per spin : −0.669441857(7) * J
        "shastry_sutherland":   {"JD": JD, "J": J, "h": h},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0.429(6)
        "ffb":                  {"JD": JD, "J": J, "h": h},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0,675(2)
        "x_shastry_sutherland": {"JD": JD, "J1": J1, "J2": J2, "h": h}, 
    }
    tiling_name = "abcd"  # "checkerboard", "abcd", "rectangle_2x3"
    D = 2
    seed = 0

    conv_tol_chi = 1e-6
    max_chi_iterations = 20
    required_stable_chi_iteration = 3
    conv_tol_ctmrg = 1e-6
    max_ctmrg_cycles = 5000

    verbosity = 2
    show_fig = False
    save_fig = False

    run_chi_scan(model_name, parameters_by_model, tiling_name, D, seed, max_chi_iterations=max_chi_iterations , conv_tol_chi=conv_tol_chi, required_stable_chi_iteration=required_stable_chi_iteration, max_ctmrg_cycles=max_ctmrg_cycles, conv_tol_ctmrg=conv_tol_ctmrg, verbosity=verbosity, show_fig=show_fig, save_fig=save_fig)


if __name__ == "__main__":
    main()
