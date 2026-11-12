"""Measure a saved PEPS state at one CTMRG bond dimension."""

from variational_tensor_network.simulation import run_ctmrg


def main() -> None:
    model_name = "ffb"  # "heisenberg", "shastry_sutherland", "ffb"
    JD = 1.0
    J = JD/4
    h = .9
    parameters_by_model = {
        "heisenberg":                   {"J": J, "h": h},  # Expected energy per spin : −0.669441857(7) * J
        "shastry_sutherland": {"JD": JD, "J": J, "h": h},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0.429(6)
        "ffb":                {"JD": JD, "J": J, "h": h},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0,675(2)
    }
    tiling_name = "abcd"  # "checkerboard", "abcd", "rectangle_2x3"
    D = 2
    seed = 0

    conv_tol_chi = 1e-6
    max_chi_iterations = 10
    conv_tol_ctmrg = 1e-7
    max_ctmrg_cycles = 2000

    verbosity = 2
    show_fig = False
    save_fig = False
    convergence_last_chi = False

    run_ctmrg(model_name, parameters_by_model, tiling_name, D, seed, max_chi_iterations=max_chi_iterations , conv_tol_chi=conv_tol_chi, max_ctmrg_cycles=max_ctmrg_cycles, conv_tol_ctmrg=conv_tol_ctmrg, verbosity=verbosity, show_fig=show_fig, save_fig=save_fig, convergence_last_chi=convergence_last_chi)



if __name__ == "__main__":
    main()
