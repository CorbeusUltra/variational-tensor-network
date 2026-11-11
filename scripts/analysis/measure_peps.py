"""Measure a saved PEPS state at one CTMRG bond dimension."""

from variational_tensor_network.simulation import run_ctmrg


def main() -> None:
    model_name = "ffb"  # "heisenberg", "shastry_sutherland", "ffb"
    JD = 1.0
    J = JD/4
    parameters_by_model = {
        "heisenberg":                   {"J": J, "h": 0.0},  # Expected energy per spin : −0.669441857(7) * J
        "shastry_sutherland": {"JD": JD, "J": J, "h": 1.6},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0.429(6)
        "ffb":                {"JD": JD, "J": J, "h": 2.5},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0,675(2)
    }
    tiling_name = "abcd"  # "checkerboard", "abcd", "rectangle_2x3"
    D = 2
    seed = 0

    n_chi = 10
    chi_values = [16 + D * i for i in range(n_chi)] # chi_values_by_D = {D: [D**2 + i for i in range(n_chi)] for D in D_values} ; chi_values_by_D = {D: [D**2 + D * i for i in range(n_chi)] for D in D_values} ; chi_values_by_D = {D: [16 + 2 * D**2 * i for i in range(n_chi)] for D in D_values} ; chi_values_by_D = {D: [16 + D * i for i in range(n_chi)] for D in D_values}
    chi_values = [chi_values[-1]]
    convergence_chi = chi_values[-1]  # chi_values[-1] | None
    conv_tol = 1e-8
    max_cycles = 2000

    verbosity = 1
    show_fig = False
    save_fig = False

    run_ctmrg(model_name, parameters_by_model, tiling_name, D, seed, chi_values, max_cycles=max_cycles, conv_tol=conv_tol, verbosity=verbosity, convergence_chi=convergence_chi, show_fig=show_fig, save_fig=save_fig)


if __name__ == "__main__":
    main()
