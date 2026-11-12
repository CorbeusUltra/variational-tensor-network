"""Run simple update and plot its convergence across a tau schedule."""

from variational_tensor_network.simulation import run_simple_update


def main() -> None:
    model_name = "ffb"  # "heisenberg", "ffb", "shastry_sutherland"
    JD = 1.0
    J = JD/4
    h = 5
    parameters_by_model = {
        "heisenberg":                   {"J": J, "h": h},  # Expected energy per spin : −0.669441857(7) * J
        "shastry_sutherland": {"JD": JD, "J": J, "h": h},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0.429(6)
        "ffb":                {"JD": JD, "J": J, "h": h},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0,675(2)
    }
    tiling_name = "abcd"  # "checkerboard", "abcd", "rectangle_2x3"
    D = 2
    seed = 0

    max_sweeps = 10000
    schedule = ((.1, 1e-4), (.01, 1e-6), (.001, 1e-8)) # schedule = ((.01, 1e-6), (.001, 1e-7)) ; schedule = ((.001, 1e-7),) ; schedule = ((.1, 1e-5), (.01, 1e-7), (.001, 1e-8))

    verbosity = 2
    show_fig = True
    save_fig = True

    run_simple_update(model_name, parameters_by_model, tiling_name, D, seed, schedule, max_sweeps=max_sweeps, verbosity=verbosity, show_fig=show_fig, save_fig=save_fig)


if __name__ == "__main__":
    main()
