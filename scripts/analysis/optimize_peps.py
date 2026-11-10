"""Run simple update and plot its convergence across a tau schedule."""

from variational_tensor_network.simulation import run_simple_update


def main() -> None:
    model_name = "shastry_sutherland"  # "heisenberg", "ffb", "shastry_sutherland"
    parameters_by_model = {
        "heisenberg":                    {"J": 1.0, "h": 0},  # Expected energy per spin : −0.669441857(7) * J
        "shastry_sutherland": {"JD": 8.0, "J": 1.0, "h": 0},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0.429(6)
        "ffb":                {"JD": 8.0, "J": 1.0, "h": 0},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0,675(2)
    }
    tiling_name = "checkerboard"  # "checkerboard", "abcd", "rectangle_2x3"
    D = 2
    seed = 0

    max_sweeps = 10000
    schedule = ((.1, 1e-5), (.01, 1e-6), (.001, 1e-7)) # schedule = ((.01, 1e-6), (.001, 1e-7)) ; schedule = ((.001, 1e-7),) ; schedule = ((.1, 1e-5), (.01, 1e-7), (.001, 1e-8))

    verbosity = 1
    show_fig = False
    save_fig = False

    run_simple_update(model_name, parameters_by_model, tiling_name, D, seed, schedule, max_sweeps=max_sweeps, verbosity=verbosity, show_fig=show_fig, save_fig=save_fig)


if __name__ == "__main__":
    main()
