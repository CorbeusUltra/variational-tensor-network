"""Format simulation results and display terminal progress."""

import sys
from shutil import get_terminal_size


def format_duration(duration: float) -> str:
    """Format a duration in seconds as days, hours, minutes, and seconds."""
    minutes, seconds = divmod(int(duration), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    text = f"{days}d " if days else ""
    text += f"{hours}h " if hours else ""
    text += f"{minutes}m " if minutes else ""
    text += f"{seconds}s" if seconds else ""
    return text.strip() or "<1s"


def print_progress(message: str, verbosity: int = 1, *, final: bool = False) -> None:
    """Print an update, replacing the current terminal line at verbosity 1."""
    if verbosity < 1:
        return
    if verbosity == 1 and sys.stdout.isatty():
        if not final:
            width = get_terminal_size((120, 24)).columns
            message = message[:max(width - 1, 1)]
        print(f"\r\033[2K{message}", end="\n" if final else "", flush=True)
    else:
        print(message, flush=True)


def print_simulation_summary(su_info: dict, ctmrg_info_by_chi: dict) -> None:
    """Print convergence and energy results for one SU and CTMRG simulation."""
    for stage_index, (tau, tolerance) in enumerate(su_info["schedule"]):
        if stage_index >= len(su_info["stages"]):
            print(f"  tau={tau:.2e}, tol={tolerance:.2e} | not executed")
            continue
        stage = su_info["stages"][stage_index]
        print(f"  tau={tau:.2e}, tol={tolerance:.2e} | sweeps={stage['sweeps']} | final error={stage['final_error']:.2e} | status={stage['status']} | time={format_duration(stage['duration'])}")
    print(f"\n  Total sweeps: {su_info['total_sweeps']} | time={format_duration(su_info['duration'])}")
    print_run_summary("Simple update", su_info)
    for chi, info in ctmrg_info_by_chi.items():
        print(f"  chi={chi} | cycles={info['total_cycles']} | final error={info['final_error']:.2e} | tol={info['tolerance']:.2e} | time={format_duration(info['duration'])}")
        print_run_summary("CTMRG", info)


def print_run_summary(name: str, diagnostics: dict, verbosity: int = 1) -> None:
    """Display the termination status of a run."""
    if verbosity < 1:
        return
    messages = {
        "converged": "converged",
        "not_converged": "stopped at the iteration limit without convergence",
        "fixed_iterations": "completed the requested fixed iterations",
    }
    status = messages[diagnostics["status"]]
    print(f"  {name} {status}", flush=True)


def print_boxed(message: str = ""):
    """Display the message in a box for emphasis."""
    dashes = "—"*len(message)
    print()
    print("┌—"+dashes +"—┐")
    print("│ "+message+" │")
    print("└—"+dashes +"—┘")
    print()


# Legacy code

# def compare_tilings(tiling_names: list[str], config: Config, chi_values: list[int], schedule: tuple[tuple[float, float], ...]) -> dict[str, dict]:
#     """Collect Heisenberg results for several tilings at fixed bond dimension."""
#     results_by_tiling = {}
#     for tiling_name in tiling_names:
#         history, energy_su, energies = run_heisenberg(tiling_name, config, chi_values, schedule)
#         results_by_tiling[tiling_name] = {"chi_values": chi_values, "history": history, "energy_su": energy_su, "energies": energies}
#     return results_by_tiling

# def format_scientific(value: float, decimals: int = 2) -> str:
#     """Better display of float values in terminal."""
#     mantissa, exponent = f"{value:.{decimals}e}".split("e")
#     superscript = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
#     power = f"10{superscript}"
#     return power if float(mantissa) == 1 else f"{mantissa} × {power}"
