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


def print_scan_summary(results_by_D: dict[int, dict], tiling_name: str, schedule: tuple[tuple[float, float], ...]) -> None:
    """Print optimization and energy results for each bond dimension."""
    for D, result in results_by_D.items():
        tau_error_history = result["tau_error_history"]
        print(f"\n\n{tiling_name} | D={D}\n")
        for tau, tolerance in schedule:
            errors = [error for recorded_tau, error in tau_error_history if recorded_tau == tau]
            print(f"tau={tau:.2e}, tolerance={tolerance:.2e}, sweeps: {len(errors)}")
        print(f"total sweeps: {len(tau_error_history)}")
        print(f"energy per PEPS site simple update : {result['energy_per_peps_site_su']:.12f}\n")
        for chi, energy_per_peps_site_ctmrg in zip(result["chi_values"], result["energies_per_peps_site_ctmrg"]):
            print(f"chi={chi}, Energy per PEPS site ctmrg={energy_per_peps_site_ctmrg:.12f}")


def print_run_summary(name: str, diagnostics: dict, verbosity: int = 1) -> None:
    """Display the termination status and final energy of a run."""
    if verbosity < 1:
        return
    messages = {
        "converged": "converged",
        "not_converged": "stopped at the iteration limit without convergence",
        "fixed_iterations": "completed the requested fixed iterations",
    }
    status = messages[diagnostics["status"]]
    energy = diagnostics["energy_per_peps_site"]
    print(f"  {name} {status} | Energy per PEPS site: {energy:.12f}\n", flush=True)



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
