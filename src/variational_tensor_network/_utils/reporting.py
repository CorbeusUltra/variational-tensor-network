"""Format simulation results and display terminal progress."""

import sys
from shutil import get_terminal_size

from ..tilings import Site


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


def print_observables(local_S_z: dict[Site, tuple[float, ...]], m_z_over_m_sat: float, energy_per_spin: float) -> None:
    for site, values_S_z in local_S_z.items():
        print(f"\n  PEPS site : {site}")
        for i, spin_z in enumerate(values_S_z, start=1):
            print(f"    spin {i} | <S_{i}^z> : {spin_z:+.12f}")
    print(f"\n  m_z/m_sat: {m_z_over_m_sat:.12f}\n")
    print(f"\n  Energy per spin: {energy_per_spin:.12f}\n")



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

