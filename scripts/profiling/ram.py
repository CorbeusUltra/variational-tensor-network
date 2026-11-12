"""Profile memory allocations during the analysis main function."""

import runpy
import subprocess
import sys

import memray

from variational_tensor_network._utils.path import find_project_root


def main() -> None:
    root = find_project_root(__file__)
    script_path = root / "scripts" / "analysis" / "measure_peps.py" # optimize_peps | measure_peps
    output_dir = root / "outputs" / "profiling"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "ram.bin"
    report_path = output_dir / "ram.html"
    analysis = runpy.run_path(str(script_path))
    print("\nProfiling memory\n", flush=True)
    destination = memray.FileDestination(data_path, overwrite=True)
    with memray.Tracker(destination=destination):
        analysis["main"]()
    subprocess.run([sys.executable, "-m", "memray", "flamegraph", "--force", "-o", str(report_path), str(data_path)], check=True)
    print(f"\nMemory report saved under '{report_path}'.\n")
    subprocess.run(["open" , report_path], check=True)


if __name__ == "__main__":
    main()
