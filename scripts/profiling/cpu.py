"""Profile the execution time of the analysis main function."""

import cProfile
import runpy
import subprocess
import sys

from variational_tensor_network.plots._utils import find_project_root


def main() -> None:
    label = "before_opt_einsum"
    root = find_project_root(__file__)
    script_path = root / "scripts" / "analysis.py"
    output_dir = root / "outputs" / "profiling"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"cpu_{label}.prof"
    analysis = runpy.run_path(str(script_path))
    print(f"\nProfiling CPU: {label}\n", flush=True)
    with cProfile.Profile() as profiler:
        analysis["main"]()
    profiler.dump_stats(str(data_path))
    print(f"\nCPU profile saved under '{data_path}'.\n", flush=True)
    try:
        subprocess.run([sys.executable, "-m", "snakeviz", str(data_path)], check=True)
    except KeyboardInterrupt:
        print("\nSnakeViz closed.")


if __name__ == "__main__":
    main()
