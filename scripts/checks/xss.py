"""Check which spin pairs carry J1 in the extended Shastry–Sutherland model."""

from torch import allclose

from variational_tensor_network.models import heisenberg_term, model_bond_hamiltonians
from variational_tensor_network.tilings import TILINGS


def main() -> None:
    # Set all other terms to zero so the matrix contains only the J1 links.
    parameters = {"JD": 0.0, "J1": 1.0, "J2": 0.0, "h": 0.0}
    pairs_spin_J1 = ((2, 3), (2, 4))
    pairs_spin_J2 = ((1, 3), (2, 3))
    reference_spin_J1 = heisenberg_term(2, 3, n=4) + heisenberg_term(2, 4, n=4)
    reference_spin_J2 = heisenberg_term(1, 3, n=4) + heisenberg_term(2, 3, n=4)
    # Expected pairs from the lattice drawing, by starting site and direction.
    expected_pairs = {
        (0, 0): {"r": pairs_spin_J1, "d": pairs_spin_J2},
        (1, 0): {"r": pairs_spin_J2, "d": pairs_spin_J1},
        (0, 1): {"r": pairs_spin_J2, "d": pairs_spin_J1},
        (1, 1): {"r": pairs_spin_J1, "d": pairs_spin_J2},
    }
    failures = 0
    for tiling in ("checkerboard", "abcd"):
        sites, _ = TILINGS[tiling]
        hamiltonians = model_bond_hamiltonians("xss", parameters, sites)
        print(f"\n{tiling}")
        for site in sites:
            for direction, expected in expected_pairs[site].items():
                H = hamiltonians[(site, direction)]
                if allclose(H, reference_spin_J1):
                    actual = pairs_spin_J1
                elif allclose(H, reference_spin_J2):
                    actual = pairs_spin_J2
                else:
                    actual = None
                passed = (actual == expected)
                status = "OK" if passed else "FAIL"
                found = actual if actual is not None else "unrecognized matrix"
                print( f"  {site} {direction} | J1 found: {found} | expected: {expected} | {status}" )
                failures += int(not passed)
    if failures:
        raise SystemExit(f"\n{failures} bond check(s) failed.")
    print("\nAll J1 bond checks passed.")


if __name__ == "__main__":
    main()
