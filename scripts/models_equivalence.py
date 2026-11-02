"""Compare zero-field Heisenberg and FFB bonds against independent references."""

from time import perf_counter  # noqa: I001

import torch

from variational_tensor_network.models import heisenberg_bond, heisenberg_bond_bis, ffb_bond, ffb_bond_bis


def check_heisenberg() -> None:
    """Compare the two Heisenberg constructions at zero field."""
    for J in (-1.0, 0.0, 1.0, 2.0):
        actual = heisenberg_bond(J=J, h=0.0)
        expected = heisenberg_bond_bis(J=J)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def check_ffb() -> None:
    """Compare the two FFB constructions for several couplings and fields."""
    parameters = ((2.0, 1.0, 0.0), (1.3, -0.4, 0.7), (0.0, 0.8, -0.5))
    for JD, J, h in parameters:
        actual = ffb_bond(JD=JD, J=J, h=h)
        expected = ffb_bond_bis(JD=JD, J=J, h=h)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def main() -> None:
    start = perf_counter()
    torch.set_default_dtype(torch.float64)

    check_heisenberg()
    print("\nHeisenberg | zero-field reference comparison: OK")

    check_ffb()
    print("\nFFB | reference comparison: OK")

    print(f"\nAll model comparisons passed in {perf_counter() - start:.3f} s.\n")


if __name__ == "__main__":
    main()
