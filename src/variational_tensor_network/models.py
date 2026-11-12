"""Define two-site Hamiltonians and their model metadata."""

from functools import reduce

from torch import Tensor, eye, kron


def interaction_term(S_alpha: Tensor, S_beta: Tensor, i: int, j: int, n: int, *, prefactor: float = 1.0) -> Tensor:
    """Return prefactor * S_alpha(i) * S_beta(j) embedded in n spins, requiring 1 <= i < j <= n."""
    if not (1 <= i < j <= n):
        raise ValueError(f"Expected 1 <= i < j <= n, got i={i}, j={j}, n={n}.")
    Left = eye(2 ** (i - 1))
    Middle = eye(2 ** (j - i - 1))
    Right = eye(2 ** (n - j))
    return prefactor * reduce(
        kron, (Left, S_alpha, Middle, S_beta, Right)
    )  # equivalent to kron(kron(kron(kron(Left, S_alpha), Middle), S_beta), Right)


def heisenberg_term(i: int = 1, j: int = 2, n: int = 2, *, J: float = 1.0) -> Tensor:
    """Return J * S_i·S_j embedded in n spins, requiring 1 <= i < j <= n."""
    S_z = Tensor([[0.5, 0.0],
                  [0.0, -0.5]])
    S_p = Tensor([[0.0, 1.0],
                  [0.0, 0.0]])
    S_m = Tensor([[0.0, 0.0],
                  [1.0, 0.0],]) # equivalent to S_m = S_p.mT.contiguous()
    H  = interaction_term(S_z, S_z, i, j, n, prefactor=J)
    H += interaction_term(S_p, S_m, i, j, n, prefactor=J / 2)
    H += interaction_term(S_m, S_p, i, j, n, prefactor=J / 2)
    return H


def spin_x(i: int, n: int) -> Tensor:
    """Return h * S_i^z embedded in n spins, requiring 1 <= i <= n."""
    if not (1 <= i <= n):
        raise ValueError(f"Expected 1 <= i <= n, got i={i}, n={n}.")
    Left = eye(2 ** (i - 1))
    S_x = Tensor([[0.0, 0.5],
                  [0.5, 0.0]])
    Right = eye(2 ** (n - i))
    return reduce(kron, (Left, S_x, Right))


def spin_z(i: int, n: int) -> Tensor:
    """Return h * S_i^z embedded in n spins, requiring 1 <= i <= n."""
    if not (1 <= i <= n):
        raise ValueError(f"Expected 1 <= i <= n, got i={i}, n={n}.")
    Left = eye(2 ** (i - 1))
    S_z = Tensor([[0.5, 0.0],
                  [0.0, -0.5]])
    Right = eye(2 ** (n - i))
    return reduce(kron, (Left, S_z, Right))



def heisenberg_bond(J: float = 1.0, h: float = 0) -> Tensor:
    """Return the square-lattice Heisenberg bond with quarter-weight Zeeman terms."""
    H = heisenberg_term(i=1, j=2, n=2, J=J)
    H -= h/4 * spin_z(i=1, n=2)
    H -= h/4 * spin_z(i=2, n=2)
    return H


def ffb_bond(JD: float = 2.0, J: float = 1.0, h: float = 0) -> Tensor:
    """
    Return the FFB bond between dimers (1, 2) and (3, 4),
    coupling all four inter-dimer spin pairs.

           1
         / ║ \
        /  ║  \
       /   2   \
      /  ╱   ╲  \
    3 ══════════ 4
    """
    # Intra-dimer terms (Share each dimer's intra-dimer term among its four incident bonds so divide prefactor by 4)
    H  = heisenberg_term(i=1, j=2, n=4, J=JD/4)
    H += heisenberg_term(i=3, j=4, n=4, J=JD/4)
    # Inter-dimer terms
    for i in (1,2):
        for j in (3,4):
            H += heisenberg_term(i,j,n=4, J=J)
    # Zeeman terms (Share each spin's Zeeman term among its dimer's four incident bonds so divide prefactor by 4)
    for i in (1,2,3,4):
        H -= h/4 * spin_z(i, n=4)
    return H


def shastry_sutherland_bond(JD: float = 2.0, J: float = 1.0, h: float = 0) -> Tensor:
    """
    Return an oriented dimer-pair term with inter-dimer couplings (2, 3) and (2, 4).
        1
        │
        2
       / \
      3───4
    """
    # Intra-dimer terms  (Share each dimer's intra-dimer term among its four incident bonds so divide prefactor by 4)
    H = heisenberg_term(i=1, j=2, n=4, J=JD/4)
    H += heisenberg_term(i=3, j=4, n=4, J=JD/4)
    # Inter-dimer terms
    H += heisenberg_term(i=2, j=3, n=4, J=J)
    H += heisenberg_term(i=2, j=4, n=4, J=J)
    # Zeeman terms  (Share each spin's Zeeman term among its dimer's four incident bonds so divide prefactor by 4)
    for i in (1,2,3,4):
        H -= h/4 * spin_z(i, n=4)
    return H


def x_shastry_sutherland(JD: float = 2.0, J1: float = 1.0, J2: float = 0.5, h: float = 0):
    """
    Extended Shastry-Sutherland bond between dimers (1, 2) and (3, 4).

               1
            ⋰ ║ ⋱
          ⋰   ║   ⋱
        ⋰     2     ⋱
      ⋰     ╱   ╲     ⋱
    3 ═══════════════════ 4

    JD: double lines (1, 2), (3, 4)
    J:  solid lines  (2, 3), (2, 4)
    J2: dotted lines (1, 3), (1, 4)
    """
    # Intra-dimer terms (Share each dimer's intra-dimer term among its four incident bonds so divide prefactor by 4)
    H  = heisenberg_term(i=1, j=2, n=4, J=JD/4)
    H += heisenberg_term(i=3, j=4, n=4, J=JD/4)
    # Inter-dimer terms
    H += heisenberg_term(i=2, j=3, n=4, J=J1)
    H += heisenberg_term(i=2, j=4, n=4, J=J1)
    H += heisenberg_term(i=1, j=3, n=4, J=J2)
    H += heisenberg_term(i=1, j=4, n=4, J=J2)
    # Zeeman terms (Share each spin's Zeeman term among its dimer's four incident bonds so divide prefactor by 4)
    for i in (1,2,3,4):
        H -= h/4 * spin_z(i, n=4)
    return H



MODELS = {
    "heisenberg": {
        "bond": heisenberg_bond,
        "d": 2,
        "spins_per_peps_site": 1,
        "label": "Heisenberg",
    },
    "ffb": {
        "bond": ffb_bond,
        "d": 4,
        "spins_per_peps_site": 2,
        "label": "Fully frustrated bilayer",
    },
    "shastry_sutherland": {
        "bond": shastry_sutherland_bond,
        "d": 4,
        "spins_per_peps_site": 2,
        "label": "Shastry-Sutherland",
    },
    "x_shastry_sutherland": {
        "bond": x_shastry_sutherland,
        "d": 4,
        "spins_per_peps_site": 2,
        "label": "Extended Shastry-Sutherland",
    }
}



# Legacy code

    # "heisenberg_bis": {
    #     "bond": heisenberg_bond_bis,
    #     "d": 2,
    #     "spins_per_peps_site": 1,
    #     "label": "Heisenberg bis",
    # },
    # "ffb_bis": {
    #     "bond": ffb_bond_bis,
    #     "d": 4,
    #     "spins_per_peps_site": 2,
    #     "label": "Fully frustrated bilayer bis",
    # },


def heisenberg_bond_bis(J: float = 1.0) -> Tensor:
    """Return the zero-field Heisenberg bond using an explicit reference matrix."""
    H = Tensor([[1.0,  0.0,  0.0, 0.0],
                [0.0, -1.0,  2.0, 0.0],
                [0.0,  2.0, -1.0, 0.0],
                [0.0,  0.0,  0.0, 1.0]])
    return (J / 4) * H


def ffb_bond_bis(JD: float = 3.0, J: float = 1.0, h: float = 0) -> Tensor:
    """Return the reference FFB bond using total-dimer spin operators."""
    I = eye(4)
    Tz = Tensor([[1.0, 0.0, 0.0,  0.0],
                 [0.0, 0.0, 0.0,  0.0],
                 [0.0, 0.0, 0.0,  0.0],
                 [0.0, 0.0, 0.0,  -1.0]])
    Tp = Tensor([[0.0, 1.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0, 0.0]])
    Tm = Tp.mT.contiguous()
    H_Heis = heisenberg_bond_bis()
    H_JD = kron(H_Heis, I) + kron(I, H_Heis)
    H_J = kron(Tz, Tz) + (1 / 2) * (kron(Tp, Tm) + kron(Tm, Tp))
    H_h = kron(Tz, I) + kron(I, Tz)
    return (JD / 4) * H_JD + J * H_J - (h / 4) * H_h
