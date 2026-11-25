DEFAULT_SU_STABLE_STEPS = 20
DEFAULT_CTMRG_STABLE_STEPS = 5
DEFAULT_CHI_STABLE_STEPS = 3

DEFAULT_SU_MAX_STEPS = 10000
DEFAULT_CTMRG_MAX_STEPS = 5000
DEFAULT_CHI_MAX_STEPS = 20

DEFAULT_SU_SCHEDULE = ((.1, 1e-6), (.01, 1e-7), (.001, 1e-8)) # DEFAULT_SU_SCHEDULE = ((.01, 1e-6), (.005, 1e-7), (.001, 1e-8)) # DEFAULT_SU_SCHEDULE = ((.01, 1e-6), (.001, 1e-8)) # DEFAULT_SU_SCHEDULE = ((.1, 1e-7), (.01, 1e-8), (.001, 1e-9))
DEFAULT_CTMRG_TOL = 1e-7
DEFAULT_CHI_TOL = 1e-6


DEFAULT_TILING = "abcd" # "checkerboard", "abcd", "vertical_stripes_2", "horizontal_stripes_2", "vertical_stripes_3", "horizontal_stripes_3", "rectangle_2x3", "diagonal_3"
DEFAULT_MODEL = "x_shastry_sutherland" # "heisenberg", "shastry_sutherland", "ffb", "x_shastry_sutherland"
DEFAULT_JD = 1.0
DEFAULT_J = DEFAULT_JD *.3
DEFAULT_J1 = DEFAULT_JD * .35
DEFAULT_J2 = DEFAULT_JD * .25
DEFAULT_h = .0
DEFAULT_PARAMETERS_BY_MODEL = {
    "heisenberg":           {"J": DEFAULT_J, "h": DEFAULT_h},  # Expected energy per spin : −0.669441857(7) * J
    "shastry_sutherland":   {"JD": DEFAULT_JD, "J": DEFAULT_J, "h": DEFAULT_h},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0.429(6)
    "ffb":                  {"JD": DEFAULT_JD, "J": DEFAULT_J, "h": DEFAULT_h},  # Expected energy per spin : -3/8 * JD  valid for J/JD <≈ 0,675(2)
    "x_shastry_sutherland": {"JD": DEFAULT_JD, "J1": DEFAULT_J1, "J2": DEFAULT_J2, "h": DEFAULT_h}, 
}

DEFAULT_SEED = 0

DEFAULT_VERBOSITY = 2
