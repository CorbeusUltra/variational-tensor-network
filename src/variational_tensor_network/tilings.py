"""Define the sites in the unit cell. Coordinates: x increases to the right; y increases downward."""

from collections.abc import Callable

type Site = tuple[int, int]
type UnitCellCoord = Callable[[int, int], Site]


def rotated_unitcell_coord(unitcell_coord: Callable[[int, int], Site]) -> Callable[[int, int], Site]:
    def rotated(x: int, y: int) -> Site:
        sx, sy = unitcell_coord(-y, x)
        return (sy, -sx)

    return rotated


def checkerboard(x: int, y: int) -> Site:
    return ((x + y) % 2, 0)

def abcd(x: int, y: int) -> Site:
    return (x % 2,y % 2)

def vertical_stripes_2(x: int, y: int) -> Site:
    return (x % 2, 0)

def horizontal_stripes_2(x: int, y: int) -> Site:
    return (0, y % 2)

def vertical_stripes_3(x: int, y: int) -> Site:
    return (x % 3, 0)

def horizontal_stripes_3(x: int, y: int) -> Site:
    return (0, y % 3)

def rectangle_2x3(x: int, y: int) -> Site:
    return (x % 2, y % 3)

def diagonal_3(x: int, y: int) -> Site:
    return ((x + y) % 3, 0)


TILINGS = {
    "checkerboard": (((0, 0), (1, 0)), checkerboard),
    "abcd": (((0,0),(1,0), (0,1), (1,1)), abcd),
    "vertical_stripes_2": (((0, 0), (1, 0)), vertical_stripes_2),
    "horizontal_stripes_2": (((0, 0), (0, 1)), horizontal_stripes_2),
    "vertical_stripes_3": (tuple((x, 0) for x in range(3)), vertical_stripes_3),
    "horizontal_stripes_3": (tuple((0, y) for y in range(3)), horizontal_stripes_3),
    "rectangle_2x3": (tuple((x, y) for y in range(3) for x in range(2)), rectangle_2x3),
    "diagonal_3": (tuple((x, 0) for x in range(3)), diagonal_3),
}


def main() -> None:
    TILING_NAME = "rectangle_2x3"
    sites_coords, unitcell_coord = TILINGS[TILING_NAME]
    print(f"unitcell_coord(2,3)={unitcell_coord(2,3)}")
    print(f"unitcell_coord(3,3)={unitcell_coord(3,3)}")
    print(f"sites_coords={sites_coords}")


if __name__ == "__main__":
    main()
