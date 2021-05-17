import math
from typing import List


def calc_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def interpolate_waypoint(way_points, margin=10) -> List:
    assert len(way_points) > 2

    curr_idx = 2
    curr_first_pt, curr_second_pt = way_points[0], way_points[1]
    ref_pt = curr_first_pt

    interpolated_pts = []

    while True:
        dist = calc_distance(ref_pt, curr_second_pt)
        print("{} - {}".format(ref_pt, dist))
        if dist < margin:
            if curr_idx >= len(way_points):
                break
            curr_first_pt, curr_second_pt = way_points[curr_idx - 1], way_points[curr_idx]
            curr_idx += 1
        else:
            xa, xb = curr_first_pt[0], curr_second_pt[0]
            ya, yb = curr_first_pt[1], curr_second_pt[1]
            xref, yref = ref_pt[0], ref_pt[1]

            int_x, int_y = circle_line_intersection(xa, xb, ya, yb, xref, yref, margin)
            ref_pt = [int_x, int_y]
            interpolated_pts.append(ref_pt)

    return interpolated_pts


def circle_line_intersection(xa, xb, ya, yb, xref, yref, radius):
    """
    An efficient implementation for the circle-line detection algorithm
    ref: https://mathworld.wolfram.com/Circle-LineIntersection.html
    """
    xa, xb = xa - xref, xb - xref
    ya, yb = ya - yref, yb - yref

    dx, dy = xb - xa, yb - ya
    dr_square = dx ** 2 + dy ** 2
    D = xa * yb - xb * ya

    # discriminant
    delta = dr_square * radius ** 2 - D ** 2
    if delta < 0:
        raise RuntimeError("circle line intersection algorithm error")
    elif delta == 0:
        isecx = D * dy / dr_square
        isecy = -D * dx / dr_square
        return (isecx + xref, isecy + yref)
    else:
        signy = 1 if dy >= 0 else -1
        isecx = (D * dy + signy * dx * math.sqrt(delta)) / dr_square

        if max(abs(isecx - xa), abs(isecx - xb)) <= abs(dx):
            isecy = (-D * dx + abs(dy) * math.sqrt(delta)) / dr_square
        else:
            isecx = (D * dy - signy * dx * math.sqrt(delta)) / dr_square
            isecy = (-D * dx - abs(dy) * math.sqrt(delta)) / dr_square

        return (isecx + xref, isecy + yref)


if __name__ == '__main__':
    demo_points = [[0, 0], [0, 11], [0, 22], [0, 33]]
    int_points = interpolate_waypoint(demo_points, 5)
    # print(circle_line_intersection(1, 2, 1, 2, 0, 0, 2))
