import svgwrite
import logging
import math

def _bounds(pts):
    first = pts[0]
    minx = maxx = float(first[0])
    miny = maxy = float(first[1])

    for i in range(1, len(pts)):
        x, y = pts[i]
        if x < minx:
            minx = x
        elif x > maxx:
            maxx = x
        if y < miny:
            miny = y
        elif y > maxy:
            maxy = y

    return (minx, miny, maxx, maxy)


def _scale_points(pts, minx, miny, s, tx, ty):
    out = []
    for i in range(len(pts)):
        x = (float(pts[i][0]) - minx) * s + tx
        y = (float(pts[i][1]) - miny) * s + ty
        out.append((x, y))
    return out


def _rotate_points(points, center, angle_rad):
    cx, cy = center
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [(cx + (x - cx)*c - (y - cy)*s,
             cy + (x - cx)*s + (y - cy)*c) for x, y in points]


def generate_2d_svg(debugger, filename, pts1, center1, angle1, pts2, center2, angle2, *, pad=10, box=1000):
    points1 = _rotate_points(pts1, center1, angle1)
    points2 = _rotate_points(pts2, center2, angle2)

    # bounds with pad in source space
    b1 = _bounds(points1)
    b2 = _bounds(points2)
    bc = _bounds([center1, center2])

    minx = min(b1[0], b2[0], bc[0])
    miny = min(b1[1], b2[1], bc[1])
    maxx = max(b1[2], b2[2], bc[2])
    maxy = max(b1[3], b2[3], bc[3])

    w = maxx - minx
    h = maxy - miny

    logging.debug(f'Bounds(point1)=', b1)
    logging.debug(f'Bounds(point2)=', b2)
    logging.debug(f'Bounds(centers)=', bc)
    logging.debug(f'Native size ({w}:{h})')

    # uniform scale to fit box×box, centered
    s = min((box - pad) / w, (box - pad) / h)
    tx = (box - w * s) / 2.0
    ty = (box - h * s) / 2.0

    logging.debug(f'Target size ({w*s}:{h*s}), scale {s}, center ({tx}:{ty})')

    # scale geometry into final coordinates (no SVG transforms)
    p1 = _scale_points(points1, minx, miny, s, tx, ty)
    p2 = _scale_points(points2, minx, miny, s, tx, ty)
    c1x, c1y = _scale_points([center1], minx, miny, s, tx, ty)[0]
    c2x, c2y = _scale_points([center2], minx, miny, s, tx, ty)[0]
    cross_size = 0.05 * min(w, h) * s  # 5% of smallest bound, scaled

    filename = debugger.file_path(filename)
    dwg = svgwrite.Drawing(filename, size=(box, box), viewBox=f"0 0 {box} {box}", profile="tiny")
    dwg.attribs["shape-rendering"] = "geometricPrecision"

    style = {"fill": "none", "stroke": "black",
             "stroke_width": 1, "vector_effect": "non-scaling-stroke"}

    dwg.add(dwg.polyline(points=p1, **style))
    dwg.add(dwg.polyline(points=p2, **style))

    # crosses (hairline)
    dwg.add(dwg.line(start=(c1x - cross_size, c1y), end=(c1x + cross_size, c1y), **style))
    dwg.add(dwg.line(start=(c1x, c1y - cross_size), end=(c1x, c1y + cross_size), **style))
    dwg.add(dwg.line(start=(c2x - cross_size, c2y), end=(c2x + cross_size, c2y), **style))
    dwg.add(dwg.line(start=(c2x, c2y - cross_size), end=(c2x, c2y + cross_size), **style))

    dwg.save()

    return filename