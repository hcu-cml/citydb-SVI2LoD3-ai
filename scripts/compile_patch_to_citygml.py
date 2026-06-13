#!/usr/bin/env python3
import argparse
import json
import os
import re
import hashlib
from copy import deepcopy
from math import sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
from lxml import etree
from shapely.geometry import Polygon, Point, box, MultiPolygon
from shapely.ops import unary_union

# ---------------------------
# Namespaces
# ---------------------------
GEN_10 = "http://www.opengis.net/citygml/generics/1.0"
GEN_20 = "http://www.opengis.net/citygml/generics/2.0"

CITYGML_10 = "http://www.opengis.net/citygml/1.0"
CITYGML_20 = "http://www.opengis.net/citygml/2.0"

BLDG_10 = "http://www.opengis.net/citygml/building/1.0"
BLDG_20 = "http://www.opengis.net/citygml/building/2.0"

APP_10 = "http://www.opengis.net/citygml/appearance/1.0"
APP_20 = "http://www.opengis.net/citygml/appearance/2.0"

GML = "http://www.opengis.net/gml"
XLINK = "http://www.w3.org/1999/xlink"
XSI = "http://www.w3.org/2001/XMLSchema-instance"

NSMAP_20 = {
    None: CITYGML_20,
    "core": CITYGML_20,
    "bldg": BLDG_20,
    "app": APP_20,
    "gen": GEN_20,
    "gml": GML,
    "xlink": XLINK,
    "xsi": XSI,
}

# ---------------------------
# ID helpers (xs:ID safe)
# ---------------------------
IDVAL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def make_xs_id_safe(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", s)
    if not s or not re.match(r"^[A-Za-z_]", s):
        s = "id_" + s
    return s


def sanitize_gml_ids_and_xlinks(root: etree._Element) -> None:
    used = set()
    id_map = {}

    for el in root.iter():
        gid = el.get(f"{{{GML}}}id")
        if not gid:
            continue
        new = make_xs_id_safe(gid)
        if not IDVAL_RE.match(new):
            new = make_xs_id_safe(new)
        base = new
        k = 1
        while new in used:
            new = f"{base}_{k}"
            k += 1
        used.add(new)
        if new != gid:
            id_map[gid] = new
            el.set(f"{{{GML}}}id", new)

    for el in root.iter():
        href = el.get(f"{{{XLINK}}}href")
        if href and href.startswith("#"):
            old = href[1:]
            if old in id_map:
                el.set(f"{{{XLINK}}}href", "#" + id_map[old])


def stable_xml_id(prefix: str, *parts: str, length: int = 24) -> str:
    raw = "|".join(parts).encode("utf-8")
    h = hashlib.sha1(raw).hexdigest()[:length]
    return f"{prefix}{h}"


# ---------------------------
# Geometry helpers
# ---------------------------
def norm(v: np.ndarray) -> np.ndarray:
    n = sqrt(float(np.dot(v, v)))
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n


def parse_poslist(text: str) -> np.ndarray:
    vals = [float(x) for x in text.strip().split()]
    if len(vals) % 3 != 0:
        raise ValueError("gml:posList is not divisible by 3")
    return np.array(vals, dtype=float).reshape(-1, 3)


def find_first_polygon_coords(surface_geom_el: etree._Element) -> np.ndarray:
    poslist = surface_geom_el.find(f".//{{{GML}}}posList")
    if poslist is None or poslist.text is None:
        raise ValueError("No gml:posList found under host surface geometry")
    return parse_poslist(poslist.text)


def plane_axes_from_polygon(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(pts) < 3:
        raise ValueError("Need at least 3 points for a plane")

    p0 = pts[0]
    i1 = None
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - p0) > 1e-8:
            i1 = i
            break
    if i1 is None:
        raise ValueError("Degenerate polygon: all points identical")

    p1 = pts[i1]
    i2 = None
    for i in range(i1 + 1, len(pts)):
        if np.linalg.norm(np.cross(p1 - p0, pts[i] - p0)) > 1e-8:
            i2 = i
            break
    if i2 is None:
        raise ValueError("Degenerate polygon: points collinear")

    p2 = pts[i2]
    u = norm(p1 - p0)
    nvec = norm(np.cross(p1 - p0, p2 - p0))
    v = norm(np.cross(nvec, u))
    return p0, u, v


def project_uv(p0: np.ndarray, u: np.ndarray, v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rel = pts - p0
    uu = rel @ u
    vv = rel @ v
    return np.vstack([uu, vv]).T


def fix_polygon(p: Polygon) -> Polygon:
    if p.is_empty:
        raise ValueError("Empty polygon")
    if p.is_valid:
        return p
    p2 = p.buffer(0)
    if p2.is_empty:
        raise ValueError("Polygon became empty after repair")
    if p2.geom_type == "MultiPolygon":
        p2 = max(p2.geoms, key=lambda g: g.area)
    if p2.geom_type != "Polygon":
        raise ValueError(f"Unexpected repaired geometry type: {p2.geom_type}")
    return p2


def rectangularity_score(p: Polygon) -> float:
    minx, miny, maxx, maxy = p.bounds
    bbox_area = max((maxx - minx) * (maxy - miny), 1e-12)
    return float(abs(p.area) / bbox_area)


def pull_point_inside(poly: Polygon, pt: Point) -> Point:
    if poly.contains(pt):
        return pt
    ip = poly.representative_point()
    for t in np.linspace(0.0, 1.0, 21):
        q = Point(pt.x * (1 - t) + ip.x * t, pt.y * (1 - t) + ip.y * t)
        if poly.contains(q):
            return q
    return ip


def place_rect_in_poly(host_poly: Polygon, x0: float, y0: float, x1: float, y1: float, max_iter: int = 18) -> Polygon:
    host_poly = fix_polygon(host_poly)

    minx, maxx = (x0, x1) if x0 <= x1 else (x1, x0)
    miny, maxy = (y0, y1) if y0 <= y1 else (y1, y0)
    w = maxx - minx
    h = maxy - miny
    if w <= 0 or h <= 0:
        raise ValueError("Degenerate opening rectangle")

    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    center = pull_point_inside(host_poly, Point(cx, cy))
    cx, cy = center.x, center.y

    for _ in range(max_iter):
        rect = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        if host_poly.contains(rect):
            return rect
        w *= 0.85
        h *= 0.85

    raise ValueError("Could not place opening rectangle fully inside host polygon")


def _ring_normal_xyz(ring_xyz: List[Tuple[float, float, float]]) -> np.ndarray:
    pts = ring_xyz[:-1] if len(ring_xyz) > 1 and ring_xyz[0] == ring_xyz[-1] else ring_xyz
    if len(pts) < 3:
        return np.array([0.0, 0.0, 0.0], dtype=float)
    p0 = np.array(pts[0], dtype=float)
    p1 = np.array(pts[1], dtype=float)
    p2 = np.array(pts[2], dtype=float)
    return np.cross(p1 - p0, p2 - p0)


def _reverse_closed_ring(ring_xyz: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    pts = ring_xyz[:-1]
    pts.reverse()
    pts.append(pts[0])
    return pts


def uv_ring_to_xyz(coords_uv, p0: np.ndarray, u: np.ndarray, v: np.ndarray) -> List[Tuple[float, float, float]]:
    ring_xyz: List[Tuple[float, float, float]] = []
    coords = list(coords_uv)
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    for uu, vv in coords:
        p = p0 + u * float(uu) + v * float(vv)
        ring_xyz.append((float(p[0]), float(p[1]), float(p[2])))
    return ring_xyz


# ---------------------------
# XML migration + LoD3 slotting
# ---------------------------
def migrate_namespaces_to_20(root: etree._Element) -> etree._Element:
    for el in root.iter():
        if not isinstance(el.tag, str) or not el.tag.startswith("{"):
            continue
        ns, local = el.tag[1:].split("}")
        if ns == CITYGML_10:
            el.tag = f"{{{CITYGML_20}}}{local}"
        elif ns == BLDG_10:
            el.tag = f"{{{BLDG_20}}}{local}"
        elif ns == APP_10:
            el.tag = f"{{{APP_20}}}{local}"
        elif ns == GEN_10:
            el.tag = f"{{{GEN_20}}}{local}"

    new_root = etree.Element(root.tag, nsmap=NSMAP_20)
    for k, v in root.attrib.items():
        new_root.set(k, v)
    for child in list(root):
        new_root.append(child)
    return new_root


def ensure_lod3_multisurfaces(building_el: etree._Element) -> None:
    # Important: do NOT create lod3Solid. A copied lod2Solid makes a closed shell
    # with no holes, which causes the classic "ghost windows" artifact.
    lod3_solid = building_el.find(f"./{{{BLDG_20}}}lod3Solid")
    if lod3_solid is not None:
        building_el.remove(lod3_solid)

    for surf in building_el.findall(f".//{{{BLDG_20}}}boundedBy/*"):
        ms2 = surf.find(f"./{{{BLDG_20}}}lod2MultiSurface")
        ms3 = surf.find(f"./{{{BLDG_20}}}lod3MultiSurface")
        if ms2 is not None and ms3 is None:
            ms3_el = etree.Element(f"{{{BLDG_20}}}lod3MultiSurface")
            for ch in list(ms2):
                ms3_el.append(deepcopy(ch))
            parent = ms2.getparent()
            parent.insert(parent.index(ms2) + 1, ms3_el)


def get_gml_id(el: etree._Element) -> Optional[str]:
    return el.get(f"{{{GML}}}id")


def make_opening_element(opening_type: str, opening_id: str,
                         ring_xyz: List[Tuple[float, float, float]]) -> etree._Element:
    opening_wrap = etree.Element(f"{{{BLDG_20}}}opening")
    feat = etree.SubElement(opening_wrap, f"{{{BLDG_20}}}{opening_type}")
    feat.set(f"{{{GML}}}id", opening_id)

    desc = etree.SubElement(feat, f"{{{GML}}}description")
    desc.text = (
        "INFERRED/SYNTHETIC OPENING: geometry is approximate; wall shell was cut using lod3MultiSurface."
    )

    lod3ms = etree.SubElement(feat, f"{{{BLDG_20}}}lod3MultiSurface")
    ms = etree.SubElement(lod3ms, f"{{{GML}}}MultiSurface")
    ms.set(f"{{{GML}}}id", stable_xml_id("ms_", opening_id))
    sm = etree.SubElement(ms, f"{{{GML}}}surfaceMember")
    poly = etree.SubElement(sm, f"{{{GML}}}Polygon")
    poly.set(f"{{{GML}}}id", stable_xml_id("pg_", opening_id))
    exterior = etree.SubElement(poly, f"{{{GML}}}exterior")
    lr = etree.SubElement(exterior, f"{{{GML}}}LinearRing")
    lr.set(f"{{{GML}}}id", stable_xml_id("lr_", opening_id))
    poslist = etree.SubElement(lr, f"{{{GML}}}posList")
    poslist.text = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in ring_xyz)
    return opening_wrap


# ---------------------------
# Surface selection
# ---------------------------
def iter_boundary_surfaces(building_el: etree._Element, surface_type: str) -> List[etree._Element]:
    out = []
    for s in building_el.findall(f".//{{{BLDG_20}}}boundedBy/*"):
        if s.tag == f"{{{BLDG_20}}}{surface_type}":
            out.append(s)
    return out


def surface_geometry_el(surface_el: etree._Element) -> etree._Element:
    geom = surface_el.find(f"./{{{BLDG_20}}}lod3MultiSurface")
    if geom is None:
        geom = surface_el.find(f"./{{{BLDG_20}}}lod2MultiSurface")
    if geom is None:
        raise ValueError("Surface has no lod2/lod3 MultiSurface")
    return geom


def estimate_building_centroid(building_el: etree._Element) -> np.ndarray:
    pts_all = []
    for poslist in building_el.findall(f".//{{{GML}}}posList"):
        if not poslist.text:
            continue
        try:
            pts_all.append(parse_poslist(poslist.text))
        except Exception:
            continue
    if not pts_all:
        raise ValueError("Could not estimate building centroid")
    arr = np.vstack(pts_all)
    return arr.mean(axis=0)


def outward_offset_normal(building_el: etree._Element,
                          host_surface_el: etree._Element,
                          host_n: np.ndarray) -> np.ndarray:
    n = np.array(host_n, dtype=float)
    if host_surface_el.tag != f"{{{BLDG_20}}}WallSurface":
        return n
    try:
        bctr = estimate_building_centroid(building_el)
        surf_pts = find_first_polygon_coords(surface_geometry_el(host_surface_el))
        sctr = surf_pts.mean(axis=0)
        outward_xy = sctr[:2] - bctr[:2]
        n_xy = n[:2]
        if np.linalg.norm(outward_xy) > 1e-8 and np.dot(n_xy, outward_xy) < 0.0:
            n = -n
    except Exception:
        pass
    return n


def compute_surface_poly_uv(surface_el: etree._Element):
    geom_el = surface_geometry_el(surface_el)
    pts = find_first_polygon_coords(geom_el)

    p0 = pts[0]
    n = None
    for i in range(1, len(pts) - 1):
        a = pts[i] - p0
        b = pts[i + 1] - p0
        c = np.cross(a, b)
        if np.linalg.norm(c) > 1e-8:
            n = norm(c)
            break
    if n is None:
        p0, u, v = plane_axes_from_polygon(pts)
        host_n = norm(np.cross(u, v))
    else:
        host_n = n
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        is_wall = (surface_el.tag == f"{{{BLDG_20}}}WallSurface") or (abs(float(np.dot(host_n, up))) < 0.3)
        if is_wall:
            v = up - float(np.dot(up, host_n)) * host_n
            if np.linalg.norm(v) < 1e-8:
                p0, u, v = plane_axes_from_polygon(pts)
                host_n = norm(np.cross(u, v))
            else:
                v = norm(v)
                u = norm(np.cross(v, host_n))
        else:
            p0, u, v = plane_axes_from_polygon(pts)
            host_n = norm(np.cross(u, v))

    uv = project_uv(p0, u, v, pts)
    host_poly = fix_polygon(Polygon(uv))
    return host_poly, p0, u, v, host_n


def choose_surface(building_el: etree._Element, selector: Dict) -> etree._Element:
    stype = selector.get("type")
    strat = selector.get("strategy")

    if stype not in ("WallSurface", "RoofSurface"):
        raise ValueError(f"surface_selector.type must be WallSurface or RoofSurface, got: {stype}")

    surfaces = iter_boundary_surfaces(building_el, stype)
    if not surfaces:
        raise ValueError(f"No {stype} surfaces found in building")

    if strat not in ("largest_area", "largest_area_rectangular"):
        raise ValueError(f"Unsupported surface selector strategy: {strat}")

    best = None
    best_score = -1.0
    for s in surfaces:
        try:
            host_poly, _, _, _, _ = compute_surface_poly_uv(s)
            area = abs(host_poly.area)
            score = area if strat == "largest_area" else area * rectangularity_score(host_poly)
            if score > best_score:
                best_score = score
                best = s
        except Exception:
            continue

    if best is None:
        raise ValueError(f"Could not score any {stype} surface")
    return best


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_opening_uv(o: Dict, eps: float = 1e-6) -> Tuple[float, float, float, float]:
    typ = o.get("type", "Window")
    u0 = float(o["u0"]); v0 = float(o["v0"])
    u1 = float(o["u1"]); v1 = float(o["v1"])

    if typ == "Door":
        default_w = 0.10
        default_h = 0.35
    else:
        default_w = 0.12
        default_h = 0.20

    if abs(u1 - u0) < eps:
        uc = u0
        u0 = uc - default_w / 2.0
        u1 = uc + default_w / 2.0
    if abs(v1 - v0) < eps:
        vc = v0
        v0 = vc - default_h / 2.0
        v1 = vc + default_h / 2.0

    if u0 > u1:
        u0, u1 = u1, u0
    if v0 > v1:
        v0, v1 = v1, v0

    u0 = clamp01(u0); u1 = clamp01(u1)
    v0 = clamp01(v0); v1 = clamp01(v1)

    if (u1 - u0) < eps:
        u1 = clamp01(u0 + default_w)
    if (v1 - v0) < eps:
        v1 = clamp01(v0 + default_h)

    return u0, v0, u1, v1


# ---------------------------
# GML writers
# ---------------------------
def ensure_lod3_multisurface_el(surface_el: etree._Element) -> etree._Element:
    ms3 = surface_el.find(f"./{{{BLDG_20}}}lod3MultiSurface")
    if ms3 is not None:
        return ms3
    ms2 = surface_el.find(f"./{{{BLDG_20}}}lod2MultiSurface")
    ms3 = etree.Element(f"{{{BLDG_20}}}lod3MultiSurface")
    if ms2 is not None:
        surface_el.insert(surface_el.index(ms2) + 1, ms3)
    else:
        surface_el.append(ms3)
    return ms3


def clear_existing_openings(surface_el: etree._Element) -> None:
    for ch in list(surface_el):
        if ch.tag == f"{{{BLDG_20}}}opening":
            surface_el.remove(ch)


def set_surface_lod3_geometry_with_holes(surface_el: etree._Element,
                                         cut_geom,
                                         p0: np.ndarray,
                                         u: np.ndarray,
                                         v: np.ndarray,
                                         desired_n: np.ndarray,
                                         host_id: str) -> None:
    ms3 = ensure_lod3_multisurface_el(surface_el)
    for ch in list(ms3):
        ms3.remove(ch)
    ms = etree.SubElement(ms3, f"{{{GML}}}MultiSurface")

    if cut_geom.is_empty:
        raise ValueError(f"Host surface became empty after cutting openings: {host_id}")

    if isinstance(cut_geom, Polygon):
        polys = [cut_geom]
    elif isinstance(cut_geom, MultiPolygon):
        polys = list(cut_geom.geoms)
    else:
        # Try to salvage polygonal pieces only
        polys = [g for g in getattr(cut_geom, 'geoms', []) if isinstance(g, Polygon)]
        if not polys:
            raise ValueError(f"Unexpected cut geometry type: {cut_geom.geom_type}")

    for idx, poly_uv in enumerate(polys, start=1):
        if poly_uv.is_empty or poly_uv.area <= 1e-10:
            continue
        sm = etree.SubElement(ms, f"{{{GML}}}surfaceMember")
        poly_el = etree.SubElement(sm, f"{{{GML}}}Polygon")
        poly_el.set(f"{{{GML}}}id", stable_xml_id("pg_", host_id, str(idx)))

        ext_xyz = uv_ring_to_xyz(poly_uv.exterior.coords, p0, u, v)
        rn = _ring_normal_xyz(ext_xyz)
        if float(np.dot(rn, desired_n)) < 0.0:
            ext_xyz = _reverse_closed_ring(ext_xyz)

        exterior = etree.SubElement(poly_el, f"{{{GML}}}exterior")
        lr = etree.SubElement(exterior, f"{{{GML}}}LinearRing")
        lr.set(f"{{{GML}}}id", stable_xml_id("lr_", host_id, str(idx), "ext"))
        poslist = etree.SubElement(lr, f"{{{GML}}}posList")
        poslist.text = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in ext_xyz)

        for hidx, interior_uv in enumerate(poly_uv.interiors, start=1):
            int_xyz = uv_ring_to_xyz(interior_uv.coords, p0, u, v)
            rn = _ring_normal_xyz(int_xyz)
            # Interior rings should have opposite winding.
            if float(np.dot(rn, desired_n)) > 0.0:
                int_xyz = _reverse_closed_ring(int_xyz)

            interior = etree.SubElement(poly_el, f"{{{GML}}}interior")
            lr_i = etree.SubElement(interior, f"{{{GML}}}LinearRing")
            lr_i.set(f"{{{GML}}}id", stable_xml_id("lr_", host_id, str(idx), f"int{hidx}"))
            poslist_i = etree.SubElement(lr_i, f"{{{GML}}}posList")
            poslist_i.text = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in int_xyz)


def collect_rectangles_uv(host_poly: Polygon, openings: List[Dict], minx: float, miny: float, du: float, dv: float,
                          host_id: str) -> List[Tuple[Dict, Polygon]]:
    rects: List[Tuple[Dict, Polygon]] = []
    for o in openings:
        otype = o.get("type")
        if otype not in ("Window", "Door"):
            continue
        u0, v0, u1, v1 = normalize_opening_uv(o)
        x0 = minx + u0 * du
        y0 = miny + v0 * dv
        x1 = minx + u1 * du
        y1 = miny + v1 * dv
        try:
            rect_uv = place_rect_in_poly(host_poly, x0, y0, x1, y1)
        except Exception as e:
            print(f"[skip] {otype} on {host_id}: {e}")
            continue
        rects.append((o, rect_uv))
    return rects


# ---------------------------
# Patch application
# ---------------------------
def apply_add_openings(building_el: etree._Element, op: Dict, building_idx: int) -> int:
    target_surface_id = op.get("target_surface_id")
    if target_surface_id:
        host_list = building_el.xpath(
            f".//*[@gml:id='{target_surface_id}']",
            namespaces={"gml": GML},
        )
        if not host_list:
            raise ValueError(f"Host surface gml:id not found: {target_surface_id}")
        host = host_list[0]
    else:
        selector = op.get("surface_selector")
        if not selector:
            raise ValueError("add_openings op requires either target_surface_id or surface_selector")
        host = choose_surface(building_el, selector)

    ensure_lod3_multisurfaces(building_el)
    clear_existing_openings(host)

    host_id = get_gml_id(host)
    if not host_id:
        raise ValueError("Chosen host surface has no gml:id")

    host_poly, p0, u, v, host_n = compute_surface_poly_uv(host)
    desired_n = outward_offset_normal(building_el, host, host_n)

    minx, miny, maxx, maxy = host_poly.bounds
    du = maxx - minx
    dv = maxy - miny
    if du <= 0 or dv <= 0:
        raise ValueError("Degenerate host surface bounds")

    rects = collect_rectangles_uv(host_poly, op.get("openings", []), minx, miny, du, dv, host_id)
    if not rects:
        return 0

    rect_polys = [r for _, r in rects]
    holes_union = unary_union(rect_polys)
    cut_geom = fix_polygon(host_poly.difference(holes_union)) if isinstance(host_poly.difference(holes_union), Polygon) else host_poly.difference(holes_union).buffer(0)
    if cut_geom.is_empty:
        raise ValueError(f"All of host surface was removed by openings on {host_id}")

    set_surface_lod3_geometry_with_holes(host, cut_geom, p0, u, v, desired_n, host_id)

    emit_opening_features = bool(op.get("emit_opening_features", True))
    opening_surface_offset_m = float(op.get("opening_surface_offset_m", 0.0))
    added = 0
    if emit_opening_features:
        for o, rect_uv in rects:
            ring_xyz = uv_ring_to_xyz(rect_uv.exterior.coords, p0, u, v)
            rn = _ring_normal_xyz(ring_xyz)
            if float(np.dot(rn, desired_n)) < 0.0:
                ring_xyz = _reverse_closed_ring(ring_xyz)
            if abs(opening_surface_offset_m) > 0.0:
                ring_xyz = [
                    (x + desired_n[0] * opening_surface_offset_m,
                     y + desired_n[1] * opening_surface_offset_m,
                     z + desired_n[2] * opening_surface_offset_m)
                    for (x, y, z) in ring_xyz
                ]
            opening_id = stable_xml_id("op_", host_id, str(building_idx), str(added + 1))
            host.append(make_opening_element(o.get("type", "Window"), opening_id, ring_xyz))
            added += 1
    else:
        added = len(rects)

    return added



def strip_lod2_geometry(building_el: etree._Element) -> None:
    # Match the valid reference style more closely: keep only LoD3 geometry.
    # Leaving lod2 geometry in place confuses some viewers, which then render
    # the old wall and the new opening panels at the same time.
    for tag in (
        f"{{{BLDG_20}}}lod2Solid",
        f"{{{BLDG_20}}}lod2MultiSurface",
        f"{{{BLDG_20}}}lod2MultiCurve",
        f"{{{BLDG_20}}}lod2TerrainIntersection",
    ):
        for el in list(building_el.findall(f"./{tag}")):
            building_el.remove(el)

    for surf in building_el.findall(f".//{{{BLDG_20}}}boundedBy/*"):
        for el in list(surf.findall(f"./{{{BLDG_20}}}lod2MultiSurface")):
            surf.remove(el)
        for el in list(surf.findall(f"./{{{BLDG_20}}}lod2MultiCurve")):
            surf.remove(el)
        for el in list(surf.findall(f"./{{{BLDG_20}}}lod2TerrainIntersection")):
            surf.remove(el)


# ---------------------------
# Building selection + main
# ---------------------------
def select_buildings(root20: etree._Element, apply_to) -> List[Tuple[int, etree._Element]]:
    buildings = root20.findall(f".//{{{BLDG_20}}}Building")
    if not buildings:
        raise SystemExit("No bldg:Building found")

    if apply_to == "all":
        return list(enumerate(buildings))
    if apply_to in (None, "first"):
        return [(0, buildings[0])]
    if isinstance(apply_to, int):
        if apply_to < 0 or apply_to >= len(buildings):
            raise SystemExit(f"apply_to index out of range: {apply_to}")
        return [(apply_to, buildings[apply_to])]

    raise SystemExit("apply_to must be 'all', 'first', or an integer index")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lod2", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.patch, "r", encoding="utf-8") as f:
        patch = json.load(f)

    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    tree = etree.parse(args.lod2, parser)
    root = tree.getroot()

    root20 = migrate_namespaces_to_20(root)
    tree._setroot(root20)

    apply_to = patch.get("apply_to", "first")
    ops = patch.get("ops", [])
    if not isinstance(ops, list) or not ops:
        raise SystemExit("Patch must contain a non-empty ops[] list")

    selected = select_buildings(root20, apply_to)

    for bidx, bldg in selected:
        ensure_lod3_multisurfaces(bldg)
        for op in ops:
            if op.get("op") == "add_openings":
                apply_add_openings(bldg, op, bidx)
            else:
                raise SystemExit(f"Unsupported op: {op.get('op')}")
        strip_lod2_geometry(bldg)

    sanitize_gml_ids_and_xlinks(root20)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tree.write(args.out, pretty_print=True, xml_declaration=True, encoding="UTF-8")


if __name__ == "__main__":
    main()
