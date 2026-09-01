#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import List, Optional, Tuple

import numpy as np
from lxml import etree
from PIL import Image, ExifTags

GML = "http://www.opengis.net/gml"
GML_ID = f"{{{GML}}}id"

# Optional geospatial deps
try:
    import pyproj
except Exception:
    pyproj = None

try:
    import rasterio
    from rasterio.transform import xy as rio_xy
except Exception:
    rasterio = None
    rio_xy = None


def parse_camera_en_from_filename(fname: str) -> Tuple[str, float, float]:
    """
    Works with spaces/underscores and ignores other junk like "97°"
    Example: "S1_97°_32U 565770 5940383_04022026_140815.png"
    """
    m = re.search(r"(\d{1,2}[A-Z])[\s_]+(\d{5,7})[\s_]+(\d{6,8})", fname)
    if not m:
        raise RuntimeError(f"Could not parse UTM coords from filename: {fname}")
    zone_band = m.group(1)
    e = float(m.group(2))
    n = float(m.group(3))
    return zone_band, e, n


def _first_lod2_srs_name(root) -> Optional[str]:
    vals = root.xpath("//@srsName")
    if vals:
        return str(vals[0])
    v = root.get("srsName")
    if v:
        return str(v)
    return None


def _crs_from_srs(srs_name: str):
    if pyproj is None:
        raise RuntimeError("pyproj is required for CRS transforms but is not installed.")
    s = str(srs_name).strip()
    try:
        return pyproj.CRS.from_user_input(s)
    except Exception:
        pass
    if s.lower().startswith("urn:adv:crs:"):
        token = s.split(":")[-1]
        horiz = token.split("*")[0].upper()
        adv_map = {
            "ETRS89_UTM32": 25832,
            "ETRS89_UTM33": 25833,
        }
        if horiz in adv_map:
            return pyproj.CRS.from_epsg(adv_map[horiz])
        raise RuntimeError(
            f"Unsupported AdV CRS token '{horiz}' in srsName '{s}'. "
            "Add it to adv_map in _crs_from_srs()."
        )
    m = re.search(r"EPSG[:/]{1,2}(\d+)", s, flags=re.IGNORECASE)
    if m:
        return pyproj.CRS.from_epsg(int(m.group(1)))
    raise RuntimeError(f"Could not parse CRS from srsName: {srs_name}")


def _transform_xy(x: float, y: float, src_crs, dst_crs) -> Tuple[float, float]:
    if pyproj is None:
        raise RuntimeError("pyproj is required for CRS transforms but is not installed.")
    if pyproj.CRS.from_user_input(src_crs) == pyproj.CRS.from_user_input(dst_crs):
        return float(x), float(y)
    tfm = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xx, yy = tfm.transform(x, y)
    return float(xx), float(yy)


def _exif_gps_to_lonlat(image_path: str) -> Optional[Tuple[float, float]]:
    try:
        img = Image.open(image_path)
        exif = img.getexif()
        if not exif:
            return None
        gps_ifd = None
        try:
            gps_ifd_tag = getattr(ExifTags, "IFD", None)
            if gps_ifd_tag is not None and hasattr(ExifTags.IFD, "GPSInfo"):
                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except Exception:
            gps_ifd = None
        if not gps_ifd:
            gps_tag = None
            for k, v in ExifTags.TAGS.items():
                if v == "GPSInfo":
                    gps_tag = k
                    break
            if gps_tag is not None:
                gps_ifd = exif.get(gps_tag)
        if not gps_ifd:
            return None
        gps_named = {}
        for k, v in gps_ifd.items():
            name = ExifTags.GPSTAGS.get(k, k)
            gps_named[name] = v
        if "GPSLatitude" not in gps_named or "GPSLongitude" not in gps_named:
            return None

        def _rat_to_float(r):
            try:
                return float(r)
            except Exception:
                try:
                    return float(r[0]) / float(r[1])
                except Exception:
                    return None

        def _dms_to_deg(dms):
            if dms is None or len(dms) < 3:
                return None
            d = _rat_to_float(dms[0])
            m = _rat_to_float(dms[1])
            s = _rat_to_float(dms[2])
            if d is None or m is None or s is None:
                return None
            return d + m / 60.0 + s / 3600.0

        lat = _dms_to_deg(gps_named.get("GPSLatitude"))
        lon = _dms_to_deg(gps_named.get("GPSLongitude"))
        if lat is None or lon is None:
            return None
        lat_ref = gps_named.get("GPSLatitudeRef", "N")
        lon_ref = gps_named.get("GPSLongitudeRef", "E")
        if isinstance(lat_ref, (bytes, bytearray)):
            lat_ref = lat_ref.decode(errors="ignore")
        if isinstance(lon_ref, (bytes, bytearray)):
            lon_ref = lon_ref.decode(errors="ignore")
        lat_ref = str(lat_ref).strip().upper()
        lon_ref = str(lon_ref).strip().upper()
        if lat_ref == "S":
            lat = -lat
        if lon_ref == "W":
            lon = -lon
        return float(lon), float(lat)
    except Exception:
        return None


def _camera_from_georef_raster(image_path: str, lod2_crs) -> Optional[Tuple[float, float]]:
    if rasterio is None:
        return None
    try:
        with rasterio.open(image_path) as ds:
            if ds.crs is None:
                return None
            col = ds.width / 2.0
            row = ds.height / 2.0
            x, y = rio_xy(ds.transform, row, col, offset="center")
            x = float(x)
            y = float(y)
            return _transform_xy(x, y, ds.crs, lod2_crs)
    except Exception:
        return None


def _camera_from_exif_gps(image_path: str, lod2_crs) -> Optional[Tuple[float, float]]:
    lonlat = _exif_gps_to_lonlat(image_path)
    if lonlat is None:
        return None
    lon, lat = lonlat
    return _transform_xy(lon, lat, "EPSG:4326", lod2_crs)


def get_camera_point_en(
    lod2_path: str,
    image_path: Optional[str],
    image_name_fallback: Optional[str],
    source_mode: str = "auto",
    debug: bool = False,
) -> Tuple[float, float, str]:
    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    root = etree.parse(lod2_path, parser).getroot()
    srs_name = _first_lod2_srs_name(root)
    if srs_name is None:
        raise RuntimeError("Could not find srsName in LoD2 GML; cannot transform image georeference.")
    lod2_crs = _crs_from_srs(srs_name)

    if source_mode not in {"auto", "exif", "raster", "filename"}:
        raise RuntimeError(f"Invalid --camera_source: {source_mode}")

    if source_mode == "exif":
        if not image_path:
            raise RuntimeError("--camera_source exif requires --image")
        pt = _camera_from_exif_gps(image_path, lod2_crs)
        if pt is None:
            raise RuntimeError("No EXIF GPS camera point found in image.")
        return pt[0], pt[1], "exif_gps"

    if source_mode == "raster":
        if not image_path:
            raise RuntimeError("--camera_source raster requires --image")
        pt = _camera_from_georef_raster(image_path, lod2_crs)
        if pt is None:
            raise RuntimeError("No usable georeferenced raster transform found in image.")
        return pt[0], pt[1], "raster_center"

    if source_mode == "filename":
        if not image_name_fallback:
            raise RuntimeError("--camera_source filename requires --image_name")
        _, e, n = parse_camera_en_from_filename(image_name_fallback)
        return e, n, "filename"

    # auto mode: EXIF -> raster -> filename fallback
    if image_path:
        pt = _camera_from_exif_gps(image_path, lod2_crs)
        if pt is not None:
            return pt[0], pt[1], "exif_gps"
        pt = _camera_from_georef_raster(image_path, lod2_crs)
        if pt is not None:
            return pt[0], pt[1], "raster_center"

    if image_name_fallback:
        _, e, n = parse_camera_en_from_filename(image_name_fallback)
        return e, n, "filename"

    raise RuntimeError(
        "Could not determine camera point. Provide --image (georeferenced) or --image_name (filename UTM fallback)."
    )


# -----------------------------
# Wall geometry helpers
# -----------------------------

def extract_ground_edge_segment_xy_from_wall(wall_el, z_tol=0.20):
    """
    Return EXACTLY two XY points (p0, p1) representing a ground edge segment
    chosen in ring order.
    """
    pos = wall_el.xpath(".//*[local-name()='posList'][1]")
    if not pos or pos[0].text is None:
        return None

    vals = [float(x) for x in pos[0].text.split()]
    if len(vals) < 9:
        return None

    pts = np.array(vals, dtype=float).reshape(-1, 3)

    if pts.shape[0] >= 2 and np.allclose(pts[0], pts[-1], atol=1e-9):
        pts = pts[:-1]

    if pts.shape[0] < 2:
        return None

    zmin = float(np.min(pts[:, 2]))
    base_mask = pts[:, 2] <= (zmin + z_tol)

    n = pts.shape[0]
    for i in range(n):
        j = (i + 1) % n
        if base_mask[i] and base_mask[j]:
            p0 = pts[i]
            p1 = pts[j]
            if float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) > 1e-6:
                return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))

    for i in range(n):
        j = (i + 1) % n
        p0 = pts[i]
        p1 = pts[j]
        if float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) > 1e-6:
            return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))

    return None


def extract_wall_height_m_from_wall(wall_el) -> Optional[float]:
    pos = wall_el.xpath(".//*[local-name()='posList'][1]")
    if not pos or pos[0].text is None:
        return None
    vals = [float(x) for x in pos[0].text.split()]
    if len(vals) < 9:
        return None
    pts = np.array(vals, dtype=float).reshape(-1, 3)
    if pts.shape[0] >= 2 and np.allclose(pts[0], pts[-1], atol=1e-9):
        pts = pts[:-1]
    if pts.shape[0] < 2:
        return None
    zmin = float(np.min(pts[:, 2]))
    zmax = float(np.max(pts[:, 2]))
    return max(0.0, zmax - zmin)


def _wall_normal_xy(seg) -> Optional[Tuple[float, float]]:
    """Return the unit outward-facing normal in XY for a ground-edge segment."""
    (x0, y0), (x1, y1) = seg
    dx = x1 - x0
    dy = y1 - y0
    nx = -dy
    ny = dx
    length = float(np.hypot(nx, ny))
    if length < 1e-9:
        return None
    return nx / length, ny / length


def _point_to_line_dist(px: float, py: float, lx0: float, ly0: float, lx1: float, ly1: float) -> float:
    """Perpendicular distance from point (px, py) to the infinite line through (lx0,ly0)-(lx1,ly1)."""
    dx = lx1 - lx0
    dy = ly1 - ly0
    length = float(np.hypot(dx, dy))
    if length < 1e-9:
        return float(np.hypot(px - lx0, py - ly0))
    # Signed area / length
    return abs((px - lx0) * dy - (py - ly0) * dx) / length


# -----------------------------
# Core: single wall selection
# -----------------------------

def choose_wall_surface_id(
    lod2_path: str,
    camera_easting: float,
    camera_northing: float,
    z_tol=0.20,
    min_wall_width_m: float = 1.0,
    min_wall_height_m: float = 1.0,
    debug=False,
) -> str:
    """
    Returns the gml:id of the single WallSurface that best faces the camera.
    (Kept for backward-compatibility; prefer choose_facade_wall_group for multi-wall facades.)
    """
    group = choose_facade_wall_group(
        lod2_path=lod2_path,
        camera_easting=camera_easting,
        camera_northing=camera_northing,
        z_tol=z_tol,
        min_wall_width_m=min_wall_width_m,
        min_wall_height_m=min_wall_height_m,
        debug=debug,
    )
    # Return just the best (first) wall id for callers that expect a single string
    return group[0]["wall_id"]


# -----------------------------
# NEW: co-planar facade group
# -----------------------------

def choose_facade_wall_group(
    lod2_path: str,
    camera_easting: float,
    camera_northing: float,
    z_tol: float = 0.20,
    min_wall_width_m: float = 1.0,
    min_wall_height_m: float = 1.0,
    normal_dot_threshold: float = 0.97,    # ~14° angular tolerance between normals
    collinear_dist_threshold: float = 0.50, # metres: how far a wall's ground midpoint
                                            # may sit from the best wall's ground line
    debug: bool = False,
) -> List[dict]:
    """
    Returns an ordered list of wall dicts that form the co-planar facade group
    best facing the camera.

    Each dict contains:
        wall_id     : gml:id string
        seg         : ((x0,y0), (x1,y1))  ground edge in model CRS
        height_m    : float
        along_m     : float  position along the facade line (for sorting left→right)
        width_m     : float
        normal      : (nx, ny) unit normal

    The list is sorted by `along_m` (left-to-right when facing the facade).

    Algorithm
    ---------
    1. Score all walls exactly as before; pick the *best* wall (anchor).
    2. Build the anchor's infinite ground line and its unit normal.
    3. For every other eligible wall:
       a. Normal alignment: dot(wall_normal, anchor_normal) >= normal_dot_threshold
       b. Ground-line collinearity: perpendicular distance from wall ground-midpoint
          to anchor ground line <= collinear_dist_threshold
       c. Faces camera: dot(camera_vec, wall_normal) > 0
    4. Sort the group by projection onto the anchor ground-line direction.
    """
    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    root = etree.parse(lod2_path, parser).getroot()

    walls = root.xpath("//*[local-name()='WallSurface']")
    if not walls:
        raise RuntimeError("No WallSurface elements found")

    # ── Step 1: score every eligible wall (same logic as before) ──────────────
    scored = []   # (dot, dist, wall_id, seg, height_m)
    eligible_count = 0

    if debug:
        print("\n--- wall scoring ---")

    for w in walls:
        wall_id = w.get(GML_ID)
        if not wall_id:
            continue
        seg = extract_ground_edge_segment_xy_from_wall(w, z_tol=z_tol)
        if seg is None:
            continue
        (x0, y0), (x1, y1) = seg
        wall_width_m = float(np.hypot(x1 - x0, y1 - y0))
        wall_height_m = extract_wall_height_m_from_wall(w)
        if wall_height_m is None:
            continue
        if wall_width_m < min_wall_width_m or wall_height_m < min_wall_height_m:
            if debug:
                print(f"  Skipped (too small): {wall_id}  w={wall_width_m:.2f}m h={wall_height_m:.2f}m")
            continue

        eligible_count += 1
        normal = _wall_normal_xy(seg)
        if normal is None:
            continue
        nx, ny = normal

        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        dx = center_x - camera_easting
        dy = center_y - camera_northing
        dot = dx * nx + dy * ny
        distance = float(np.hypot(dx, dy))

        if debug:
            print(f"\nWall: {wall_id}")
            print(f"  Width: {wall_width_m:.2f}m  Height: {wall_height_m:.2f}m")
            print(f"  Center: ({center_x:.2f}, {center_y:.2f})  Dist: {distance:.2f}m  Dot: {dot:.4f}")

        scored.append((dot, distance, wall_id, seg, wall_height_m, wall_width_m, (nx, ny)))

    if not scored:
        if eligible_count == 0:
            raise RuntimeError(
                f"No WallSurface met the minimum size constraints "
                f"(width >= {min_wall_width_m:.2f}m and height >= {min_wall_height_m:.2f}m)."
            )
        raise RuntimeError("Could not score any walls (no valid segments extracted).")

    # Best wall: highest dot; tie-break by shortest distance
    scored.sort(key=lambda s: (-s[0], s[1]))
    best = scored[0]
    best_dot, best_dist, best_id, best_seg, best_h, best_w, best_normal = best

    if debug:
        print(f"\nAnchor wall: {best_id}  dot={best_dot:.4f}  dist={best_dist:.2f}m")

    # ── Step 2: anchor geometry ────────────────────────────────────────────────
    (ax0, ay0), (ax1, ay1) = best_seg
    anx, any_ = best_normal

    # Unit vector along the anchor ground line (left-to-right direction)
    along_dx = ax1 - ax0
    along_dy = ay1 - ay0
    along_len = float(np.hypot(along_dx, along_dy))
    along_ux = along_dx / along_len
    along_uy = along_dy / along_len

    # ── Step 3: group co-planar walls ─────────────────────────────────────────
    group_entries = []

    for dot, dist, wall_id, seg, height_m, width_m, normal in scored:
        nx, ny = normal
        (x0, y0), (x1, y1) = seg
        mid_x = (x0 + x1) / 2.0
        mid_y = (y0 + y1) / 2.0

        # a) Normal alignment
        normal_alignment = abs(nx * anx + ny * any_)
        if normal_alignment < normal_dot_threshold:
            if debug:
                print(f"  Excluded (normal mismatch {normal_alignment:.3f}): {wall_id}")
            continue

        # b) Ground-line collinearity: perpendicular distance from wall midpoint
        #    to the anchor's infinite ground line
        perp_dist = _point_to_line_dist(mid_x, mid_y, ax0, ay0, ax1, ay1)
        if perp_dist > collinear_dist_threshold:
            if debug:
                print(f"  Excluded (not collinear, perp_dist={perp_dist:.3f}m): {wall_id}")
            continue

        # c) Must face the camera (dot product with camera direction > 0)
        cam_dx = mid_x - camera_easting
        cam_dy = mid_y - camera_northing
        cam_dot = cam_dx * nx + cam_dy * ny
        if cam_dot <= 0:
            if debug:
                print(f"  Excluded (faces away from camera): {wall_id}")
            continue

        # Position along the facade line (for left-to-right ordering)
        along_m = (mid_x - ax0) * along_ux + (mid_y - ay0) * along_uy

        group_entries.append({
            "wall_id":  wall_id,
            "seg":      seg,
            "height_m": height_m,
            "width_m":  width_m,
            "normal":   normal,
            "along_m":  along_m,
        })

        if debug:
            print(f"  Included in group: {wall_id}  along={along_m:.2f}m  perp={perp_dist:.3f}m")

    # Sort left → right along the facade
    group_entries.sort(key=lambda e: e["along_m"])

    if not group_entries:
        # Fallback: just the best wall
        group_entries = [{
            "wall_id":  best_id,
            "seg":      best_seg,
            "height_m": best_h,
            "width_m":  best_w,
            "normal":   best_normal,
            "along_m":  0.0,
        }]

    if debug:
        ids = [e["wall_id"] for e in group_entries]
        print(f"\nFacade group ({len(group_entries)} walls): {ids}")

    return group_entries


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Choose the WallSurface gml:id(s) that face the camera point. "
            "Returns a JSON list of wall ids forming the co-planar facade group, "
            "or a single id when --single is set (backward-compatible mode)."
        )
    )
    ap.add_argument("--lod2", required=True, help="LoD2 CityGML .gml")
    ap.add_argument("--image", default=None, help="Georeferenced image path (EXIF GPS or georeferenced raster)")
    ap.add_argument("--image_name", default=None, help="Fallback filename containing UTM coords like '32U 565770 5940383'")
    ap.add_argument("--camera_source", default="auto", choices=["auto", "exif", "raster", "filename"])
    ap.add_argument("--z_tol", type=float, default=0.20)
    ap.add_argument("--min_wall_width", type=float, default=2.0)
    ap.add_argument("--min_wall_height", type=float, default=2.0)
    ap.add_argument("--normal_dot_threshold", type=float, default=0.97,
                    help="Minimum dot product between wall normals for co-planar grouping (~14° tolerance)")
    ap.add_argument("--collinear_dist_threshold", type=float, default=2.0,
                    help="Max perpendicular distance (m) from wall ground midpoint to anchor ground line")
    ap.add_argument("--single", action="store_true",
                    help="Legacy mode: print only the best single wall id (last line of stdout)")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    cam_e, cam_n, src = get_camera_point_en(
        lod2_path=args.lod2,
        image_path=args.image,
        image_name_fallback=args.image_name,
        source_mode=args.camera_source,
        debug=args.debug,
    )

    if args.debug:
        print(f"\nCamera point source: {src}")
        print(f"Camera EN in LoD2 CRS: ({cam_e:.3f}, {cam_n:.3f})")

    group = choose_facade_wall_group(
        lod2_path=args.lod2,
        camera_easting=cam_e,
        camera_northing=cam_n,
        z_tol=args.z_tol,
        min_wall_width_m=args.min_wall_width,
        min_wall_height_m=args.min_wall_height,
        normal_dot_threshold=args.normal_dot_threshold,
        collinear_dist_threshold=args.collinear_dist_threshold,
        debug=args.debug,
    )

    if args.single:
        # Backward-compatible: just print the best wall id on the last line
        print(group[0]["wall_id"])
    else:
        # Print JSON array of {wall_id, width_m} objects — last line is the JSON (runner-friendly).
        # width_m is the real physical width of each wall segment so vlm_simple.py can
        # split the composite UV space proportionally instead of assuming equal widths.
        payload = [
            {"wall_id": e["wall_id"], "width_m": round(float(e["width_m"]), 4)}
            for e in group
        ]
        print(json.dumps(payload))


if __name__ == "__main__":
    main()