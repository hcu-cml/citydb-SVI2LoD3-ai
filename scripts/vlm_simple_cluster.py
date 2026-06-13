from __future__ import annotations
import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw
from openai import OpenAI
import cv2


@dataclass(frozen=True)
class Box2D:
    typ: str
    x0: float
    y0: float
    x1: float
    y1: float


# Image processing helpers
def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

# Isolat specific color in the image
def color_isolation(arr: np.ndarray, rgb: Tuple[int, int, int], tol: int = 40) -> np.ndarray:
    t = np.array(rgb, dtype=np.int16).reshape(1, 1, 3)
    d = np.abs(arr.astype(np.int16) - t)
    return (d[:, :, 0] <= tol) & (d[:, :, 1] <= tol) & (d[:, :, 2] <= tol)

# Find largest connected component in binary mask
def largest_connected_components(mask_u8: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    lab = 1 + int(np.argmax(areas))
    return (labels == lab).astype(np.uint8)

# Identify all separate mask regions
def seperate_component_bboxes(mask_u8: np.ndarray, min_area_px: int) -> List[List[int]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out: List[List[int]] = []
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x = int(stats[lab, cv2.CC_STAT_LEFT])
        y = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        if w <= 1 or h <= 1:
            continue
        out.append([x, y, x + w, y + h])
    out.sort(key=lambda b: (b[1], b[0]))
    return out

# Find edges of openings in color based mask
def compute_edges(pts_xy: np.ndarray) -> List[List[int]]:
    pts = np.asarray(pts_xy, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    out = np.array([tl, tr, br, bl], dtype=np.float32)
    return [[int(round(x)), int(round(y))] for x, y in out]

# Find coordinates of all pixels in the mask and detect its edges
def facade_edges(main_mask_u8: np.ndarray) -> List[List[int]]:
    ys, xs = np.where(main_mask_u8 > 0)
    if len(xs) < 4:
        H, W = main_mask_u8.shape
        return [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    hull = cv2.convexHull(pts)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    quad = compute_edges(box)
    H, W = main_mask_u8.shape
    return [[max(0, min(W - 1, x)), max(0, min(H - 1, y))] for x, y in quad]



# Extract JSON object from LLM response
def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[-1:]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start: end + 1])
    raise ValueError("No JSON found in text")

# Buffer PIL image as data URL for LLM input
def buffer(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def clip(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else (hi if v > hi else v)


# Keep only boxes that are near the facade, defined as either having their center inside the facade quad or touching the facade.
def keep_boxes_near_facade(
    boxes: List[List[int]],
    facade_quad_px: List[List[int]],
    facade_main_u8: np.ndarray,
    pad_px: int = 5,
) -> List[List[int]]:
    poly = np.array(facade_quad_px, dtype=np.int32)
    k = max(1, int(pad_px))
    kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    facade_loose = cv2.dilate(facade_main_u8.astype(np.uint8), kernel, iterations=1)
    kept: List[List[int]] = []
    H, W = facade_loose.shape
    for x0, y0, x1, y1 in boxes:
        cx = int(round((x0 + x1) * 0.5))
        cy = int(round((y0 + y1) * 0.5))
        center_in_quad = cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0
        xa = max(0, min(W, x0))
        xb = max(0, min(W, x1))
        ya = max(0, min(H, y0))
        yb = max(0, min(H, y1))
        crop = facade_loose[ya:yb, xa:xb]
        touches_loose_facade = crop.size > 0 and np.any(crop > 0)
        if center_in_quad or touches_loose_facade:
            kept.append([x0, y0, x1, y1])
    kept.sort(key=lambda b: (b[1], b[0]))
    return kept

# Parse class-colored SAM mask and returns original-image-space openings in pixel coordinates
def parse_sam_mask(mask_path: str, tol: int = 40, min_cc_px: int = 150) -> Dict[str, Any]:
    im = Image.open(mask_path).convert("RGB")
    arr = np.array(im)
    H, W = arr.shape[:2]
    m_red   = color_isolation(arr, (255,   0,   0), tol=tol)
    m_green = color_isolation(arr, (  0, 255,   0), tol=tol)
    m_blue  = color_isolation(arr, (  0,   0, 255), tol=tol)

    support = (m_blue | m_red | m_green).astype(np.uint8)
    facade_main = largest_connected_components(support)
    facade_quad_px = facade_edges(facade_main)

    win_boxes_raw  = seperate_component_bboxes(m_red.astype(np.uint8),   min_area_px=min_cc_px)
    door_boxes_raw = seperate_component_bboxes(m_green.astype(np.uint8), min_area_px=min_cc_px)

    win_boxes  = keep_boxes_near_facade(win_boxes_raw,  facade_quad_px, facade_main, pad_px=150)
    door_boxes = keep_boxes_near_facade(door_boxes_raw, facade_quad_px, facade_main, pad_px=150)

    openings: List[Dict[str, Any]] = []
    for idx, bb in enumerate(win_boxes, 1):
        openings.append({"id": f"w_{idx:02d}", "type": "Window", "bbox_px": bb})
    for idx, bb in enumerate(door_boxes, 1):
        openings.append({"id": f"d_{idx:02d}", "type": "Door",   "bbox_px": bb})
    openings.sort(key=lambda o: (o["bbox_px"][1], o["bbox_px"][0], o["type"]))

    return {
        "image_size":         [W, H],
        "facade_quad_px":     facade_quad_px,
        "candidate_openings": openings,
        "debug": {
            "num_windows": len(win_boxes),
            "num_doors":   len(door_boxes),
        },
    }


# LLM reasoning prompt and API call 
def call_vlm(
    *,
    rgb_original: Image.Image,
    mask_original: Image.Image,
    candidate_payload: Dict[str, Any],
    model: str,
    api_key: Optional[str],
    base_url: Optional[str],
    timeout_s: int = 120,
) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("Missing API key for LLM refinement")

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    W, H = candidate_payload["image_size"]
    candidates_json = json.dumps(candidate_payload["candidate_openings"], ensure_ascii=False)
    prompt = f"""
You are a facade-geometry refinement assistant for LoD3 reconstruction.

You will receive 2 images:
1) original RGB facade photo
2) original SAM semantic mask (blue facade, red windows, green doors)

Task:
- Compare RGB and SAM mask together.
- Remove noise (trees, neighboring buildings, signs, cars).
- Remove roof windows from the mask.
- Do not remove facade windows/doors that are clearly detected by sam mask.
- ALways double check for overlaps, openings should NEVER overlap.
- windows and doors should be row/column axis aligned and rectangles.
- Align all row of windows of each row and all colomn of windows of each colomn.
- Use class labels only: Window, Door.
- Keep the mask objects fully inside the facade rectangle.

Work in RECTIFIED coordinates only.
Rectified image size: width={W}, height={H}
Facade rectangle: x:[0,{W-1}], y:[0,{H-1}]

Candidate boxes from deterministic parsing (rectified coordinates):
{candidates_json}

Return STRICT JSON only with this schema:
{{
  "facade_bbox": {{"x0":0, "y0":0, "x1":{W-1}, "y1":{H-1}}},
  "openings": [
    {{"type":"Window"|"Door", "x0":int, "y0":int, "x1":int, "y1":int, "reason":"short"}}
  ],
  "confidence": "high|medium|low",
  "notes": ["optional short notes"]
}}

Rules:
- x0 < x1 and y0 < y1
- axis-aligned only
- no duplicates
- prefer consistent rows/columns for windows when supported by evidence
- do not invent openings that are not visible
""".strip()

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": buffer(rgb_original)}},
                {"type": "image_url", "image_url": {"url": buffer(mask_original)}},
            ],
        }
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0,
        timeout=timeout_s,
        response_format={"type": "json_object"},
    )
    txt = resp.choices[0].message.content or "{}"
    out = extract_json(txt)
    out["_raw"] = txt
    return out

# Validation & UV conversion
def validate_image_openings(
    payload: Dict[str, Any],
    img_w: int,
    img_h: int,
    min_area_px: int,
    max_boxes: int = 200,
) -> List[Dict[str, Any]]:
    raw = payload.get("openings", [])
    if not isinstance(raw, list):
        raw = []
    out: List[Dict[str, Any]] = []
    for it in raw[:max_boxes]:
        if not isinstance(it, dict):
            continue
        typ = str(it.get("type", "")).strip().title()
        if typ not in ("Window", "Door"):
            continue
        try:
            x0 = int(round(float(it["x0"])))
            y0 = int(round(float(it["y0"])))
            x1 = int(round(float(it["x1"])))
            y1 = int(round(float(it["y1"])))
        except Exception:
            continue
        x0 = clip(x0, 0, img_w - 1)
        x1 = clip(x1, 0, img_w - 1)
        y0 = clip(y0, 0, img_h - 1)
        y1 = clip(y1, 0, img_h - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        if (x1 - x0) * (y1 - y0) < min_area_px:
            continue
        out.append({"type": typ, "bbox_px": [x0, y0, x1, y1]})
    out.sort(key=lambda o: (o["bbox_px"][1], o["bbox_px"][0], o["type"]))
    return out

# Compute IoU between two boxes
def iou(a: Box2D, b: Box2D) -> float:
    x0 = max(a.x0, b.x0); y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1); y1 = min(a.y1, b.y1)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    aa = (a.x1 - a.x0) * (a.y1 - a.y0)
    bb = (b.x1 - b.x0) * (b.y1 - b.y0)
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def postprocess_uv_boxes(boxes: List[Box2D], min_area: float = 0.0003, nmsiou: float = 0.90) -> List[Box2D]:
    filt = [b for b in boxes if (b.x1 - b.x0) * (b.y1 - b.y0) >= min_area]
    kept: List[Box2D] = []
    for b in filt:
        drop = any(b.typ == k.typ and iou(b, k) > nmsiou for k in kept)
        if not drop:
            kept.append(b)
    kept.sort(key=lambda z: (z.y0, z.x0, z.typ))
    return kept

# Map image-pixel bounding boxes to UV coordinates on the full composite facade.
# U=0 is the left edge, U=1 the right edge, V=0 is the bottom edge (when v_origin='bottom'), V=1 the top.
def image_openings_to_uv(
    openings_px: List[Dict[str, Any]],
    facade_quad_px: List[List[int]],
    img_w: int,
    img_h: int,
    *,
    v_origin: str = "bottom",
) -> List[Box2D]:
    xs = [p[0] for p in facade_quad_px]
    ys = [p[1] for p in facade_quad_px]
    fac_x0, fac_x1 = float(min(xs)), float(max(xs))
    fac_y0, fac_y1 = float(min(ys)), float(max(ys))
    fac_w = max(1.0, fac_x1 - fac_x0)
    fac_h = max(1.0, fac_y1 - fac_y0)

    out: List[Box2D] = []
    for it in openings_px:
        typ = str(it["type"])
        x0, y0, x1, y1 = map(float, it["bbox_px"])

        u0 = clamp01((x0 - fac_x0) / fac_w)
        u1 = clamp01((x1 - fac_x0) / fac_w)

        if v_origin == "bottom":
            v0 = clamp01((fac_y1 - y1) / fac_h)
            v1 = clamp01((fac_y1 - y0) / fac_h)
        else:
            v0 = clamp01((y0 - fac_y0) / fac_h)
            v1 = clamp01((y1 - fac_y0) / fac_h)

        if u1 <= u0 or v1 <= v0:
            continue
        out.append(Box2D(typ=typ, x0=u0, y0=v0, x1=u1, y1=v1))

    out.sort(key=lambda b: (b.y0, b.x0, b.typ))
    return out


# Multi-WallSurface UV assignment
def assign_openings_to_walls(
    uv_boxes: List[Box2D],
    facade_group: List[dict],
) -> Dict[str, List[Box2D]]:
    if not facade_group:
        return {}

    # Build cumulative U spans for each wall in the composite [0,1] space
    total_width = sum(e["width_m"] for e in facade_group)
    if total_width < 1e-6:
        # Degenerate: put everything on the first wall unchanged
        return {facade_group[0]["wall_id"]: list(uv_boxes)}

    # wall_spans[i] = (u_start, u_end) in composite UV space
    wall_spans: List[Tuple[float, float]] = []
    cursor = 0.0
    for entry in facade_group:
        frac = entry["width_m"] / total_width
        wall_spans.append((cursor, cursor + frac))
        cursor += frac

    # Assign each opening to the wall with maximum U-overlap
    result: Dict[str, List[Box2D]] = {e["wall_id"]: [] for e in facade_group}

    for box in uv_boxes:
        best_wall_idx = 0
        best_overlap = -1.0

        for i, (ws, we) in enumerate(wall_spans):
            overlap = max(0.0, min(box.x1, we) - max(box.x0, ws))
            if overlap > best_overlap:
                best_overlap = overlap
                best_wall_idx = i

        # Re-normalise U into this wall's local space
        ws, we = wall_spans[best_wall_idx]
        wall_span_width = max(we - ws, 1e-9)

        local_u0 = clamp01((box.x0 - ws) / wall_span_width)
        local_u1 = clamp01((box.x1 - ws) / wall_span_width)

        if local_u1 <= local_u0:
            # Opening was almost entirely outside this wall — skip
            continue

        wall_id = facade_group[best_wall_idx]["wall_id"]
        result[wall_id].append(Box2D(typ=box.typ, x0=local_u0, y0=box.y0, x1=local_u1, y1=box.y1))

    return result


# Debug visualisation for the refined mask and openings
def save_clean_mask(
    out_path: str,
    image_size_wh: Tuple[int, int],
    facade_quad_px: List[List[int]],
    openings_px: List[Dict[str, Any]],
) -> None:
    W, H = image_size_wh
    out = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(out)
    draw.polygon([tuple(p) for p in facade_quad_px], fill=(0, 0, 255))
    for it in openings_px:
        x0, y0, x1, y1 = map(int, it["bbox_px"])
        color = (255, 0, 0) if it["type"] == "Window" else (0, 255, 0)
        draw.rectangle([x0, y0, x1, y1], fill=color)
    out.save(out_path)


#####External script helpers

# Wall selection script
def run_choose_wall_group(
    choose_py: str,
    lod2_path: str,
    camera_image: Optional[str] = None,
    image_name_fallback: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> List[dict]:
    cmd = [sys.executable, choose_py, "--lod2", lod2_path]
    if camera_image:
        cmd += ["--image", camera_image]
    elif image_name_fallback:
        cmd += ["--image_name", image_name_fallback]
    else:
        raise RuntimeError("Need camera_image or image_name_fallback for wall selection")
    if extra_args:
        cmd += extra_args

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"choose_wall failed\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")

    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("choose_wall returned no output")

    last = lines[-1]

    # Current format: JSON array of {wall_id, width_m} objects
    try:
        parsed = json.loads(last)
        if (
            isinstance(parsed, list)
            and parsed
            and isinstance(parsed[0], dict)
            and "wall_id" in parsed[0]
        ):
            return [
                {"wall_id": str(e["wall_id"]), "width_m": float(e.get("width_m", 1.0))}
                for e in parsed
            ]
    except Exception:
        pass

    # Legacy format: JSON array of plain strings (ids only) — equal widths
    try:
        parsed = json.loads(last)
        if isinstance(parsed, list) and all(isinstance(i, str) for i in parsed):
            return [{"wall_id": wid, "width_m": 1.0} for wid in parsed]
    except Exception:
        pass

    # Oldest legacy: single wall id string on the last line
    if re.match(r"^[A-Za-z_][A-Za-z0-9_.-]*$", last):
        return [{"wall_id": last, "width_m": 1.0}]

    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]*", p.stdout)
    if not toks:
        raise RuntimeError(f"Could not parse wall id(s) from choose_wall output:\n{p.stdout}")
    return [{"wall_id": toks[-1], "width_m": 1.0}]

# LoD2 to LoD3 reconstruction script
def run_compile(compile_py: str, lod2_path: str, patch_path: str, out_gml: str) -> None:
    cmd = [sys.executable, compile_py, "--lod2", lod2_path, "--patch", patch_path, "--out", out_gml]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"compile failed\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")

# GML validator script
def run_validator(validate_py: str, gml_path: str) -> bool:
    validate_py_abs = os.path.abspath(validate_py)        # resolve relative --validator arg first
    validator_dir   = os.path.dirname(validate_py_abs)    # then derive dir from the absolute path
    gml_path_abs    = os.path.abspath(gml_path)
    cmd = [sys.executable, validate_py_abs, gml_path_abs] # use absolute path to script
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=validator_dir)

    # print(f"[validate debug] validator_dir : {validator_dir}")
    # print(f"[validate debug] gml_path_abs  : {gml_path_abs}")
    # print(f"[validate debug] returncode    : {p.returncode}")
    # print(f"[validate debug] stdout        : {p.stdout!r}")
    # print(f"[validate debug] stderr        : {p.stderr!r}")

    if "Document is valid." in p.stdout:
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lod2",       required=True, help="LoD2 building .gml/.xml")
    ap.add_argument("--rgb_image",  required=True, help="Facade RGB image")
    ap.add_argument("--mask_image", required=True, help="SAM semantic mask (blue/red/green)")
    ap.add_argument("--out_dir",    required=True)

    ap.add_argument("--choose_wall_py",      default=None)
    ap.add_argument("--compile_py",          default=None)
    ap.add_argument("--validator",      default=None)
    ap.add_argument("--image_name_for_wall", default=None)
    ap.add_argument("--target_surface_id",   default=None,
                    help="Single wall id override (bypasses choose_wall). "
                         "For multi-wall override use --target_surface_ids.")
    ap.add_argument("--target_surface_ids",  default=None,
                    help="Comma-separated list of wall ids to use as the facade group "
                         "(ordered left-to-right). Bypasses choose_wall.")

    ap.add_argument("--tol",         type=int,   default=40)
    ap.add_argument("--min_cc_px",   type=int,   default=700)
    ap.add_argument("--min_area_uv", type=float, default=0.0003)
    ap.add_argument("--nmsiou",      type=float, default=0.90)
    ap.add_argument("--offset_m",    type=float, default=1.0)
    ap.add_argument("--v_origin",    choices=["bottom", "top"], default="bottom")

    ap.add_argument("--llm_model",       choices=["gpt-5.1", "qwen3-vl-30b-a3b-instruct"])
    ap.add_argument("--llm_api_key_env", choices=["ACADEMICCLOUD_API_KEY", "OPENAI_API_KEY"])
    ap.add_argument("--llm_base_url",    choices=["https://chat-ai.academiccloud.de/v1", None], default=None)
    ap.add_argument("--llm_timeout_s",   type=int, default=120)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load images 
    rgb_im  = Image.open(args.rgb_image).convert("RGB")
    mask_im = Image.open(args.mask_image).convert("RGB")
    if rgb_im.size != mask_im.size:
        raise RuntimeError(f"RGB/mask size mismatch: rgb={rgb_im.size}, mask={mask_im.size}")
    img_w, img_h = rgb_im.size

    # Deterministic SAM parse (candidates + facade quad)
    parsed = parse_sam_mask(args.mask_image, tol=args.tol, min_cc_px=args.min_cc_px)
    facade_quad_px = parsed["facade_quad_px"]

    # LLM refinement
    api_key = os.environ.get(args.llm_api_key_env)

    llm_payload = call_vlm(
        rgb_original=rgb_im,
        mask_original=mask_im,
        candidate_payload=parsed,
        model=args.llm_model,
        api_key=api_key,
        base_url=args.llm_base_url,
        timeout_s=args.llm_timeout_s,
    )

    with open(os.path.join(args.out_dir, "llm_response_raw.txt"), "w", encoding="utf-8") as f:
        f.write(str(llm_payload.get("_raw", "")))

    # Validate LLM output 
    final_openings_px = validate_image_openings(
        llm_payload, img_w, img_h, min_area_px=args.min_cc_px
    )


    base = os.path.splitext(os.path.basename(args.rgb_image))[0]
    save_clean_mask(
        os.path.join(args.out_dir, base + "_shape_completion.png"),
        (img_w, img_h),
        facade_quad_px,
        final_openings_px,
    )

    # UV conversion (composite facade space)
    uv_boxes = image_openings_to_uv(
        final_openings_px, facade_quad_px, img_w, img_h, v_origin=args.v_origin
    )
    uv_boxes = postprocess_uv_boxes(uv_boxes, min_area=args.min_area_uv, nmsiou=args.nmsiou)


    facade_group: List[dict] = []

    if args.target_surface_ids:
        ids = [s.strip() for s in args.target_surface_ids.split(",") if s.strip()]
        facade_group = [{"wall_id": wid, "width_m": 1.0} for wid in ids]

    elif args.target_surface_id:
        facade_group = [{"wall_id": args.target_surface_id, "width_m": 1.0}]

    elif args.choose_wall_py:
        # run_choose_wall_group now returns List[{"wall_id": str, "width_m": float}]
        # with real physical widths from the GML geometry — used for proportional UV splitting.
        facade_group = run_choose_wall_group(
            choose_py=args.choose_wall_py,
            lod2_path=args.lod2,
            camera_image=args.rgb_image,
            image_name_fallback=args.image_name_for_wall,
        )

    else:
        facade_group = [{"wall_id": "REPLACE_WITH_TARGET_SURFACE_ID", "width_m": 1.0}]

    # Assign openings to individual walls 
    if len(facade_group) == 1:
        # Fast path: single wall, no assignment needed
        per_wall_boxes: Dict[str, List[Box2D]] = {
            facade_group[0]["wall_id"]: uv_boxes
        }
    else:
        per_wall_boxes = assign_openings_to_walls(uv_boxes, facade_group)

    # Build patch JSON
    ops = []
    for entry in facade_group:
        wid = entry["wall_id"]
        wall_openings = per_wall_boxes.get(wid, [])
        if not wall_openings:
            continue
        ops.append({
            "op": "add_openings",
            "target_surface_id": wid,
            "offset_m": float(args.offset_m),
            "openings": [
                {"type": b.typ, "u0": b.x0, "v0": b.y0, "u1": b.x1, "v1": b.y1}
                for b in wall_openings
            ],
        })

    # If no ops produced (e.g. all walls got 0 openings), emit one empty op
    if not ops and facade_group:
        ops.append({
            "op": "add_openings",
            "target_surface_id": facade_group[0]["wall_id"],
            "offset_m": float(args.offset_m),
            "openings": [],
        })

    patch = {"apply_to": "first", "ops": ops}
    patch_path = os.path.join(args.out_dir, "patch-agent.json")
    with open(patch_path, "w", encoding="utf-8") as f:
        json.dump(patch, f, indent=2)

    # Compile to GML and validate
    out_gml = os.path.join(args.out_dir, "LOD3-agent.gml")
    compiled = False
    placeholder = "REPLACE_WITH_TARGET_SURFACE_ID"
    all_ids_known = all(e["wall_id"] != placeholder for e in facade_group)

    if args.compile_py and all_ids_known:
        run_compile(args.compile_py, args.lod2, patch_path, out_gml)
        compiled = True
        if args.validator:
            gml_valid = run_validator(args.validator, out_gml)
            if gml_valid:
                print(f"{out_gml} is a valid LoD3 GML file")
            else:
                print(f"{out_gml} is not a valid LoD3 GML file")


    total_windows = sum(1 for b in uv_boxes if b.typ == "Window")
    total_doors   = sum(1 for b in uv_boxes if b.typ == "Door")

    print("[done]")
    print(f"windows={total_windows}, doors={total_doors}")
    print(f"facade_walls={len(facade_group)}")
    for entry in facade_group:
        wid = entry["wall_id"]
        w_m = entry.get("width_m", 1.0)
        n = len(per_wall_boxes.get(wid, []))
        print(f"  {wid}: {n} opening(s)  width={w_m:.2f}m")
    print(f"clean_mask={os.path.join(args.out_dir, 'clean_mask.png')}")
    print(f"patch_json={patch_path}")
    print(f"lod3_gml={out_gml if compiled else 'SKIPPED (missing compile_py or wall id)'}")


if __name__ == "__main__":
    main()