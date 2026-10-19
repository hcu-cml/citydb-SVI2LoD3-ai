
# import torch
# import lpips
# from PIL import Image
# import torchvision.transforms as transforms
# import open_clip
# from torchmetrics.image import StructuralSimilarityIndexMeasure
#
# from transformers import AutoImageProcessor, AutoModel


# device = "cuda" if torch.cuda.is_available() else "cpu"

# # load CLIP model
# model, _, preprocess = open_clip.create_model_and_transforms(
#     "ViT-L-14",
#     pretrained="openai"
# )
# model = model.to(device)
# model.eval()
#
# # load images
# # img3 = preprocess(Image.open("/home/luarzou/Downloads/evaluation/llm/basel_000005_mv0_mask_llm.png").convert("RGB")).unsqueeze(0).to(device)
# img1 = preprocess(Image.open("/home/luarzou/Downloads/evaluation/sam/basel_000005_mv0_mask.png").convert("RGB")).unsqueeze(0).to(device)
# img2 = preprocess(Image.open("/home/luarzou/Downloads/evaluation/gt_fixed/basel_000005_mv0.png").convert("RGB")).unsqueeze(0).to(device)
#
#
# # compute embeddings
# with torch.no_grad():
#     feat1 = model.encode_image(img1)
#     feat2 = model.encode_image(img2)
#
# # normalize embeddings
# feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
# feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
#
# # cosine distance
# clip_distance = 1 - (feat1 * feat2).sum(dim=-1)
#
# print("CLIP distance:", clip_distance.item())
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # load DINOv2 model
# model_name = "facebook/dinov2-base"
#
# processor = AutoImageProcessor.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name).to(device)
# model.eval()
#
# # load images
# img1 = Image.open("/home/luarzou/Downloads/evaluation/llm/basel_000004_mv0_mask_llm.png").convert("RGB")
# img2 = Image.open("/home/luarzou/Downloads/evaluation/gt_fixed/basel_000004_mv0.png").convert("RGB")
#
# # preprocess
# inputs1 = processor(images=img1, return_tensors="pt").to(device)
# inputs2 = processor(images=img2, return_tensors="pt").to(device)
#
# # compute embeddings
# with torch.no_grad():
#     feat1 = model(**inputs1).last_hidden_state[:,0]  # CLS token
#     feat2 = model(**inputs2).last_hidden_state[:,0]
#
# # normalize
# feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
# feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
#
# # cosine distance
# distance = 1 - (feat1 * feat2).sum(dim=-1)
#
# print("DINOv2 feature distance:", distance.item())


import re
import csv
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def extract_gt_key(filename: str) -> str | None:
    """
    Example:
        prague_000002_gt_mask.png -> prague_000002
    """
    m = re.match(r"^(.*)_gt_mask\.[^.]+$", filename)
    return m.group(1) if m else None


def extract_recon_key(filename: str) -> str | None:
    """
    Example:
        prague_000002_clean_mask.png -> prague_000002
    """
    m = re.match(r"^(.*)_clean_mask\.[^.]+$", filename)
    return m.group(1) if m else None


def get_matching_pairs(gt_dir: str, recon_dir: str):
    gt_dir = Path(gt_dir)
    recon_dir = Path(recon_dir)

    gt_map = {}
    recon_map = {}

    for p in gt_dir.iterdir():
        if p.is_file() and is_image_file(p):
            key = extract_gt_key(p.name)
            if key is not None:
                gt_map[key] = p

    for p in recon_dir.iterdir():
        if p.is_file() and is_image_file(p):
            key = extract_recon_key(p.name)
            if key is not None:
                recon_map[key] = p

    common_keys = sorted(set(gt_map.keys()) & set(recon_map.keys()))
    missing_in_recon = sorted(set(gt_map.keys()) - set(recon_map.keys()))
    missing_in_gt = sorted(set(recon_map.keys()) - set(gt_map.keys()))

    pairs = [(key, gt_map[key], recon_map[key]) for key in common_keys]
    return pairs, missing_in_recon, missing_in_gt


def compute_dinov2_distances(
    gt_dir: str,
    recon_dir: str,
    model_name: str = "facebook/dinov2-base",
    batch_size: int = 16,
    use_patch_distance: bool = False,
    save_csv_path: str | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    pairs, missing_in_recon, missing_in_gt = get_matching_pairs(gt_dir, recon_dir)

    print(f"Matched image pairs: {len(pairs)}")
    if missing_in_recon:
        print(f"Missing reconstructions for {len(missing_in_recon)} GT images")
    if missing_in_gt:
        print(f"Missing GT images for {len(missing_in_gt)} reconstructions")

    if len(pairs) == 0:
        raise ValueError("No matching image pairs found.")

    results = []

    for start in tqdm(range(0, len(pairs), batch_size), desc="Computing DINOv2 distances"):
        batch_pairs = pairs[start:start + batch_size]

        gt_images = [load_rgb_image(gt_path) for _, gt_path, _ in batch_pairs]
        recon_images = [load_rgb_image(recon_path) for _, _, recon_path in batch_pairs]

        gt_inputs = processor(images=gt_images, return_tensors="pt").to(device)
        recon_inputs = processor(images=recon_images, return_tensors="pt").to(device)

        with torch.no_grad():
            gt_outputs = model(**gt_inputs).last_hidden_state
            recon_outputs = model(**recon_inputs).last_hidden_state

            if use_patch_distance:
                gt_feat = gt_outputs[:, 1:, :]      # patch tokens
                recon_feat = recon_outputs[:, 1:, :]

                gt_feat = gt_feat / gt_feat.norm(dim=-1, keepdim=True)
                recon_feat = recon_feat / recon_feat.norm(dim=-1, keepdim=True)

                distances = 1.0 - (gt_feat * recon_feat).sum(dim=-1)
                distances = distances.mean(dim=-1)
            else:
                gt_feat = gt_outputs[:, 0, :]       # CLS token
                recon_feat = recon_outputs[:, 0, :]

                gt_feat = gt_feat / gt_feat.norm(dim=-1, keepdim=True)
                recon_feat = recon_feat / recon_feat.norm(dim=-1, keepdim=True)

                distances = 1.0 - (gt_feat * recon_feat).sum(dim=-1)

        for (key, gt_path, recon_path), dist in zip(batch_pairs, distances.cpu().tolist()):
            results.append({
                "key": key,
                "ground_truth_filename": gt_path.name,
                "reconstruction_filename": recon_path.name,
                "ground_truth_path": str(gt_path),
                "reconstruction_path": str(recon_path),
                "dinov2_distance": float(dist),
            })

    mean_distance = sum(r["dinov2_distance"] for r in results) / len(results)

    print("\n--- Results ---")
    print(f"Images evaluated: {len(results)}")
    print(f"Mean DINOv2 distance: {mean_distance:.6f}")
    print(f"Distance type: {'patch-token' if use_patch_distance else 'CLS-token'}")

    if save_csv_path is not None:
        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "key",
                    "ground_truth_filename",
                    "reconstruction_filename",
                    "ground_truth_path",
                    "reconstruction_path",
                    "dinov2_distance",
                ],
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"Saved per-image results to: {save_csv_path}")

    return results, mean_distance


if __name__ == "__main__":
    gt_dir = "/home/luarzou/projects/aussenraume/imageToLOD3/LoD3-Reconstruction/evaluation/gt_fixed/"
    recon_dir = "/home/luarzou/projects/aussenraume/imageToLOD3/LoD3-Reconstruction/evaluation/llm/"

    results, mean_distance = compute_dinov2_distances(
        gt_dir=gt_dir,
        recon_dir=recon_dir,
        model_name="facebook/dinov2-base",
        batch_size=16,
        use_patch_distance=True,   #TRUE for object/layout sensitivity
        save_csv_path="dinov2_results.csv",
    )
