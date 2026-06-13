import numpy as np
from PIL import Image
import glob
import os

# This script computes the IoU between SAM's predictions and the GT, and the FRDS between LLM's predictions and the GT.

sam_dir = "evaluation/sam"
llm_dir = "evaluation/llm"
gt_dir = "evaluation/gt_fixed"
output_file = "evaluation/results.txt"

ious = []
mfrds = []

with open(output_file, "w") as f:
    for gt_path in sorted(glob.glob(os.path.join(gt_dir, "*_gt_mask.png"))):
        base = os.path.basename(gt_path).replace("_gt_mask.png", "")  # "basel_000003_mv0"

        sam_path = os.path.join(sam_dir, base + "_sam_mask.png")
        llm_path = os.path.join(llm_dir, base + "_clean_mask.png")


        gt = np.array(Image.open(gt_path))
        sam = np.array(Image.open(sam_path))
        llm = np.array(Image.open(llm_path))

        # GT
        gt_r, gt_g, gt_b = gt[:,:,0], gt[:,:,1], gt[:,:,2]
        gt_windows = (gt_r == 255) & (gt_g == 0) & (gt_b == 0)
        gt_doors = (gt_r == 0) & (gt_g == 255) & (gt_b == 0)
        gt_facade = (gt_r == 0) & (gt_g == 0) & (gt_b == 255)
        gt_binary = (gt_windows | gt_doors | gt_facade).astype(np.float64)

        # SAM
        sam_r, sam_g, sam_b = sam[:,:,0], sam[:,:,1], sam[:,:,2]
        sam_windows = (sam_r == 255) & (sam_g == 0) & (sam_b == 0)
        sam_doors = (sam_r == 0) & (sam_g == 255) & (sam_b == 0)
        sam_facade = (sam_r == 0) & (sam_g == 0) & (sam_b == 255)
        sam_binary = (sam_windows | sam_doors | sam_facade).astype(np.float64)

        # LLM
        llm_r, llm_g, llm_b = llm[:,:,0], llm[:,:,1], llm[:,:,2]
        llm_windows = (llm_r == 255) & (llm_g == 0) & (llm_b == 0)
        llm_doors = (llm_r == 0) & (llm_g == 255) & (llm_b == 0)
        llm_facade = (llm_r == 0) & (llm_g == 0) & (llm_b == 255)
        llm_binary = (llm_windows | llm_doors | llm_facade).astype(np.float64)

        # LLM vs GT — FRDS
        a = np.sum(llm_binary * gt_binary)
        b = np.sum(llm_binary ** 2) + np.sum(gt_binary ** 2)
        frds = 2.0 * a / b 

        # SAM vs GT — IoU
        d = np.sum(sam_binary * gt_binary)
        c = np.sum((sam_binary + gt_binary) > 0)
        iou = d / c 

        ious.append(iou)
        mfrds.append(frds)

        line = f"{base}: SAM vs GT IoU = {iou:.4f}, LLM vs GT FRDS = {frds:.4f}"
        print(line)
        f.write(line + "\n")


    miou_line = f"\nSAM vs GT mIoU = {np.mean(ious):.4f}"
    print(miou_line)
    f.write(miou_line + "\n")

    mFRDS_line = f"\nLLM vs GT mFRDS = {np.mean([mfrds]):.4f}"
    print(mFRDS_line)
    f.write(mFRDS_line + "\n")
print(f"\nResults saved to {output_file}")