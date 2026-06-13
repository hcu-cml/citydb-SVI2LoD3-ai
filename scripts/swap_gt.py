from PIL import Image
import os
import glob

# This script converts the ground truth masks from the original color scheme to the new one used by SAM and LLM.
os.makedirs("/home/mokads/Desktop/Lod3/evaluation/gt_fixed/", exist_ok=True)

for image_path in glob.glob("/home/mokads/Desktop/Lod3/evaluation/gt/*.png"):
    image = Image.open(image_path)
    img= image.convert("RGB")
    # img = Image.open("/home/mokads/Desktop/Lod3/evaluation/gt/basel_000009_mv0.png").convert("RGB")

    mapping = {
        (0, 0, 128): (255, 0, 0),    # blue to red
        (128, 0, 0): (0, 0, 255),    # red to blue
        (128, 128, 0): (0, 255, 0),  # yellow to green
    }

    out = Image.new("RGB", img.size, (0, 0, 0))
    pix_in = img.load()
    pix_out = out.load()

    for y in range(img.height):
        for x in range(img.width):
            pix_out[x, y] = mapping.get(pix_in[x, y], (0, 0, 0))
    filename = os.path.splitext(os.path.basename(image_path))[0]
    out.save(os.path.join("/home/mokads/Desktop/Lod3/evaluation/gt_test/", filename + "_gt_mask.png"))
    # out.save("/home/mokads/Desktop/Lod3/evaluation/gt_fixed/basel_000009_mv0.png")