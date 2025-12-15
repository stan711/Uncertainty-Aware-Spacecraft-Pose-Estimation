import json
from pathlib import Path

def load_shirt_roe1(root="/home/stan711/datasets/shirtv1", domain="lightbox"):
    """
    Load SHIRT ROE1 labels and return a list of dicts with:
      - img_path: full path to image
      - q: quaternion [w, x, y, z]
      - t: translation [tx, ty, tz]
    """
    root = Path(root)
    roe1_dir = root / "roe1"

    # JSON with labels
    json_path = roe1_dir / "roe1.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    # Image directory (lightbox or synthetic)
    img_dir = roe1_dir / domain / "images"

    samples = []
    for entry in data:
        filename = entry["filename"]              # e.g. "img000001.jpg"
        q = entry["q_vbs2tango_true"]             # [w, x, y, z]
        t = entry["r_Vo2To_vbs_true"]             # [tx, ty, tz]

        img_path = img_dir / filename

        samples.append({
            "img_path": str(img_path),
            "q": q,
            "t": t,
        })

    return samples

if __name__ == "__main__":
    samples = load_shirt_roe1()
    print(f"Loaded {len(samples)} ROE1 samples.")
    print("First 3 samples:")
    for s in samples[:3]:
        print(s["img_path"])
        print("  q:", s["q"])
        print("  t:", s["t"])
