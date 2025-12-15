import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class ShirtRoeDataset(Dataset):
    """
    SHIRT ROE dataset (ROE1 or ROE2, lightbox or synthetic).

    Returns a dict with:
      - image:  CxHxW tensor
      - q_gt:   quaternion [w, x, y, z]
      - t_gt:   translation [tx, ty, tz]
      - idx:    frame index (int)
    """
    def __init__(
        self,
        root="/home/stan711/datasets/shirtv1",
        traj="roe1",          # 'roe1' or 'roe2'
        domain="lightbox",    # 'lightbox' or 'synthetic'
        transform=None,
    ):
        super().__init__()
        self.root = Path(root)
        self.traj = traj
        self.domain = domain

        # ---- Load JSON labels ----
        json_path = self.root / traj / f"{traj}.json"
        with open(json_path, "r") as f:
            data = json.load(f)

        # ---- Image directory ----
        img_dir = self.root / traj / domain / "images"

        self.samples = []
        for entry in data:
            filename = entry["filename"]              # e.g. "img000001.jpg"
            q = entry["q_vbs2tango_true"]             # [w, x, y, z]
            t = entry["r_Vo2To_vbs_true"]             # [tx, ty, tz]

            img_path = img_dir / filename

            self.samples.append({
                "img_path": img_path,
                "q": torch.tensor(q, dtype=torch.float32),
                "t": torch.tensor(t, dtype=torch.float32),
            })

        # ---- Default transform (resize to 512x768, then ToTensor) ----
        # You can override this by passing your own transform in __init__.
        if transform is None:
            self.transform = T.Compose([
                T.Resize((512, 768)),   # (H, W) to match SPEED+ input size
                T.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        q = sample["q"]
        t = sample["t"]

        # Load image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "q_gt": q,
            "t_gt": t,
            "idx": idx,
        }
if __name__ == "__main__":
    ds = ShirtRoeDataset(
        root="/home/stan711/datasets/shirtv1",
        traj="roe1",
        domain="lightbox",
    )

    print("Dataset length:", len(ds))
    sample = ds[0]
    print("Keys:", sample.keys())
    print("image shape:", sample["image"].shape)
    print("q_gt:", sample["q_gt"])
    print("t_gt:", sample["t_gt"])
