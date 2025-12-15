import torch
import torch.nn as nn


def mc_infer(model, img, n_samples=20, device="cpu", extract_pose_fn=None):
    """
    Monte Carlo Dropout inference.

    Args:
      model:        PyTorch model. Must support dropout in forward().
      img:          Input image tensor, shape [C, H, W] or [1, C, H, W].
      n_samples:    Number of stochastic forward passes.
      device:       'cpu' or 'cuda'.
      extract_pose_fn:
                    Function that takes model output and returns a 1D pose tensor.
                    e.g. lambda out: out['pose'] (shape [D])

    Returns:
      mu:   mean pose tensor, shape [D]
      cov:  covariance matrix tensor, shape [D, D]
    """
    device = torch.device(device)

    # Ensure batch dimension
    if img.dim() == 3:
        img = img.unsqueeze(0)  # [1, C, H, W]

    img = img.to(device)

    # If no extractor given, assume model(img) directly returns a [D] vector
    if extract_pose_fn is None:
        def extract_pose_fn_default(out):
            # out expected to be [1, D] or [D]
            if isinstance(out, dict):
                raise ValueError("Model output is dict but no extract_pose_fn was provided.")
            if out.dim() == 2 and out.shape[0] == 1:
                return out.squeeze(0)
            return out
        extract_pose_fn = extract_pose_fn_default

    model = model.to(device)

    # Enable dropout: we deliberately keep the model in train() mode
    model.train()
    poses = []

    with torch.no_grad():
        for _ in range(n_samples):
            out = model(img)            # whatever your model returns
            pose = extract_pose_fn(out) # must be 1D [D]
            pose = pose.view(-1)        # flatten
            poses.append(pose.cpu())

    poses = torch.stack(poses, dim=0)   # [N, D]
    mu = poses.mean(dim=0)              # [D]

    # Flatten [N, D] → covariance [D, D]
    # torch.cov expects [D, N], so transpose
    poses_T = poses.T                   # [D, N]
    cov = torch.cov(poses_T)            # [D, D]

    return mu, cov


# Small self-test with a dummy model to verify shapes/logic
class _DummyModel(nn.Module):
    def __init__(self, D=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, D),
        )

    def forward(self, x):
        # Return dict similar to SPNv2 style: {'pose': ...}
        pose = self.net(x)
        return {"pose": pose}


if __name__ == "__main__":
    # CPU-only test: random image and dummy model
    device = "cpu"
    model = _DummyModel(D=6)

    img = torch.randn(3, 8, 8)  # fake image

    def extract_pose(out):
        # out is dict {'pose': [1, D]}
        return out["pose"].squeeze(0)

    mu, cov = mc_infer(model, img, n_samples=20, device=device, extract_pose_fn=extract_pose)
    print("mu shape:", mu.shape)
    print("cov shape:", cov.shape)
