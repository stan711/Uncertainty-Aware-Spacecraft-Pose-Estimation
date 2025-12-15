import torch


class SimpleKalmanCV3D:
    """
    Simple 3D constant-velocity Kalman filter.

    State s = [px, py, pz, vx, vy, vz]^T  (6x1)
    Measurement z = [px, py, pz]^T        (3x1)

    You provide:
      - dt: timestep
      - Q: process noise covariance (6x6)
      - P0: initial state covariance (6x6)
      - s0: initial state (6,)
    """

    def __init__(self, dt=1.0, Q=None, P0=None, s0=None, device="cpu"):
        self.device = torch.device(device)
        self.dt = dt

        # State transition matrix A (constant velocity)
        dt = float(dt)
        self.A = torch.tensor(
            [
                [1, 0, 0, dt, 0,  0],
                [0, 1, 0, 0,  dt, 0],
                [0, 0, 1, 0,  0,  dt],
                [0, 0, 0, 1,  0,  0],
                [0, 0, 0, 0,  1,  0],
                [0, 0, 0, 0,  0,  1],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Measurement matrix H: we observe only position
        self.H = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Process noise Q (6x6)
        if Q is None:
            Q = 1e-4 * torch.eye(6, dtype=torch.float32)
        self.Q = Q.to(self.device)

        # Initial state covariance P (6x6)
        if P0 is None:
            P0 = torch.eye(6, dtype=torch.float32) * 1.0
        self.P = P0.to(self.device)

        # Initial state s (6,)
        if s0 is None:
            s0 = torch.zeros(6, dtype=torch.float32)
        self.s = s0.to(self.device)

    def predict(self):
        """Predict step: s = A s,  P = A P A^T + Q"""
        self.s = self.A @ self.s
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z, R):
        """
        Update step with measurement z (3,) and measurement noise R (3x3).

        R will later come from your MC-Dropout covariance on translation.
        """
        z = z.to(self.device).view(3, 1)
        R = R.to(self.device)

        H = self.H
        s = self.s.view(6, 1)
        P = self.P

        # Innovation covariance
        S = H @ P @ H.T + R

        # Kalman gain
        K = P @ H.T @ torch.linalg.inv(S)

        # Innovation
        y = z - H @ s

        # Updated state
        s_new = s + K @ y

        # Updated covariance
        I = torch.eye(P.shape[0], device=self.device)
        P_new = (I - K @ H) @ P

        self.s = s_new.view(6)
        self.P = P_new

    @property
    def position(self):
        """Current estimated position [px, py, pz]."""
        return self.s[0:3]

    @property
    def velocity(self):
        """Current estimated velocity [vx, vy, vz]."""
        return self.s[3:6]


if __name__ == "__main__":
    # Tiny self-test with fake data
    device = "cpu"
    kf = SimpleKalmanCV3D(dt=1.0, device=device)

    # True trajectory: moving along +z
    true_positions = [torch.tensor([0.0, 0.0, 7.5 + i * 0.1]) for i in range(10)]

    for i, p_true in enumerate(true_positions):
        # Fake noisy measurement z = p_true + noise
        noise = torch.randn(3) * 0.05
        z = p_true + noise

        # Fixed measurement noise for now (we'll later make this R_t from MC covariance)
        R = (0.05 ** 2) * torch.eye(3)

        # Kalman predict + update
        kf.predict()
        kf.update(z, R)

        print(f"t={i:02d} | true={p_true.numpy()} | meas={z.numpy()} | est={kf.position.detach().numpy()}")
