import torch
from torch import nn


class IterativeRegressionHead(nn.Module):
    """
    RoboPEPP-style iterative MLP regression head.

    The initial state is a registered buffer so the module is device agnostic.
    """

    def __init__(self, feature_dim, output_dim, init_value=None, hidden_dim=1024, n_iter=4, dropout=0.3):
        super().__init__()
        self.n_iter = n_iter
        if init_value is None:
            init = torch.zeros(output_dim, dtype=torch.float32)
        else:
            init = torch.as_tensor(init_value, dtype=torch.float32).reshape(output_dim)
        self.register_buffer("init_value", init.unsqueeze(0))
        self.fc1 = nn.Linear(feature_dim + output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, output_dim)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        nn.init.xavier_uniform_(self.dec.weight, gain=0.01)
        nn.init.zeros_(self.dec.bias)

    def forward(self, xf):
        pred = self.init_value.expand(xf.shape[0], -1)
        for _ in range(self.n_iter):
            x = torch.cat([xf, pred], dim=1).to(torch.float32)
            x = self.drop1(self.fc1(x))
            x = self.drop2(self.fc2(x))
            pred = self.dec(x) + pred
        return pred


class InstrumentActionHead(IterativeRegressionHead):
    def __init__(self, feature_dim, n_actions=3, **kwargs):
        super().__init__(feature_dim, n_actions, **kwargs)


class InstrumentWristPoseHead(IterativeRegressionHead):
    """
    Predicts wrist rotation as quaternion plus xyz translation.
    """

    def __init__(self, feature_dim, **kwargs):
        init = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        super().__init__(feature_dim, 7, init_value=init, **kwargs)

    def forward(self, xf):
        out = super().forward(xf)
        quat = torch.nn.functional.normalize(out[:, :4], p=2, dim=1)
        return quat, out[:, 4:7]
