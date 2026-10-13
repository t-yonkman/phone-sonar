import torch.nn as nn


def block(c_in, c_out, k):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, padding=k // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


class EchoCNN(nn.Module):
    """Echo profile (B, pairs, delays) -> 1D CNN along delay -> class logits (B, n_classes).

    The speaker-to-mic pairs are the input channels, so a filter sees them side by side at the same delay.
    Taking the maximum over delay at the end makes the answer depend on what shapes the echo contains more
    than on exactly when they arrive.
    """

    def __init__(self, n_classes, n_paths=4):
        super().__init__()
        self.conv = nn.Sequential(block(n_paths, 16, 7), block(16, 32, 5), block(32, 64, 3))
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.conv(x).amax(dim=2))  # global max pool
