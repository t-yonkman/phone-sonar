"""Export the trained fish-or-bobblehead CNN for the app, and note what PyTorch makes of the recordings so
the phone's answers can be checked against it.

    ..\\.venv-executorch\\Scripts\\python export.py     (after classify.py has saved fish_or_bobblehead.pt)

Needs the executorch package (1.5.1 here, in its own environment made with
python -m venv --system-site-packages .venv-executorch so it shares the PyTorch already installed).

Writes:

- app/assets/fish_or_bobblehead.pte: the model as an ExecuTorch program, which the app hands to
  ExecuTorch, PyTorch's runtime for phones. It takes one example, shaped (1, pairs, delays) as
  classify.py chose them, and ends in a softmax, so it returns how likely each class is.
- app/assets/selftest.bin: a few examples with PyTorch's answer to each. The app runs them through the
  program every time it opens and only guesses if the phone agrees with the PC.
- app/sim/fish_or_bobblehead.bin: the same weights as plain numbers, for JavaNet, which stands in for the
  phone's runtime when the rest of the app's code is checked on the PC (app/sim/check.ps1).
- parity.json: the average P(fish) PyTorch gives each recording, for that check.
"""
import json
import os
import struct
from pathlib import Path

import executorch.data.bin
import numpy as np
import torch

# The export runs the flatbuffers compiler that comes inside the executorch package. On Windows it only finds
# it if that environment has been activated, so point it there in case it was not.
os.environ.setdefault("FLATC_EXECUTABLE", str(Path(executorch.data.bin.__file__).with_name("flatc.exe" if os.name == "nt" else "flatc")))

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

HERE = Path(__file__).parent
from classify import features, load
from model import EchoCNN
from sets import OBJECTS


class Probabilities(torch.nn.Module):
    """The network followed by a softmax, so the phone is handed probabilities and does no sums of its own."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


saved = torch.load(HERE / "fish_or_bobblehead.pt")
model = EchoCNN(len(OBJECTS), n_paths=len(saved["pair_index"]))
model.load_state_dict(saved["model"])
model.eval()
whole = Probabilities(model).eval()

# every example the model was trained on, as the network is given it, and PyTorch's answer to each
names, ids, E, y, g, r = load()
X = features(E[:, saved["pair_index"], :saved["bins"]], saved["mode"])
with torch.no_grad():
    P = whole(torch.from_numpy(X)).numpy()

# 1. the ExecuTorch program
exported = torch.export.export(whole, (torch.from_numpy(X[:1]),))
program = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()]).to_executorch()
assets = HERE.parent / "app" / "assets"
assets.mkdir(exist_ok=True)
pte = assets / "fish_or_bobblehead.pte"
pte.write_bytes(program.buffer)
print(f"wrote {pte} ({pte.stat().st_size:,} bytes), for input: {', '.join(saved['pairs'])}; loudness: {saved['mode']}; shaped {tuple(X[:1].shape)}")

# ...run on this PC by the same runtime the phone uses, over every example
forward = Runtime.get().load_program(str(pte)).load_method("forward")
back = np.array([forward.execute([torch.from_numpy(X[i:i + 1])])[0].numpy()[0] for i in range(len(X))])
print(f"ExecuTorch on this PC against PyTorch, {len(X)} examples: largest difference in P(fish) {np.abs(back[:, 0] - P[:, 0]).max():.2e}, "
      f"same class for {np.sum(back.argmax(1) == P.argmax(1))}")

# 2. examples for the app to check itself with: the first from each recording
picks = [int(np.flatnonzero(r == k)[0]) for k in np.unique(r)]
test = bytearray(b"TEST") + struct.pack("<3i", len(picks), X.shape[1], X.shape[2])
for i in picks:
    test += X[i].astype("<f4").tobytes() + struct.pack("<f", P[i, 0])
(assets / "selftest.bin").write_bytes(test)
print(f"wrote {assets / 'selftest.bin'}: {len(picks)} examples with PyTorch's P(fish) for each")

# 3. plain weights for the PC stand-in: for each convolution its weights [out][in][tap] then its biases, with
# batch normalisation folded in; then the last layer's weights [class][64] and biases
out = bytearray(b"FISH")
for block in model.conv:
    conv, bn = block[0], block[1]
    scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    w = (conv.weight.detach() * scale[:, None, None]).numpy().astype("<f4")
    b = ((conv.bias.detach() - bn.running_mean) * scale + bn.bias.detach()).numpy().astype("<f4")
    out += w.tobytes() + b.tobytes()
out += model.head.weight.detach().numpy().astype("<f4").tobytes() + model.head.bias.detach().numpy().astype("<f4").tobytes()
(HERE.parent / "app" / "sim" / "fish_or_bobblehead.bin").write_bytes(out)

# 4. what the model makes of each recording: the average over its examples of P(fish)
ref = {f"{ids[k // 2]}_{OBJECTS[k % 2]}": round(float(P[r == k, 0].mean()), 4) for k in np.unique(r)}
(HERE / "parity.json").write_text(json.dumps(ref, indent=1))
print("P(fish) per recording:", " ".join(f"{v:.2f}" for v in ref.values()))
