"""Fish or bobblehead? Train on some sets, test on sets never seen, for a small CNN and a nearest-average
baseline.

    python classify.py                 each set held out in turn, for what the phone is given and for what it
                                       was first given; then save the phone's model trained on every set
    python classify.py grid            the same test over every choice of window, pairs and loudness removal
    python classify.py since 203955    train on the sets before that one, test on it and the ones after

An example is the average of four consecutive pings with the object in place, minus the background of
the same set. Four, to bring the noise down: three to eight do as well when each set is held out, one
or two worse (four_ping.py). The phone's level also repeats every four pings, but dividing each ping by
its direct sound leaves less of that than the noise. That gives about 24 examples per recording, all
from the same placement, so the only fair test is on whole sets held out.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import EchoCNN
from scene import blocks
from sets import FOLDER, HI, LO, OBJECTS, read

# What the network on the phone is given (Judge.java does the same there). Arrived at from the first sets
# the phone guessed at, on 2026-10-12: see the README.
GATE = 52        # delay steps kept, from 0.2 ms: up to 1.3 ms, about as late as an object 10 cm away can echo
MODE = "each"    # each pair's own loudness removed
CROSS = True     # only the pairs from one end of the phone to the other; the same-end pairs are left out

ECHOES = Path(__file__).parent / "echoes.npz"


def examples(names, sets):
    """-> complex echoes (n, pairs, HI - LO), object index, set index, recording index"""
    E, y, g, r = [], [], [], []
    for i, s in enumerate(sets):
        for j, o in enumerate(OBJECTS):
            e, _ = blocks(s[o + " as recorded"], s["background"], names)
            E.append(e[:, :, LO:HI])
            y += [j] * len(e); g += [i] * len(e); r += [2 * i + j] * len(e)
    return np.concatenate(E), np.array(y), np.array(g), np.array(r)


def load():
    """-> pair names, set ids, and what examples() gives for every set.

    The raw recordings are room audio and are not kept in the repository; the echoes taken from them are, in
    echoes.npz. Where the recordings are present the echoes are taken afresh and that file rewritten.
    """
    if any(FOLDER.glob("set_*.wav")):
        names, sets = read()
        E, y, g, r = examples(names, sets)
        np.savez_compressed(ECHOES, names=names, ids=[s["id"] for s in sets], made=[s["made"] for s in sets],
                            E=E.astype(np.complex64), y=y, g=g, r=r)
    d = np.load(ECHOES)
    return [str(n) for n in d["names"]], [str(i) for i in d["ids"]], d["E"], d["y"], d["g"], d["r"]


def cross(names):
    """Which pairs go from a speaker to the mic at the other end of the phone."""
    return [i for i, n in enumerate(names) if n.split(" spk > ")[0] != n.split(" spk > ")[1].replace(" mic", "")]


def given(names):
    """Which pairs the phone's network is given."""
    return cross(names) if CROSS else list(range(len(names)))


MODES = {"heard": "as heard",
         "shape": "overall loudness removed",
         "each": "each pair's loudness removed"}


def features(E, mode):
    """dB along delay for each pair, scaled to 0..1 over a 60 dB range, after removing loudness as the mode says."""
    d = 20 * np.log10(np.abs(E) + 1e-12)
    floor = -80.0
    if mode != "heard":
        d = d - 10 * np.log10(np.mean(np.abs(E) ** 2, axis=(1, 2) if mode == "shape" else 2, keepdims=True))
        floor = -50.0
    return np.ascontiguousarray((np.clip(d, floor, floor + 60) - floor) / 60, dtype=np.float32)


def fit(X, y, epochs):
    torch.manual_seed(0)
    model = EchoCNN(len(OBJECTS), n_paths=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(X)).split(32):
            loss = F.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def cnn(Xa, ya, Xb, epochs):
    """-> P(fish) for each example of Xb"""
    model = fit(Xa, ya, epochs)
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(Xb)), 1)[:, 0].numpy()


def centroid(Xa, ya, Xb, epochs=None):
    """-> 1 for each example of Xb nearer the average fish than the average bobblehead, else 0"""
    c = np.array([Xa[ya == k].reshape((ya == k).sum(), -1).mean(0) for k in range(len(OBJECTS))])
    return (np.argmin(((Xb.reshape(len(Xb), 1, -1) - c[None]) ** 2).sum(-1), axis=1) == 0).astype(float)


def report(p, y, r, ids):
    """How a test came out: share of examples right, and recordings right going by their average P(fish)."""
    rec = [(k, y[r == k][0], int(p[r == k].mean() < 0.5)) for k in np.unique(r)]
    wrong = [f"{ids[k // 2][-4:]}{OBJECTS[k % 2][0]}" for k, a, b in rec if a != b]
    return f"{np.mean((p < 0.5) == y):4.0%} of examples, {len(rec) - len(wrong):2d} of {len(rec)} recordings" \
           + (f" (wrong: {' '.join(wrong)})" if wrong else "")


def test(X, y, g, r, ids, epochs, train=None):
    """Each set held out in turn, or (train given) trained on those examples and tested on the rest."""
    line = ""
    for name, model in (("CNN", cnn), ("nearest average", centroid)):
        p = np.zeros(len(y))
        if train is None:
            for held in np.unique(g):
                t = g == held
                p[t] = model(X[~t], y[~t], X[t], epochs)
            line += f" | {name}: {report(p, y, r, ids)}"
        else:
            p[~train] = model(X[train], y[train], X[~train], epochs)
            line += f" | {name}: {report(p[~train], y[~train], r[~train], ids)}"
    return line


if __name__ == "__main__":
    epochs = 30
    names, ids, E, y, g, r = load()
    every = list(range(len(names)))
    print(f"{len(E)} examples from {len(ids)} sets; pairs: {names}")
    ours = (f"what the phone is given ({'cross pairs' if CROSS else 'all four pairs'}, to {(10 + GATE) / 48:.1f} ms, {MODES[MODE]})", given(names), GATE, MODE)
    first = (f"what it was first given (all four pairs, to {(10 + HI - LO) / 48:.1f} ms, {MODES['shape']})", every, HI - LO, "shape")

    if "since" in sys.argv:
        start = sys.argv[sys.argv.index("since") + 1]
        saved = np.load(ECHOES)
        made = [str(m) for m in saved["made"]] if "made" in saved.files else ids   # ids alone are only the time of day
        train = np.isin(g, [i for i, m in enumerate(made) if m < made[ids.index(start)]])
        print(f"\nTrained on the {len(np.unique(g[train]))} sets made before set {start}, tested on it and the {len(np.unique(g[~train])) - 1} made after it")
        for label, pairs, bins, mode in (ours, first):
            print(f"  {label}\n   {test(features(E[:, pairs, :bins], mode), y, g, r, ids, epochs, train)}")
    elif "grid" in sys.argv:
        print("\nEach set held out in turn")
        for bins in (GATE, 80, HI - LO):
            for label, pairs in (("all four pairs", every), ("cross pairs", cross(names))):
                for mode in ("shape", "each"):
                    print(f"  to {(10 + bins) / 48:.1f} ms, {label:14s}, {MODES[mode]:28s}{test(features(E[:, pairs, :bins], mode), y, g, r, ids, epochs)}", flush=True)
    else:
        print("\nEach set held out in turn")
        for label, pairs, bins, mode in (ours, first):
            print(f"  {label}\n   {test(features(E[:, pairs, :bins], mode), y, g, r, ids, epochs)}", flush=True)
        _, pairs, bins, mode = ours
        torch.save({"model": fit(features(E[:, pairs, :bins], mode), y, epochs).state_dict(), "classes": list(OBJECTS),
                    "pairs": [names[i] for i in pairs], "pair_index": pairs, "bins": bins, "mode": mode, "sets": ids},
                   Path(__file__).parent / "fish_or_bobblehead.pt")
        print(f"\nsaved fish_or_bobblehead.pt, trained on all {len(ids)} sets")
