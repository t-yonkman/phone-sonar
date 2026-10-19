"""How much does it matter if the object stands a little closer or further than 10 cm?    python placement.py

Needs the raw recordings in recordings/sets. Moving the object changes three things about its echo: when it
arrives, how strong it is, and the angle each end of the phone sees the object from. The first two follow
from the geometry and can be imitated on the recordings already made: on the pairs that cross the phone, the
part of each echo late enough to have come from the object is moved in delay and scaled for the longer or
shorter path, and the network is tested on that with each set held out in turn. The angle cannot be
imitated, so a real move does at least this much.

The same test is run on a network trained on copies of the training sets moved between 7 and 13 cm, to see
whether a model can be taught not to care.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import steps  # noqa: F401  (house style for the figure)
from classify import GATE, centroid, cross, features, fit
from scene import PRE, W, slid
from scene import blocks as to_blocks
from sets import LO, OBJECTS, read
from steps import BLUE, GREEN, INK2, MUTED, ORANGE, PURPLE

HALF = 8.0           # cm from the middle of the phone to the speaker and mic at each end (16 cm apart)
D0 = 10.0            # cm from the phone to the object, as recorded
C = 34.3             # cm per ms
FROM = PRE + 22      # 0.46 ms: a cross pair cannot hear anything off the object sooner (16 cm / 34.3 cm/ms)
DISTANCES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
TRAIN_AT = [7, 8.5, 10, 11.5, 13]

ramp = np.clip((np.arange(W) - (FROM - 3)) / 6, 0, 1)
TAPER = 0.5 - 0.5 * np.cos(np.pi * ramp)   # 0 before 0.40 ms, 1 after 0.52 ms


def path(d):
    """Speaker to object to the mic at the other end, cm, for an object d cm out on the centre line."""
    return 2 * np.sqrt(HALF ** 2 + d ** 2)


def later(d):
    """How much later the object's echo arrives at d cm than at D0, in samples."""
    return (path(d) - path(D0)) / C * 48


def louder(d):
    """How much stronger (amplitude) the echo is at d cm than at D0: spreading on the way out and back."""
    return (HALF ** 2 + D0 ** 2) / (HALF ** 2 + d ** 2)


def moved(e, d):
    """Echoes e (examples, pairs, W) as if the object had stood d cm from the phone."""
    if d == D0:
        return e
    obj = e * TAPER
    return e - obj + louder(d) * slid(obj, -later(d))


def echoes():
    """-> the two cross pairs' echoes over the whole window (examples, 2, W), complex; object, set and recording
    of each example; and the set ids. Taken afresh from the recordings, as blocks() does on the phone."""
    names, sets = read()
    pairs = cross(names)
    E, y, g, r = [], [], [], []
    for i, s in enumerate(sets):
        for j, o in enumerate(OBJECTS):
            e, _ = to_blocks(s[o + " as recorded"], s["background"], names)
            E.append(e[:, pairs]); y += [j] * len(e); g += [i] * len(e); r += [2 * i + j] * len(e)
    return np.concatenate(E), np.array(y), np.array(g), np.array(r), [s["id"] for s in sets]


def p_fish(model, X):
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X)), 1)[:, 0].numpy()


def score(p, y, r):
    """-> share of examples right, recordings right, and the recordings' average P(fish) per object."""
    rec = np.array([(y[r == k][0], p[r == k].mean()) for k in np.unique(r)])
    right = (rec[:, 1] >= 0.5) == (rec[:, 0] == 0)
    return np.mean((p >= 0.5) == (y == 0)), int(right.sum()), [rec[rec[:, 0] == j, 1].mean() for j in range(len(OBJECTS))]


if __name__ == "__main__":
    x = np.zeros(W); x[100] = 1
    assert np.argmax(np.abs(slid(x, -3.0))) == 103   # slid(q, -s) puts things s samples later

    print("Geometry: an object d cm out, speaker and mic 8 cm either side")
    for d in DISTANCES:
        print(f"  {d:4.1f} cm: echo at {path(d) / C:.2f} ms ({later(d):+5.1f} samples), {20 * np.log10(louder(d)):+5.1f} dB")

    E, y, g, r, ids = echoes()

    def X(e):
        return features(e[:, :, LO:LO + GATE], "each")

    # Where each recording's echo actually peaked on the louder cross pair: how much placement already varied
    print("\nWhere the echo peaked in each recording (bottom spk > top mic), and the distance that implies")
    peaks = {}
    for j, o in enumerate(OBJECTS):
        ms = []
        for k in np.unique(r[y == j]):
            m = np.abs(E[r == k, 0].mean(0))
            ms.append((FROM + np.argmax(m[FROM:LO + GATE]) - PRE) / 48)
        ms = np.array(ms)
        d = np.sqrt(np.maximum((ms * C / 2) ** 2 - HALF ** 2, 0))
        peaks[o] = d
        print(f"  {o:10s}: {ms.min():.2f} to {ms.max():.2f} ms, i.e. {d.min():.1f} to {d.max():.1f} cm (spread {d.std():.1f} cm)")
    for i, s in enumerate(ids):
        print(f"    {s}: " + "  ".join(f"{o} {peaks[o][i]:4.1f} cm" for o in OBJECTS))

    # Each set held out in turn, tested as if moved
    names_ = ("CNN", "nearest average", f"CNN trained at {TRAIN_AT[0]} to {TRAIN_AT[-1]} cm")
    P = {n: np.zeros((len(DISTANCES), len(y))) for n in names_}
    for held in np.unique(g):
        t = g == held
        Xa = X(E[~t])
        plain = fit(Xa, y[~t], 30)
        wide = fit(np.concatenate([X(moved(E[~t], d)) for d in TRAIN_AT]), np.tile(y[~t], len(TRAIN_AT)), 30)
        for k, d in enumerate(DISTANCES):
            Xb = X(moved(E[t], d))
            P[names_[0]][k, t] = p_fish(plain, Xb)
            P[names_[1]][k, t] = centroid(Xa, y[~t], Xb)
            P[names_[2]][k, t] = p_fish(wide, Xb)
        print(f"  held out {ids[held]}", flush=True)

    n_rec = len(np.unique(r))
    print(f"\nEach set held out, its object moved; recordings right of {n_rec} (and share of examples)")
    print(f"  {'cm':>4s}  " + "  ".join(f"{n:>34s}" for n in names_))
    results = {n: [score(P[n][k], y, r) for k in range(len(DISTANCES))] for n in names_}
    for k, d in enumerate(DISTANCES):
        print(f"  {d:4.1f}  " + "  ".join(f"{results[n][k][1]:>24d} ({results[n][k][0]:4.0%})" for n in names_))
    print("\nCNN's average P(fish) per object as it moves")
    for k, d in enumerate(DISTANCES):
        print(f"  {d:4.1f} cm: " + "  ".join(f"{o} {results[names_[0]][k][2][j]:.2f}" for j, o in enumerate(OBJECTS))
              + "   | trained wide: " + "  ".join(f"{o} {results[names_[2]][k][2][j]:.2f}" for j, o in enumerate(OBJECTS)))

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    a = ax[0]
    lo, hi = np.percentile(peaks["fish"], [25, 75])
    a.axvspan(lo, hi, color=GREEN, alpha=0.1, lw=0)
    a.text((lo + hi) / 2, 1.5, "middle half of the fish\nrecordings' echo peaks\n(rough placement)", ha="center", fontsize=8.5, color=GREEN)
    for n, col in zip(names_, (BLUE, MUTED, PURPLE)):
        a.plot(DISTANCES, [v[1] for v in results[n]], color=col, lw=1.8, marker="o", ms=4, label=n)
    a.axhline(n_rec / 2, color=INK2, lw=1, ls=":")
    a.text(DISTANCES[0], n_rec / 2 - 1.4, "guessing", ha="left", fontsize=8.5, color=INK2)
    a.set(title="Recordings named right, each set held out, object moved", xlabel="object's distance from the phone, cm",
          ylabel=f"recordings right, of {n_rec}", ylim=(0, n_rec + 1), xticks=DISTANCES)
    a.legend(loc="lower left", fontsize=8.5)
    a = ax[1]
    for j, o in enumerate(OBJECTS):
        col = BLUE if o == "fish" else ORANGE
        a.plot(DISTANCES, [v[2][j] for v in results[names_[0]]], color=col, lw=1.8, marker="o", ms=4, label=f"{o}, CNN")
        a.plot(DISTANCES, [v[2][j] for v in results[names_[2]]], color=col, lw=1.4, ls="--", label=f"{o}, CNN trained at 7 to 13 cm")
    a.axhline(0.5, color=INK2, lw=1, ls=":")
    a.set(title="Average P(fish) the network gives each object", xlabel="object's distance from the phone, cm",
          ylabel="P(fish)", ylim=(-0.03, 1.03), xticks=DISTANCES)
    a.legend(loc="center right", fontsize=8.5)
    fig.text(0.99, 0.005, "only delay and strength are imitated from the 10 cm recordings; the change in viewing angle is not",
             ha="right", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = Path(__file__).parent.parent / "docs" / "placement.png"
    fig.savefig(out, dpi=110)
    print("\nwrote", out)
