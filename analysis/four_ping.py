"""The four-ping pattern: what it is, how large, what survives into the model's input, and what to do about it.

    python four_ping.py

Needs the recordings in recordings/sets. Each speaker's sound comes out of the phone a little differently on
every fourth ping. Measured at the mic beside each speaker, in every recording:

- the level, against the recording's average, in four steps that repeat;
- the moment the ping arrives, to a small fraction of a sample, from how its phase runs across the band.

Dividing each ping by its own direct sound takes out a change of level, but not a change of timing, so some of
the pattern is left in what the model is given. Two ways to deal with it are compared on the empty scenes
(what is left that is not an object, against the number of pings averaged) and on the objects (each set held
out in turn):

- as now: average the pings in fours, over which the pattern cancels;
- line each ping up on its own first (the fractional slide that best lays its first 0.1 to 0.4 ms over the
  recording's average ping, as scene.match() does for whole recordings), then average any number.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy import signal

import steps  # noqa: F401  (house style for the figure)
from classify import GATE, centroid, cross, features, fit
from placement import p_fish, score
from scene import GUARD, PRE, W, blocks, chirp, load, speakers, slid
from sets import FOLDER, LO, OBJECTS, read
from steps import BLUE, GREEN, INK2, MUTED, ORANGE, PURPLE

WIN = slice(PRE + 10, PRE + 10 + GATE)   # what the model is given: 0.2 to 1.3 ms
FINE = np.arange(-0.15, 0.15001, 0.0025)  # per-ping slides tried, samples
_F = np.fft.fftfreq(W)
_AT = np.r_[PRE, np.arange(GUARD.start, GUARD.stop)]
_BASIS = np.exp(2j * np.pi * _F[None, None, :] * (_AT[None, :, None] + FINE[:, None, None])) / W   # (slides, points, W)


def arrivals(path):
    """Per speaker, per ping: level at the mic beside it (against the recording's average) and arrival time
    (samples, against the recording's average ping), from the matched filter's phase across 18.5 to 22 kHz."""
    s = json.loads(Path(path).with_suffix(".json").read_text())
    x, fs = sf.read(str(path), dtype="float64", always_2d=True)
    out = []
    for sp in s["speakers"]:
        m = signal.correlate(x[:, sp["reference_channel"]], chirp(s, sp["sweep"] == "up"), mode="valid")
        n = len(m) // s["cycle"]
        k0 = int(np.argmax(np.median(np.abs(m[: n * s["cycle"]]).reshape(n, -1), axis=0)))
        at = k0 + np.arange(1, n - 2) * s["cycle"]
        seg = np.array([m[a - 24:a + 24] for a in at]) * np.hanning(48)
        S = np.fft.fft(seg, axis=1)
        f = np.fft.fftfreq(48) * fs
        band = (f > 18500) & (f < 22000)
        ph = np.unwrap(np.angle(S[:, band] * np.conj(S[:, band].mean(0))), axis=1)
        out.append((np.abs(m[at]) / np.abs(m[at]).mean() - 1, -np.polyfit(f[band], ph.T, 1)[0] / (2 * np.pi) * fs))
    return out


def pattern(v):
    """-> the average of v at each of the four places in the cycle of four, and what is left ping to ping"""
    k = np.arange(len(v))
    p = np.array([v[k % 4 == q].mean() for q in range(4)])
    return p, np.std(v - p[k % 4])


def steady(Q, names):
    """Each ping slid on its own onto the recording's average ping (the slide that best lays its first 0.1 to
    0.4 ms after the direct sound over the average's, on the pair with the mic beside its speaker), and measured
    again against its own slid direct sound. Q (pairs, pings, W) -> the same, and the slides found (speakers, pings)."""
    out, found = Q.copy(), []
    for pairs, own in speakers(names):
        q = Q[own]
        target = np.nanmean(q, 0)[GUARD]
        s = np.einsum("pf,gnf->pgn", np.fft.fft(q, axis=1), _BASIS)           # (pings, slides, points)
        d = FINE[np.argmin(np.sum(np.abs(s[:, :, 1:] / s[:, :, :1] - target) ** 2, -1), 1)]
        ref = slid(q, d[:, None])[:, PRE:PRE + 1]
        for p in pairs:
            out[p] = slid(Q[p], d[:, None]) / ref
        found.append(d)
    return out, np.array(found)


def left_over(Q, names, size):
    """What is left, on the two crossing pairs over the model's window, when the pings of an empty scene are
    averaged `size` at a time and the scene's own average is taken away: dB against the direct sound."""
    keep = cross(names)
    n = Q.shape[1] // size
    A = Q[:, : n * size].reshape(Q.shape[0], n, size, -1).mean(2)
    M = Q.mean(1)
    out = []
    for pairs, own in speakers(names):
        for p in pairs:
            if p in keep:
                e = A[p] / A[own][:, PRE:PRE + 1] - M[p] / M[own][PRE]
                out.append(np.mean(np.abs(e[:, WIN]) ** 2))
    return 10 * np.log10(np.mean(out))


if __name__ == "__main__":
    files = sorted(FOLDER.glob("set_*.wav"))
    print(f"1. The pattern at the mic beside each speaker, {len(files)} recordings")
    rows = {0: [], 1: []}
    example = None
    for f in files:
        for si, (lv, t) in enumerate(arrivals(f)):
            pl, nl = pattern(lv)
            pt, nt = pattern(t)
            rows[si].append((np.sqrt(np.mean(pl ** 2)), pl.max() - pl.min(), nl, pt.max() - pt.min(), nt, np.corrcoef(pl, pt)[0, 1]))
            if f.stem.endswith("185816_1_background") and si == 0:
                example = (lv, t)
    for si, name in enumerate(("bottom speaker", "top speaker")):
        a = np.array(rows[si])
        print(f"  {name}: level, rms of the pattern {100 * np.median(a[:, 0]):.1f}%, peak to trough {100 * np.median(a[:, 1]):.1f}% "
              f"(recordings {100 * a[:, 1].min():.1f} to {100 * a[:, 1].max():.1f}), left ping to ping {100 * np.median(a[:, 2]):.2f}%")
        print(f"  {' ' * len(name)}  timing, peak to trough {np.median(a[:, 3]):.3f} samples = {1e3 * np.median(a[:, 3]) / 48:.1f} µs, "
              f"left ping to ping {np.median(a[:, 4]):.3f} samples; level and timing patterns correlate {np.median(a[:, 5]):+.2f}")

    print("\n2. What is left of an empty scene, crossing pairs, 0.2 to 1.3 ms, dB against the direct sound")
    names, sets = read()
    sizes = [1, 2, 3, 4, 5, 6, 8, 12]
    rest = {"as now": [], "each ping lined up": []}
    for s in sets:
        qb = s["background_pings"]
        rest["as now"].append([left_over(qb, names, b) for b in sizes])
        rest["each ping lined up"].append([left_over(steady(qb, names)[0], names, b) for b in sizes])
    rest = {k: 10 * np.log10(np.mean(10 ** (np.array(v) / 10), 0)) for k, v in rest.items()}
    print("  pings averaged      " + "".join(f"{b:>7d}" for b in sizes))
    for k, v in rest.items():
        print(f"  {k:20s}" + "".join(f"{x:7.1f}" for x in v))
    noise1 = rest["each ping lined up"][0]
    print("  noise alone would be" + "".join(f"{noise1 - 10 * np.log10(b):7.1f}" for b in sizes) + "   (one lined-up ping, divided by the number averaged)")

    print("\n3. Each set held out in turn: recordings right, share of examples right, average P(the right object)")
    pairs = cross(names)
    configs = [("as now", 4), ("as now", 1), ("as now", 2), ("as now", 3), ("as now", 8),
               ("each ping lined up", 1), ("each ping lined up", 2), ("each ping lined up", 4), ("each ping lined up", 8)]
    lined = [{o: steady(s[o + " as recorded"], names)[0] for o in OBJECTS} | {"B": steady(s["background_pings"], names)[0].mean(1)} for s in sets]
    results = {}
    for how, b in configs:
        X, y, g, r = [], [], [], []
        for i, s in enumerate(sets):
            for j, o in enumerate(OBJECTS):
                if how == "as now":
                    e, _ = blocks(s[o + " as recorded"], s["background"], names, size=b)
                else:
                    e, _ = blocks(lined[i][o], lined[i]["B"], names, size=b)
                X.append(features(e[:, pairs, LO:LO + GATE], "each")); y += [j] * len(e); g += [i] * len(e); r += [2 * i + j] * len(e)
        X, y, g, r = np.concatenate(X), np.array(y), np.array(g), np.array(r)
        line = f"  {how:20s} {b:2d} pings ({len(X):4d} examples)"
        for model in ("CNN", "nearest average"):
            p = np.zeros(len(y))
            for held in np.unique(g):
                t = g == held
                p[t] = p_fish(fit(X[~t], y[~t], 30), X[t]) if model == "CNN" else centroid(X[~t], y[~t], X[t])
            share, right, _ = score(p, y, r)
            sure = np.mean([p[r == k].mean() if y[r == k][0] == 0 else 1 - p[r == k].mean() for k in np.unique(r)])
            results[(how, b, model)] = (right, share, sure)
            line += f" | {model}: {right:2d}, {share:4.0%}, {sure:.2f}"
        print(line, flush=True)

    # the figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    a = ax[0]
    lv, t = example
    k = np.arange(48)
    a.plot(k, 100 * lv[:48], color=BLUE, lw=1.3, marker="o", ms=3, label="level, % from average")
    a2 = a.twinx()
    a2.plot(k, 1000 * t[:48] / 48, color=ORANGE, lw=1.3, marker="s", ms=3, label="arrival, µs from average")
    a2.spines["right"].set_visible(True)
    a2.set_ylabel("arrival, µs from average", color=ORANGE)
    a2.grid(False)
    a.set(title="a  Bottom speaker at the bottom mic, ping by ping", xlabel="ping", ylabel="level, % from average")
    a.yaxis.label.set_color(BLUE)
    a.text(0.01, 0.02, "empty scene of set 185816; the first 48 of its pings", transform=a.transAxes, fontsize=8.5, color=MUTED, style="italic")
    a = ax[1]
    for k_, col in (("as now", BLUE), ("each ping lined up", GREEN)):
        a.plot(sizes, rest[k_], color=col, lw=1.8, marker="o", ms=4, label=k_)
    a.plot(sizes, [noise1 - 10 * np.log10(b) for b in sizes], color=MUTED, lw=1.2, ls="--", label="noise alone")
    a.set(title="b  What is left of an empty scene", xlabel="pings averaged", ylabel="dB against the direct sound", xticks=sizes)
    a.legend(loc="upper right", fontsize=8.5)
    a = ax[2]
    for (how, col), off in ((("as now", BLUE), -0.12), (("each ping lined up", GREEN), 0.12)):
        bs = [b for h, b in configs if h == how]
        a.plot(bs, [results[(how, b, "CNN")][1] * 100 for b in bs], color=col, lw=1.8, marker="o", ms=4, label=f"{how}, CNN")
        a.plot(bs, [results[(how, b, "nearest average")][1] * 100 for b in bs], color=col, lw=1.2, ls="--", marker="o", ms=3,
               label=f"{how}, nearest average")
    a.set(title="c  Examples named right, each set held out", xlabel="pings averaged per example", ylabel="% of examples", xticks=[1, 2, 3, 4, 8])
    a.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    out = Path(__file__).parent.parent / "docs" / "four_ping.png"
    fig.savefig(out, dpi=110)
    print("\nwrote", out)
