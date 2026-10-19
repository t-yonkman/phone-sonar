"""Features that keep the shape of the object's echo and throw away when it arrived.    python shape.py

Subtracting the background leaves what the object added, and removing each pair's loudness leaves the echo's
form; but both still sit where they arrived in time, and that is the object's distance. The network is given
a fixed stretch of delay, so it can use where the echo falls in it, and placement.py shows it does. Three ways
to give it the shape alone, each tested against the input used now:

- aligned: each recording's echo is found (the power-weighted middle of where an object can echo, both cross
  pairs together), slid to a fixed delay, and a window kept around it: 0.29 ms before its middle to 0.5 after.
- autocorrelation: how much the echo resembles itself shifted by 0 to 0.5 ms, on each pair. Moving the echo
  in time does not change this at all. A compact echo falls away quickly; one from parts at different
  distances has bumps at the spacings between them.
- shape statistics: per pair, how wide the echo is (rms about its middle), the share of its energy within
  0.1 ms of its peak, how far its peak stands above its average, and how soon its autocorrelation falls to half.

Each is tested with each set held out in turn, at 10 cm as recorded and with the object's echo moved as in
placement.py. Moving there imitates only the change in delay and strength, which these features are built to
ignore, so their staying flat is expected; what the test can show is whether they still tell the objects apart,
and whether what is left near the edges of the window gives distance away.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import steps  # noqa: F401  (house style for the figure)
from classify import GATE, centroid, features, fit
from placement import D0, DISTANCES, FROM, echoes, moved, p_fish, score
from scene import PRE, slid
from sets import LO, OBJECTS
from steps import BLUE, COLOUR, GREEN, INK2, MUTED, PURPLE

SEARCH = slice(FROM, PRE + 78)          # 0.46 to 1.6 ms: where an object about 5 to 18 cm out can echo
REF = PRE + 36                          # 0.75 ms: where each echo's middle is put
AROUND = slice(REF - 14, REF + 24)      # the window kept around it
LAGS = 25                               # autocorrelation out to 0.5 ms
_n = SEARCH.stop - SEARCH.start
EDGE = np.ones(_n)
EDGE[:4] = EDGE[-4:][::-1] = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, 5) / 5)


def middles(E, r):
    """Each recording's echo middle, in samples: power-weighted over SEARCH, both pairs, all its blocks, above
    the median (so the noise across the window does not pull it to the centre)."""
    out = np.zeros(len(E))
    for k in np.unique(r):
        p = (np.abs(E[r == k][:, :, SEARCH]) ** 2).mean((0, 1))
        p = np.clip(p - np.median(p), 0, None)
        out[r == k] = SEARCH.start + np.sum(np.arange(len(p)) * p) / p.sum()
    return out


def aligned(E, r):
    A = slid(E, (middles(E, r) - REF)[:, None, None])
    return features(A[:, :, AROUND], "each")


def autocorr(E):
    """|sum over delay of e(t) e*(t + lag)| / the same at lag 0, per pair: (examples, pairs, LAGS) in 0..1"""
    e = E[:, :, SEARCH] * EDGE
    R = np.stack([np.abs(np.sum(e[..., :_n - l] * np.conj(e[..., l:]), -1)) for l in range(LAGS)], -1)
    return (R / R[..., :1]).astype(np.float32)


def stats(E):
    """(examples, 8): per pair, width (ms), share near the peak, peak over average (dB), autocorrelation half-width (ms)"""
    p = np.abs(E[:, :, SEARCH]) ** 2
    q = np.clip(p - np.median(p, -1, keepdims=True), 0, None) + 1e-30
    t = np.arange(_n) / 48
    mid = (q * t).sum(-1) / q.sum(-1)
    width = np.sqrt((q * (t - mid[..., None]) ** 2).sum(-1) / q.sum(-1))
    peak = t[p.argmax(-1)]
    core = (p * (np.abs(t - peak[..., None]) <= 0.1)).sum(-1) / p.sum(-1)
    peaky = 10 * np.log10(p.max(-1) / p.mean(-1))
    R = autocorr(E)
    half = np.where(R.min(-1) >= 0.5, LAGS, (R >= 0.5).argmin(-1)) / 48
    return np.concatenate([width, core, peaky, half], 1).astype(np.float32)


FEATURES = {
    "as now: 0.2 to 1.3 ms": (lambda E, r: features(E[:, :, LO:LO + GATE], "each"), ("CNN", "nearest average")),
    "aligned on the echo": (aligned, ("CNN", "nearest average")),
    "autocorrelation": (lambda E, r: autocorr(E), ("CNN", "nearest average")),
    "shape statistics": (lambda E, r: stats(E), ("logistic", "nearest average")),
}
LINE = {"as now: 0.2 to 1.3 ms": MUTED, "aligned on the echo": BLUE, "autocorrelation": PURPLE, "shape statistics": GREEN}


def logistic(Xa, ya, Xb):
    torch.manual_seed(0)
    lin = torch.nn.Linear(Xa.shape[1], len(OBJECTS))
    opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-3)
    A, yy = torch.from_numpy(Xa), torch.from_numpy(ya)
    for _ in range(300):
        loss = F.cross_entropy(lin(A), yy)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return [torch.softmax(lin(torch.from_numpy(x)), 1)[:, 0].numpy() for x in Xb]


def train_and_test(model, Xa, ya, Xbs):
    """-> P(fish) for each array in Xbs"""
    if Xa.ndim == 2:   # statistics: put each on the same footing first
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
        Xa, Xbs = (Xa - mu) / sd, [(x - mu) / sd for x in Xbs]
        Xa, Xbs = Xa.astype(np.float32), [x.astype(np.float32) for x in Xbs]
    if model == "CNN":
        net = fit(Xa, ya, 30)
        return [p_fish(net, x) for x in Xbs]
    if model == "logistic":
        return logistic(Xa, ya, Xbs)
    return [centroid(Xa, ya, x) for x in Xbs]


if __name__ == "__main__":
    E, y, g, r, ids = echoes()
    at = {d: {n: f(moved(E, d), r) for n, (f, _) in FEATURES.items()} for d in DISTANCES}
    k0 = DISTANCES.index(D0)
    print("Echo middles found at 10 cm: " + ", ".join(f"{o} {np.mean((middles(E, r)[y == j] - PRE) / 48):.2f} ms" for j, o in enumerate(OBJECTS)))

    P = {(n, m): np.zeros((len(DISTANCES), len(y))) for n, (_, ms) in FEATURES.items() for m in ms}
    for held in np.unique(g):
        t = g == held
        for (n, m), out in P.items():
            for k, p in enumerate(train_and_test(m, at[D0][n][~t], y[~t], [at[d][n][t] for d in DISTANCES])):
                out[k, t] = p
        print(f"  held out {ids[held]}", flush=True)

    n_rec = len(np.unique(r))
    results = {key: [score(P[key][k], y, r) for k in range(len(DISTANCES))] for key in P}
    print(f"\nRecordings right of {n_rec}, each set held out, object moved (cm across the top)")
    print(f"  {'':24s} {'':16s}" + "".join(f"{d:>5d}" for d in DISTANCES) + "   at 10 cm: examples, wrong")
    for (n, m), res in results.items():
        p10 = P[(n, m)][k0]
        wrong = [f"{ids[k // 2][-4:]}{OBJECTS[k % 2][0]}" for k in np.unique(r) if (p10[r == k].mean() >= 0.5) != (y[r == k][0] == 0)]
        print(f"  {n:24s} {m:16s}" + "".join(f"{v[1]:>5d}" for v in res) + f"   {res[k0][0]:4.0%}  {' '.join(wrong)}")

    # the figure: what each feature looks like for the two objects, then how each holds up
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 6)
    a = fig.add_subplot(gs[0, 0:2])
    X = at[D0]["aligned on the echo"]
    ms = (np.arange(AROUND.start, AROUND.stop) - REF) / 48
    for j, o in enumerate(OBJECTS):
        v = X[y == j, 0]
        a.fill_between(ms, np.percentile(v, 15, 0), np.percentile(v, 85, 0), color=COLOUR[o], alpha=0.18, lw=0)
        a.plot(ms, np.median(v, 0), color=COLOUR[o], lw=1.6, label=o)
    a.set(title="Aligned: echo about its own middle", xlabel="ms from the echo's middle", ylabel="feature (0..1)", ylim=(0, 1))
    a.legend(loc="lower center")
    a = fig.add_subplot(gs[0, 2:4])
    X = at[D0]["autocorrelation"]
    lag = np.arange(LAGS) / 48
    for j, o in enumerate(OBJECTS):
        v = X[y == j, 0]
        a.fill_between(lag, np.percentile(v, 15, 0), np.percentile(v, 85, 0), color=COLOUR[o], alpha=0.18, lw=0)
        a.plot(lag, np.median(v, 0), color=COLOUR[o], lw=1.6, label=o)
    a.set(title="Autocorrelation: echo against itself", xlabel="shift, ms", ylabel="similarity (0..1)", ylim=(0, 1.02))
    a = fig.add_subplot(gs[0, 4:6])
    X = at[D0]["shape statistics"]
    for j, o in enumerate(OBJECTS):
        a.scatter(1000 * X[y == j, 0], X[y == j, 2], s=8, color=COLOUR[o], alpha=0.35, lw=0, label=o)
    a.set(title="Two of the statistics", xlabel="echo width, rms about its middle, µs", ylabel="share of energy within 0.1 ms of the peak")
    for col, (title, which) in enumerate((("CNN (logistic regression for the statistics)", ("CNN", "logistic")), ("Nearest average", ("nearest average",)))):
        a = fig.add_subplot(gs[1, 3 * col:3 * col + 3])
        for (n, m), res in results.items():
            if m in which:
                a.plot(DISTANCES, [v[1] for v in res], color=LINE[n], lw=1.8, marker="o", ms=4, label=n)
        a.axhline(n_rec / 2, color=INK2, lw=1, ls=":")
        a.text(DISTANCES[0], n_rec / 2 - 1.4, "guessing", fontsize=8.5, color=INK2)
        a.set(title=title, xlabel="object's distance from the phone, cm", ylabel=f"recordings right, of {n_rec}",
              ylim=(0, n_rec + 1), xticks=DISTANCES)
        a.legend(loc="lower center", fontsize=8.5, ncol=2)
    fig.text(0.99, 0.005, "top: bottom spk > top mic, real echoes at 10 cm, median and 15th to 85th percentile.  bottom: each set held out; "
             "moving imitates delay and strength only", ha="right", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = Path(__file__).parent.parent / "docs" / "shape_features.png"
    fig.savefig(out, dpi=110)
    print("\nwrote", out)
