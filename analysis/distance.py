"""How well can the object's distance be measured from its echo, before knowing which object it is?

    python distance.py

Needs the recordings in recordings/sets. Uses the sets made to a run sheet, whose distance was measured. Block D is
kept in recordings/sealed and is not read here. Every estimate is calibrated on the other blocks and tested on the
one left out, with both objects pooled, so nothing here knows which object it is measuring.

The echo of a recording is the average of its four-ping examples, compared with its own empty scene, on the two
crossing paths. Its delay is read from where the echo first rises to half of its peak power: the nearest part of
the object, which depends less on its shape than the peak or the middle of the echo does. Those two, and a ridge
regression on the whole echo, are there to compare against.

Delay runs from the direct sound at the mic beside the speaker, so an echo's delay is the whole path from speaker
to object to the far mic, nearly a straight line in distance over 10 to 20 cm.

Also here: the same object at the same mark placed twice, how much of the error is noise, whether the error grows
through a block, and how far an object's echo stands above an empty trial.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import steps  # noqa: F401  (house style for the figure)
from classify import cross
from scene import PRE, blocks
from sets import OBJECTS, read
from steps import COLOUR, INK2, MUTED

MS = 48                                 # samples per ms
WINDOW = slice(PRE + 12, PRE + 76)      # 0.25 to 1.6 ms: starts before the echo from 10 cm, which shape.SEARCH does not
ALPHAS = (1, 10, 100, 1000, 10000)
EMPTY = 2                               # stands for an empty trial where a recording's object is numbered


def power(e):
    """The echo's power over WINDOW, both crossing paths: e is (examples, pairs, W), complex."""
    return (np.abs(e.mean(0)[:, WINDOW]) ** 2).sum(0)


def edge(p):
    k = int(np.argmax(p))
    i = k
    while i > 0 and p[i - 1] >= 0.5 * p[k]:
        i -= 1
    at = i if i == 0 else i - 1 + (0.5 * p[k] - p[i - 1]) / (p[i] - p[i - 1])
    return (at + WINDOW.start - PRE) / MS


def peak(p):
    k = int(np.argmax(p))
    if 0 < k < len(p) - 1:
        a, b, c = np.log(p[k - 1:k + 2] + 1e-20)
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return (k + WINDOW.start - PRE) / MS


def centre(p):
    p = np.clip(p - np.median(p), 0, None)
    return (np.sum(np.arange(len(p)) * p) / p.sum() + WINDOW.start - PRE) / MS


def level(e):
    return 10 * np.log10(power(e).max())


def held_out(fit, block):
    """Fit on all blocks but one, for each block: fit(train mask, test mask) -> the distances estimated for test."""
    out = np.zeros(len(block))
    for b in np.unique(block):
        t = block == b
        out[t] = fit(~t, t)
    return out


def ridge(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda A: (A - mu) / sd @ w + y.mean()


def rms(err):
    return np.sqrt(np.mean(err ** 2))


def line(name, err):
    return f"  {name:24s} rms {rms(err):4.2f} cm  within 1 cm {np.mean(np.abs(err) <= 1):4.0%}  within 2 cm {np.mean(np.abs(err) <= 2):4.0%}  ({len(err)})"


if __name__ == "__main__":
    names, sets = read()
    old = [s for s in sets if not s["block"]]
    sets = sorted((s for s in sets if s["block"]), key=lambda s: s["id"])
    pairs = cross(names)
    rows, empties = [], []
    for s in sets:
        run = [t["id"] for t in sets if t["block"] == s["block"]].index(s["id"])
        for j, o in enumerate(OBJECTS):
            e, _ = blocks(s[o + " as recorded"], s["background"], names)
            rows.append((s["block"], j, s["distance"][o], e[:, pairs], s["id"], run))
        if "empty as recorded" in s:
            e, _ = blocks(s["empty as recorded"], s["background"], names)
            empties.append(e[:, pairs])
    block = np.array([r[0] for r in rows])
    obj = np.array([r[1] for r in rows])
    d = np.array([r[2] for r in rows], float)
    set_id = np.array([r[4] for r in rows])
    run = np.array([r[5] for r in rows])
    print(f"{len(rows)} recordings and {len(empties)} empty trials from {len(sets)} sets, blocks {' '.join(np.unique(block))}")

    print("\nEach block held out in turn, both objects pooled, one recording at a time:")
    times = {n: np.array([f(power(r[3])) for r in rows]) for n, f in (("edge", edge), ("peak", peak), ("centre", centre))}
    est = {}
    for n, t in times.items():
        def fit(a, b, t=t):
            k, c = np.polyfit(t[a], d[a], 1)
            return k * t[b] + c
        est[n] = held_out(fit, block)
        print(line(n, est[n] - d))
    X = np.array([(10 * np.log10(np.abs(r[3].mean(0)[:, WINDOW]) ** 2 + 1e-12)).ravel() for r in rows])

    def fitted(a, b):
        scores = []
        for alpha in ALPHAS:                 # chosen inside the training blocks only
            inner = held_out(lambda u, v: ridge(X[a][u], d[a][u], alpha)(X[a][v]), block[a])
            scores.append(np.mean((inner - d[a]) ** 2))
        return ridge(X[a], d[a], ALPHAS[int(np.argmin(scores))])(X[b])

    est["whole echo, ridge"] = held_out(fitted, block)
    print(line("whole echo, ridge", est["whole echo, ridge"] - d))

    best = est["edge"]
    err = best - d
    print("\nThe edge, by group:")
    for n, t in (("fish", obj == 0), ("bobblehead", obj == 1), ("16 to 20 cm, both", d >= 16), ("10 to 14 cm, fish", (d <= 14) & (obj == 0)),
                 ("10 to 14 cm, bobblehead", (d <= 14) & (obj == 1))):
        print(line(n, err[t]))
    print(f"  mean error: fish {err[obj == 0].mean():+.2f} cm, bobblehead {err[obj == 1].mean():+.2f} cm")
    print(f"  {1000 / np.polyfit(times['edge'], d, 1)[0]:.0f} microseconds per cm")

    rng = np.random.default_rng(1)
    ids = np.unique(set_id)
    boot = []
    for _ in range(5000):
        i = np.concatenate([np.flatnonzero(set_id == s) for s in rng.choice(ids, len(ids))])
        boot.append((rms(err[i]), np.mean(np.abs(err[i]) <= 1), np.polyfit(run[i], err[i], 1)[0]))
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    print(f"  resampling sets: rms {low[0]:.2f} to {high[0]:.2f} cm, within 1 cm {low[1]:.0%} to {high[1]:.0%}")
    print(f"  error through a block: {np.polyfit(run, err, 1)[0]:+.2f} cm a set ({low[2]:+.2f} to {high[2]:+.2f})")

    print("\nWorst recordings:")
    for i in np.argsort(-np.abs(err))[:4]:
        print(f"  block {block[i]} set {set_id[i][-6:]} {OBJECTS[obj[i]]:10s} true {d[i]:.0f} cm, estimated {best[i]:.1f} ({err[i]:+.1f}), level {level(rows[i][3]):.0f} dB")

    print("\nSame object, same mark, first and last set of a block, placed by hand each time:")
    moved = []
    for b in np.unique(block):
        for j, o in enumerate(OBJECTS):
            first, last = np.flatnonzero((block == b) & (obj == j) & (run == 0))[0], np.flatnonzero((block == b) & (obj == j) & (run == 6))[0]
            moved.append(best[last] - best[first])
            print(f"  {b} {o:10s} at {d[first]:.0f} cm: {best[first]:.1f} then {best[last]:.1f}")
    print(f"  one reading's repeat error: {np.sqrt(np.mean(np.square(moved)) / 2):.2f} cm")

    half, single = [], []
    k = np.polyfit(times["edge"], d, 1)[0]
    for r in rows:
        e = r[3]
        half.append((edge(power(e[:len(e) // 2])) - edge(power(e[len(e) // 2:]))) * k)
        single.append(np.std([edge(power(e[i:i + 1])) for i in range(len(e))]) * k)
    print(f"\nNoise: first half of a recording against its second {rms(np.array(half)):.2f} cm; "
          f"one 0.4 s example inside a recording, typical spread {np.median(single):.2f} cm")

    objects = np.array([level(r[3]) for r in rows])
    blanks = np.array([level(e) for e in empties])
    print(f"\nLoudest point of the echo, dB: empty trials {blanks.max():.1f} at most; objects {objects.min():.1f} at least, "
          f"{np.sum(objects > blanks.max())} of {len(objects)} above every empty trial")
    for n, t in (("fish", obj == 0), ("bobblehead", obj == 1)):
        print(f"  {n:10s} 10 to 14 cm {np.median(objects[t & (d <= 14)]):.0f}, 16 cm {np.median(objects[t & (d == 16)]):.0f}, 18 to 20 cm {np.median(objects[t & (d >= 18)]):.0f}")
    one = np.concatenate([[level(r[3][i:i + 1]) for i in range(len(r[3]))] for r in rows])
    none = np.concatenate([[level(e[i:i + 1]) for i in range(len(e))] for e in empties])
    print(f"  one 0.4 s example above the loudest of the empty trials' {len(none)}: {np.mean(one > none.max()):.0%} of objects' {len(one)}")

    if old:
        a, b = np.polyfit(times["edge"], d, 1)
        for o in OBJECTS:
            t = np.array([a * edge(power(blocks(s[o + " as recorded"], s["background"], names)[0][:, pairs])) + b for s in old])
            print(f"\nThe {len(old)} earlier sets, placed by eye: {o} reads {np.median(t):.1f} cm typically, {t.min():.1f} to {t.max():.1f}")

    shown = sorted(set(d))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    marks = {"fish": "o", "bobblehead": "^"}
    a = ax[0]
    a.fill_between([8.5, 21.5], [7.5, 20.5], [9.5, 22.5], color=MUTED, alpha=0.15, lw=0)
    a.plot([8.5, 21.5], [8.5, 21.5], color=MUTED, lw=1, ls="--")
    for j, o in enumerate(OBJECTS):
        a.scatter(d[obj == j] + (j - 0.5) * 0.3, best[obj == j], s=44, marker=marks[o], color=COLOUR[o], edgecolor="white", linewidth=0.8, label=o)
    a.text(21.2, 8, "grey band: within 1 cm", fontsize=9, color=INK2, ha="right")
    a.set(title="a  Distance read from the echo, without knowing the object", xlabel="distance placed, cm", ylabel="distance read, cm",
          xticks=shown, xlim=(8.5, 21.5), ylim=(7.5, 22.5))
    a.legend(loc="upper left")
    a = ax[1]
    a.axhspan(blanks.min(), blanks.max(), color=MUTED, alpha=0.25, lw=0)
    a.text(20.6, blanks.max() - 1.6, "empty trials", fontsize=9, color=INK2, ha="right")
    for j, o in enumerate(OBJECTS):
        a.scatter(d[obj == j] + (j - 0.5) * 0.3, objects[obj == j], s=44, marker=marks[o], color=COLOUR[o], edgecolor="white", linewidth=0.8, label=o)
    a.set(title="b  The bobblehead's echo is weak at 10 to 14 cm", xlabel="distance placed, cm", ylabel="loudest point of the echo, dB", xticks=shown, xlim=(8.5, 21.5))
    a.legend(loc="upper right")
    fig.tight_layout()
    out = Path(__file__).parent.parent / "docs" / "distance.png"
    fig.savefig(out, dpi=110)
    print("\nwrote", out)
