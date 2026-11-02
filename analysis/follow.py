"""Find the object, read its distance, and only then say which it is.    python follow.py

Needs the recordings in recordings/sets. Block D is kept in recordings/sealed and is not read here.

What is frozen here, for a recording:

1. Is anything there? The loudest point of the averaged echo (distance.py) has to pass DETECT.
2. How far? The echo's edge, through a straight line fitted on the measured blocks.
3. Which? A 1 ms window starting just before that edge goes to the same small network as before, in dB as heard: the
   echo stays measured against the direct sound, so its strength goes in with its shape. In training the window is
   moved at random by up to the distance a reading can be off, so the answer survives a misread.

Tested with each measured block held out in turn, then trained on all of them and tested on the earlier sets, which
were made on other days in other spots. Those sets are not used for training: they did no good and a little harm.

Strength carries most of the answer, so a second model is kept beside it: the same, but with every example made up
to BLUR dB louder or quieter in training, so that it cannot lean on strength. Block D will test both.

Saves follow.pt, trained on every measured block. fish_or_bobblehead.pt, which the phone runs today, is untouched.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import steps  # noqa: F401  (house style for the figure)
from classify import centroid, cross, features
from classify import fit as fit_fixed
from distance import WINDOW, edge
from model import EchoCNN
from scene import PRE, blocks
from sets import LO, OBJECTS, read
from shape import REF, logistic, stats
from steps import BLUE, COLOUR, GRID, INK, INK2, MUTED

BEFORE, AFTER = 10, 38    # the window, in samples before and after the echo's edge: 1 ms
MODE = "heard"
MOVE = 5                  # samples the window is moved by in training, about 2 cm
BLUR = 10                 # dB, the second model's training only
DETECT = -54.0            # dB: between the empty trials, -55.6 at most, and all but one object, -52.3 at least
EPOCHS = 30


def recordings(names, sets):
    """-> echoes on the crossing paths (n, pairs, W), object (2 for an empty trial), set, recording, distance in cm"""
    pairs = cross(names)
    E, y, g, r, d = [], [], [], [], []
    for i, s in enumerate(sets):
        for j, o in enumerate(OBJECTS + ("empty",)):
            if o + " as recorded" in s:
                e, _ = blocks(s[o + " as recorded"], s["background"], names)
                E.append(e[:, pairs].astype(np.complex64))
                y += [j] * len(e); g += [i] * len(e); r += [3 * i + j] * len(e); d += [s["distance"].get(o, np.nan)] * len(e)
    return np.concatenate(E), np.array(y), np.array(g), np.array(r), np.array(d, float)


def read_echo(E, r):
    """-> for each example its recording's edge in ms after the direct sound, and the loudest point of its echo in dB"""
    at, level = np.zeros(len(E)), np.zeros(len(E))
    for k in np.unique(r):
        t = r == k
        p = (np.abs(E[t].mean(0)[:, WINDOW]) ** 2).sum(0)
        at[t], level[t] = edge(p), 10 * np.log10(p.max())
    return at, level


def cut(E, start):
    """The window of each example, starting at sample start[n]"""
    i = (start[:, None] + np.arange(BEFORE + AFTER))[:, None].repeat(E.shape[1], 1)
    return np.take_along_axis(E, i, 2)


def fit(E, y, start, seed=0, blur=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EchoCNN(len(OBJECTS), n_paths=E.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    labels = torch.from_numpy(y)
    model.train()
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(E)).split(32):
            i = idx.numpy()
            louder = 10 ** (rng.uniform(-blur, blur, len(i)) / 20)
            x = features(cut(E[i] * louder[:, None, None], start[i] + rng.integers(-MOVE, MOVE + 1, len(i))), MODE)
            loss = F.cross_entropy(model(torch.from_numpy(x)), labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def p_fish(model, X):
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X)), 1)[:, 0].numpy()


def right(p, y, r):
    """-> for each recording: whether its average P(fish) names it, its average P(fish), its object"""
    ks = np.unique(r)
    mean = np.array([p[r == k].mean() for k in ks])
    truth = np.array([y[r == k][0] for k in ks])
    return (mean < 0.5) == truth, mean, truth


def statistics(E, r, start):
    A = np.array(E)
    for k in np.unique(r):
        t = r == k
        A[t] = np.roll(E[t], REF - start[t][0] - BEFORE, 2)
    return stats(A)


def standardised(model):
    def go(Xa, ya, Xb):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
        return model(((Xa - mu) / sd).astype(np.float32), ya, [((Xb - mu) / sd).astype(np.float32)])[0]
    return go


if __name__ == "__main__":
    names, sets = read()
    measured = [s["id"] for s in sets if s["block"]]
    E, y, g, r, d = recordings(names, sets)
    delay, level = read_echo(E, r)
    block = np.array([sets[i]["block"] for i in g])
    start = PRE + np.round(delay * 48).astype(int) - BEFORE
    objects = y < 2
    new, old = (block != "") & objects, (block == "") & objects
    letters = np.unique(block[new])
    print(f"{len(np.unique(r[new]))} recordings measured, blocks {' '.join(letters)}; {len(np.unique(r[old]))} earlier")

    now = torch.load(Path(__file__).parent / "fish_or_bobblehead.pt")
    phone = EchoCNN(len(OBJECTS), n_paths=2)
    phone.load_state_dict(now["model"])
    fixed = features(E[:, :, LO:LO + now["bins"]], now["mode"])
    print(f"\nThe phone's model today, trained on the earlier sets: {right(p_fish(phone.eval(), fixed[new]), y[new], r[new])[0].sum()} of 42 on the measured blocks")

    def held_out(guess):
        p = np.zeros(len(y))
        for b in letters:
            a, t = new & (block != b), new & (block == b)
            p[t] = guess(a, t)
        return p[new]

    window = cut(E, start)
    shape = features(window, "each")
    heard = features(window, MODE)
    two = np.c_[level, delay]
    inputs = {
        "fixed window, loudness removed": lambda a, t: p_fish(fit_fixed(fixed[a], y[a], EPOCHS), fixed[t]),
        "shape statistics": lambda a, t: standardised(logistic)(st[a], y[a], st[t]),
        "follows the echo, loudness removed": lambda a, t: p_fish(fit_fixed(shape[a], y[a], EPOCHS), shape[t]),
        "strength and delay, two numbers": lambda a, t: standardised(logistic)(two[a], y[a], two[t]),
        "nearest average, as heard": lambda a, t: centroid(heard[a], y[a], heard[t]),
        f"follows the echo, strength blurred {BLUR} dB": lambda a, t: p_fish(fit(E[a], y[a], start[a], blur=BLUR), heard[t]),
        "follows the echo, as heard": lambda a, t: p_fish(fit(E[a], y[a], start[a]), heard[t]),
    }
    st = statistics(E, r, start)
    print("\nEach block held out in turn, recordings right of 42:")
    score = {}
    for n, guess in inputs.items():
        score[n] = right(held_out(guess), y[new], r[new])[0].sum()
        print(f"  {n:38s} {score[n]}", flush=True)
    print("  the last again, other seeds:", [int(right(held_out(lambda a, t: p_fish(fit(E[a], y[a], start[a], s), heard[t])), y[new], r[new])[0].sum()) for s in (1, 2)])

    def tested(blur, gain_db=0, shift=0):
        p = np.zeros(len(y))
        for b in letters:
            a, t = new & (block != b), new & (block == b)
            p[t] = p_fish(models[blur][b], features(cut(E[t] * 10 ** (gain_db / 20), start[t] + shift), MODE))
        return p[new]

    models = {u: {b: fit(E[new & (block != b)], y[new & (block != b)], start[new & (block != b)], blur=u) for b in letters} for u in (0, BLUR)}
    final = {u: fit(E[new], y[new], start[new], blur=u) for u in (0, BLUR)}
    for u, n in ((0, "As heard"), (BLUR, f"Strength blurred {BLUR} dB")):
        base = tested(u)
        ok, mean, truth = right(base, y[new], r[new])
        print(f"\n{n}: {np.mean((base < 0.5) == y[new]):.0%} of 0.4 s examples right; closest calls: fish {mean[truth == 0].min():.2f}, bobblehead {mean[truth == 1].max():.2f}")
        print("  window moved, 5 samples being about 2 cm:", {s: int(right(tested(u, shift=s), y[new], r[new])[0].sum()) for s in (-12, -8, -5, 5, 8, 12)})
        print("  echo made louder or quieter, dB:", {x: int(right(tested(u, gain_db=x), y[new], r[new])[0].sum()) for x in (-6, -3, 3, 6)})
        print(f"  trained on every measured block, on the earlier sets: {right(p_fish(final[u], heard[old]), y[old], r[old])[0].sum()} of {len(np.unique(r[old]))}")
    ok, mean, truth = right(tested(0), y[new], r[new])

    line = np.polyfit(delay[new], d[new], 1)
    empty = level[(block != "") & (y == 2)]
    print(f"\nAll three steps on the measured recordings and empty trials: {len(np.unique(r[new & (level <= DETECT)]))} object(s) missed, "
          f"{np.unique(empty[empty > DETECT]).size} of {np.unique(empty).size} empty trials taken for an object")

    torch.save({"model": final[0].state_dict(), "strength_blurred": final[BLUR].state_dict(), "classes": list(OBJECTS), "pairs": [names[i] for i in cross(names)], "pair_index": cross(names),
                "before": BEFORE, "after": AFTER, "mode": MODE, "detect_db": DETECT, "cm_per_ms": line[0], "cm_at_zero": line[1], "sets": measured},
               Path(__file__).parent / "follow.pt")
    print("saved follow.pt")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), width_ratios=(1, 1.15))
    a = ax[0]
    labels = list(score)
    a.barh(labels, [score[n] for n in labels], color=[BLUE if n in labels[-2:] else GRID for n in labels], height=0.62)
    for i, n in enumerate(labels):
        a.text(score[n] + 0.4, i, str(score[n]), va="center", fontsize=10, color=INK)
    a.axvline(21, color=INK2, lw=1, ls=":")
    a.text(21.3, -0.62, "guessing", fontsize=8.5, color=INK2)
    a.set(title="a  Recordings right of 42, each block held out", xlim=(0, 46))
    a.tick_params(axis="y", length=0)
    a.grid(False)
    a = ax[1]
    a.axhspan(0, 0.5, color=MUTED, alpha=0.08, lw=0)
    a.axhline(0.5, color=INK2, lw=1, ls=":")
    distance = np.array([d[r == k][0] for k in np.unique(r[new])])
    for j, o in enumerate(OBJECTS):
        a.scatter(distance[truth == j] + (j - 0.5) * 0.3, mean[truth == j], s=44, marker="o^"[j], color=COLOUR[o], edgecolor="white", linewidth=0.8, label=o)
    a.set(title="b  What it said of each recording", xlabel="distance placed, cm", ylabel="P(fish), averaged over the recording",
          xticks=sorted(set(distance)), xlim=(8.5, 21.5), ylim=(-0.04, 1.04))
    a.legend(loc="center right", bbox_to_anchor=(1, 0.66))
    fig.tight_layout()
    out = Path(__file__).parent.parent / "docs" / "follow.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)
