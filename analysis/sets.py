"""Sets of three recordings (background, fish, bobblehead): what each object adds to its own background,
and a picture of it. Also the starting point for classify.py.

    python sets.py [folder of set_*.wav] [picture to write]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from scene import GUARD, PRE, load, match

FOLDER = Path(__file__).parent.parent / "recordings" / "sets"
LO, HI = PRE + 10, PRE + 150          # 0.2 to 3.1 ms after the direct sound
OBJECTS = ("fish", "bobblehead")


def read(folder=FOLDER):
    """-> names of the speaker-to-mic pairs, and per set: its id (the time of day it was started), when it was
    made (date and time, which sort in the order the sets were made), the background's average ping, the
    background's own pings, and each object's pings slid onto the background's timing (pairs, pings, W)."""
    files, made, names = defaultdict(dict), {}, None
    for wav in sorted(Path(folder).glob("set_*.wav")):
        _, set_id, _, label = wav.stem.split("_", 3)[0], wav.stem.split("_")[2], None, wav.stem.split("_")[-1]
        files[set_id][label] = wav
        made[set_id] = wav.stem.split("_")[1] + "_" + set_id
    out = []
    for set_id in sorted(files):
        if not all(k in files[set_id] for k in ("background",) + OBJECTS):
            continue
        s, names, _, Q, _ = load(files[set_id]["background"])
        qb = Q[:, ~np.isnan(Q[:, :, 0]).any(axis=0)]
        entry = dict(id=set_id, made=made[set_id], background=qb.mean(1), background_pings=qb, settings=s, slides={})
        for o in OBJECTS:
            _, _, _, Q, _ = load(files[set_id][o])
            entry[o + " as recorded"] = Q[:, ~np.isnan(Q[:, :, 0]).any(axis=0)]
            entry[o], entry["slides"][o] = match(entry[o + " as recorded"], entry["background"], names)
        out.append(entry)
    return names, out


def db(x):
    return 20 * np.log10(np.abs(x) + 1e-12)


def energy(x, win=slice(LO, HI)):
    return 10 * np.log10(np.sum(np.abs(x[..., win]) ** 2, axis=-1) + 1e-20)


if __name__ == "__main__":
    names, sets = read(sys.argv[1] if len(sys.argv) > 1 else FOLDER)
    print(f"{len(sets)} complete sets")
    sig = {o: [s[o].mean(1) - s["background"] for s in sets] for o in OBJECTS}

    print("\nSlide needed to put each recording on its background's timing, samples (bottom speaker, top speaker)")
    for i, s in enumerate(sets):
        print(f"  set {i + 1}: " + "   ".join(f"{o} {s['slides'][o][0]:+.2f} {s['slides'][o][1]:+.2f}" for o in OBJECTS))

    print("\nWhat each object adds to its background, total over 0.2-3.1 ms, dB below the direct sound")
    print("(too soon = what is left in the stretch where no echo can have arrived: the smaller the better)")
    print(f"  {'pair':24s} {'':11s}" + " ".join(f"set {i + 1:<2d}" for i in range(len(sets))) + "     too soon")
    for p, name in enumerate(names):
        for o in OBJECTS:
            print(f"  {name if o == OBJECTS[0] else '':24s} {o:11s}" + " ".join(f"{energy(s[p]):6.1f}" for s in sig[o])
                  + "     " + " ".join(f"{energy(s[p], GUARD):4.0f}" for s in sig[o]))

    if len(sys.argv) > 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from classify import GATE, given
        shown = given(names)
        SURFACE, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
        COLOUR = {"fish": "#2a78d6", "bobblehead": "#eb6834"}
        plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                             "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold",
                             "axes.titlelocation": "left", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
        ms = (np.arange(PRE, PRE + 170) - PRE) / 48
        fig, ax = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
        for p, name in enumerate(names):
            a = ax.flat[p]
            # the background against itself: alternate groups of four pings (one turn of the phone's four-ping pattern)
            halves = []
            for s in sets:
                q = s["background_pings"][p]
                blocks = q[: len(q) // 4 * 4].reshape(-1, 4, q.shape[-1]).mean(1)
                halves.append(np.abs(blocks[::2].mean(0) - blocks[1::2].mean(0)))
            a.fill_between(ms, -100, db(np.mean(halves, axis=0)[PRE:PRE + 170]), color=GRID, lw=0, label="background against itself")
            for o in OBJECTS:
                for j, s in enumerate(sig[o]):
                    a.plot(ms, db(s[p][PRE:PRE + 170]), color=COLOUR[o], lw=1.3, alpha=0.75, label=f"{o}, one line per set" if j == 0 else None)
            if p in shown:   # the stretch of this pair the network is given
                for edge in (10 / 48, (10 + GATE) / 48):
                    a.axvline(edge, color=INK2, lw=1, ls=(0, (3, 3)))
                a.text((20 + GATE) / 96, -27, "given to the network", color=INK2, ha="center", va="top", fontsize=9)
            a.set(title=name + ("" if p in shown else "   (used for timing only)"), xlim=(0, 3.5), ylim=(-85, -25))
            a.grid(color=GRID, lw=0.8)
            a.set_axisbelow(True)
            for sp in ("top", "right"): a.spines[sp].set_visible(False)
            if p >= 2: a.set_xlabel("ms after the speaker's direct sound")
            if p % 2 == 0: a.set_ylabel("added by the object, dB below the direct sound")
        ax.flat[1].legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(sys.argv[2], dpi=80)
        print("\nwrote", sys.argv[2])
