"""The signal chain on the real recordings pulled off the phone, written to docs/steps/real/.

    python steps_real.py [set id to show the steps on]

Needs the recordings in recordings/sets (they are not in the repository). Unlike classify.load(), this does
not rewrite echoes.npz.
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy import signal

import steps  # noqa: F401  (house style for the figures)
from classify import features
from model import EchoCNN
from scene import GUARD, PRE, W, load
from scene import blocks as to_blocks
from sets import FOLDER, OBJECTS, read
from steps import AXIS, BLUE, COLOUR, GREEN, GRID, INK2, MUTED, ORANGE, PURPLE, db, note

OUT = steps.OUT / "real"
MS = (np.arange(W) - PRE) / 48


def save(fig, name):
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)
    print("wrote", OUT / name)


def wav(set_id, label):
    return next(FOLDER.glob(f"set_*_{set_id}_*_{label}.wav"))


def raw(set_id):
    """What the mics recorded: one cycle in time, and a spectrogram of a few."""
    path = wav(set_id, "background")
    s = json.loads(path.with_suffix(".json").read_text())
    x, fs = sf.read(str(path), always_2d=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    a = ax[0]
    t0 = fs  # one second in
    t = np.arange(4800) / fs * 1000
    for ch, col in ((1, BLUE), (0, ORANGE)):
        a.plot(t, x[t0:t0 + 4800, ch], color=col, lw=0.5, alpha=0.85)
        a.plot([], [], color=col, lw=2.5, label=f"{s['channel_mics'][ch]} mic")
    a.set(title="2a  One 100 ms cycle as the mics recorded it", xlabel="ms", ylabel="sample value (full scale = 1)")
    a.legend(loc="upper left")
    note(a, f"background of set {set_id}", top=False)
    a = ax[1]
    f, tt, S = signal.spectrogram(x[t0:t0 + 3 * 4800, 1], fs, nperseg=128, noverlap=112, nfft=512)
    a.pcolormesh(tt * 1000, f / 1000, 10 * np.log10(S + 1e-15), cmap="Greys", vmin=-120, vmax=-50, shading="auto")
    a.set(title="2b  Bottom mic, three cycles: pings above 18 kHz, room noise below 5", xlabel="ms", ylabel="kHz", ylim=(0, 24))
    a.grid(False)
    fig.tight_layout()
    save(fig, "2_raw.png")


def profiles(set_id):
    """After the matched filter and the window: every ping of the background, per pair."""
    _, names, tt, Q, direct = load(wav(set_id, "background"))
    fig, ax = plt.subplots(2, 2, figsize=(15, 7.4), sharex=True, sharey=True)
    for p, name in enumerate(names):
        a = ax.flat[p]
        good = ~np.isnan(Q[p, :, 0])
        a.plot(MS, db(Q[p, good]).T, color=BLUE, lw=0.3, alpha=0.08)
        a.plot(MS, db(np.mean(Q[p, good], 0)), color=INK2, lw=1.2, label="average of all pings")
        a.axvspan((GUARD.start - PRE) / 48, (GUARD.stop - PRE) / 48, color=PURPLE, alpha=0.15, lw=0)
        a.set(title=name, xlim=(-1, 9), ylim=(-75, 25))
        if p >= 2: a.set_xlabel("ms after the speaker's direct sound at the near mic")
        if p % 2 == 0: a.set_ylabel("dB relative to that direct sound")
    ax.flat[0].legend(loc="upper right")
    fig.suptitle(f"3-5  Matched filter, window, divided by the direct sound: background of set {set_id}, {good.sum()} pings (faint lines)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "5_profiles.png")
    return names, direct


def four_ping(direct, set_id):
    """The phone's own level, ping by ping: drifts, and repeats every four."""
    lv = direct / np.nanmean(direct, axis=1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]
    for k, (name, col) in enumerate((("bottom speaker", BLUE), ("top speaker", ORANGE))):
        a.plot(100 * (lv[k] - 1), color=col, lw=1, marker="o", ms=2.5, label=name)
    a.set(title="8a  Direct sound level, ping by ping", xlabel="ping", ylabel="% from its average")
    a.legend(loc="upper right")
    a = ax[1]
    for k, (name, col) in enumerate((("bottom speaker", BLUE), ("top speaker", ORANGE))):
        v = lv[k][~np.isnan(lv[k])]
        v = v - v.mean()
        spec = np.abs(np.fft.rfft(v * np.hanning(len(v))))
        a.plot(np.fft.rfftfreq(len(v)), spec / spec.max(), color=col, lw=1.3, label=name)
    a.axvline(0.25, color=INK2, ls="--", lw=1)
    a.text(0.26, 0.92, "once every 4 pings", fontsize=9, color=INK2)
    a.set(title="8b  How often it repeats", xlabel="cycles per ping", ylabel="relative strength")
    note(a, f"background of set {set_id}")
    fig.tight_layout()
    save(fig, "8_four_ping.png")


def timing(names, sets, set_id):
    s = next(x for x in sets if x["id"] == set_id)
    cross = [i for i, n in enumerate(names) if n.split(" spk > ")[0] != n.split(" spk > ")[1].replace(" mic", "")]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]
    p = cross[0]
    before = np.nanmean(s["fish as recorded"][p], 0) - s["background"][p]
    after = np.nanmean(s["fish"][p], 0) - s["background"][p]
    a.plot(MS, db(before), color=MUTED, lw=1.1, label="fish minus background, as recorded")
    a.plot(MS, db(after), color=BLUE, lw=1.4, label=f"after the slide ({s['slides']['fish'][0]:+.2f} samples)")
    a.axvspan((GUARD.start - PRE) / 48, (GUARD.stop - PRE) / 48, color=PURPLE, alpha=0.15, lw=0)
    a.set(title=f"7a  {names[p]}, set {set_id}", xlabel="ms after the direct sound", ylabel="dB relative to the direct sound",
          xlim=(-0.5, 3), ylim=(-90, -10))
    a.legend(loc="upper right")
    a = ax[1]
    for k, (spk, col) in enumerate((("bottom", BLUE), ("top", ORANGE))):
        for j, o in enumerate(OBJECTS):
            v = [x["slides"][o][k] for x in sets]
            a.scatter(np.arange(len(sets)) + (k - 0.5) * 0.25, v, s=22, color=col, marker="o" if j == 0 else "s",
                      label=f"{spk} speaker, {o}", alpha=0.85)
    a.axhline(0, color=AXIS, lw=1)
    a.set_xticks(range(len(sets)), [x["id"][-4:] for x in sets], rotation=90, fontsize=8)
    a.set(title="7b  Slide found for every recording", ylabel="samples", xlabel="set")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=8, ncol=4)
    fig.tight_layout()
    save(fig, "7_timing.png")


def echoes(names, sets, new):
    cross = [i for i, n in enumerate(names) if n.split(" spk > ")[0] != n.split(" spk > ")[1].replace(" mic", "")]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for a, p in zip(ax, cross):
        for s in sets:
            for o in OBJECTS:
                e = np.nanmean(s[o], 1)[p] - s["background"][p]
                hot = s["id"] in new
                a.plot(MS, db(e), color=COLOUR[o], lw=2.2 if hot else 0.9, alpha=1 if hot else 0.45,
                       ls="-" if not hot else (0, (4, 1.5)))
        for edge in (10 / 48, 62 / 48):
            a.axvline(edge, color=INK2, lw=1, ls=(0, (3, 3)))
        a.set(title=names[p], xlim=(0, 3.1), ylim=(-85, -25), xlabel="ms after the direct sound")
    ax[0].set_ylabel("added by the object, dB relative to the direct sound")
    for o in OBJECTS:
        ax[1].plot([], [], color=COLOUR[o], lw=1, label=f"{o}, one line per set")
        ax[1].plot([], [], color=COLOUR[o], lw=2.2, ls=(0, (4, 1.5)), label=f"{o}, the {len(new)} sets the saved model was not trained on")
    ax[1].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("8-9  What each object added to its own background; between the dashed lines is what the network sees",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "9_echoes.png")


def guesses(names, sets, new):
    """The phone's own answers, from the .json beside each recording, against the PC running the same model."""
    ck = torch.load(steps.Path(__file__).parent / "fish_or_bobblehead.pt", weights_only=False)
    net = EchoCNN(len(ck["classes"]), n_paths=len(ck["pair_index"]))
    net.load_state_dict(ck["model"]); net.eval()
    rows = []
    for s in sets:
        for j, o in enumerate(OBJECTS):
            e, _ = to_blocks(s[o + " as recorded"], s["background"], names)
            X = features(e[:, ck["pair_index"], PRE + 10: PRE + 10 + ck["bins"]], ck["mode"])
            with torch.no_grad():
                pc = torch.softmax(net(torch.from_numpy(X)), 1)[:, 0].numpy()
            meta = json.loads(wav(s["id"], o).with_suffix(".json").read_text())
            rows.append((s["id"], o, pc, meta.get("p_fish")))
    fig, ax = plt.subplots(figsize=(15, 5))
    for k, (sid, o, pc, phone) in enumerate(rows):
        x = k // 2 + (0.18 if o == OBJECTS[1] else -0.18)
        trained = sid in ck["sets"]
        ax.scatter(np.full(len(pc), x) + 0.08 * (np.random.default_rng(k).random(len(pc)) - 0.5), pc, s=6, color=COLOUR[o], alpha=0.3, lw=0)
        ax.plot([x - 0.14, x + 0.14], [pc.mean()] * 2, color=COLOUR[o], lw=2.4, alpha=0.5 if trained else 1)
        if phone is not None:
            right = (phone >= 0.5) == (o == "fish")
            ax.scatter([x], [phone], s=60, marker="D", facecolor="none" if right else "#d03030", edgecolor=INK2, zorder=5, lw=1)
    n = len(rows) // 2
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    first_new = [i for i, s in enumerate(sets) if s["id"] in new]
    for i in first_new:
        ax.axvspan(i - 0.5, i + 0.5, color=GREEN, alpha=0.1, lw=0)
    if first_new:
        ax.text((first_new[0] + first_new[-1]) / 2, 1.06, "not in the saved model's training data", ha="center", fontsize=8.5, color=GREEN)
    ax.set_xticks(range(n), [s["id"][-4:] for s in sets], fontsize=7.5, rotation=90)
    ax.set(xlim=(-0.6, n - 0.4), ylim=(-0.04, 1.14), ylabel="P(fish)", xlabel="set")
    ax.scatter([], [], marker="D", facecolor="none", edgecolor=INK2, label="the phone's answer at the time, from the model then on it: right")
    ax.scatter([], [], marker="D", facecolor="#d03030", edgecolor=INK2, label="the phone's answer at the time: wrong")
    ax.plot([], [], color=INK2, lw=2.4, label=f"the saved model, trained on {len(ck['sets'])} sets, run on the PC (faded: set in its training data)")
    ax.legend(loc="center left", fontsize=8.5, bbox_to_anchor=(0, 0.62))
    ax.set_title("11  Every recording: what the phone said at the time, and what the saved model says on the PC")
    fig.tight_layout()
    save(fig, "11_guesses.png")
    for sid, o, pc, phone in rows:
        if sid in new:
            print(f"  {sid} {o:10s} PC P(fish) {pc.mean():.3f}   phone {phone}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names, sets = read()
    ids = [s["id"] for s in sets]
    # the folder name gives only the time of day; put the sets in the order they were made
    sets = sorted(sets, key=lambda s: s["made"])
    # "new": the sets the saved model (fish_or_bobblehead.pt) was not trained on
    trained_on = torch.load(steps.Path(__file__).parent / "fish_or_bobblehead.pt", weights_only=False)["sets"]
    new = [s["id"] for s in sets if s["id"] not in trained_on]
    show = sys.argv[1] if len(sys.argv) > 1 else sets[0]["id"]
    print(f"{len(sets)} sets; showing the steps on {show}; not in the saved model's training data: {new}")
    raw(show)
    names, direct = profiles(show)
    four_ping(direct, show)
    timing(names, sets, show)
    echoes(names, sets, new)
    guesses(names, sets, new)
