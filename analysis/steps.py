"""Pictures of each step of the signal chain, written to docs/steps/.

    python steps.py

The raw recordings are not in the repository, so the steps before the echoes (the ping, the matched filter,
the window, the timing correction) are shown on a simulated recording made with the app's own settings: a
direct sound, a few reflections from the room, a small echo from the object, and noise. The steps from the
echoes on use the real echoes in echoes.npz.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import signal

from classify import GATE, cnn, cross, features, load
from scene import GUARD, POST, PRE, W, slid
from sets import LO, OBJECTS

OUT = Path(__file__).parent.parent / "docs" / "steps"
FS, F0, F1, CHIRP, FADE, CYCLE = 48000, 18000, 22500, 480, 96, 4800

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#22211f", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#2f9e6e", "#7a5bd1"
COLOUR = {"fish": BLUE, "bobblehead": ORANGE}
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def chirp(up=True):
    """The app's chirp, complex: its real part is what is played, the whole of it is the matched filter."""
    i = np.arange(CHIRP)
    t, T = i / FS, CHIRP / FS
    fa, fb = (F0, F1) if up else (F1, F0)
    edge = np.minimum(i, CHIRP - 1 - i)
    env = np.where(edge < FADE, 0.5 - 0.5 * np.cos(np.pi * edge / FADE), 1.0)
    return env * np.exp(2j * np.pi * (fa * t + (fb - fa) * t * t / (2 * T))), env


def placed(c, at, n):
    """A real chirp starting at sample `at` (may be a fraction) in a buffer of n samples."""
    x = np.zeros(n)
    x[:CHIRP] = c.real
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * at), n)


# A made-up scene, in ms after the direct sound and amplitude relative to it. Close coupling through the phone
# and stand, then floor and walls further out; the object 10 cm away echoes inside 0.2 to 1.3 ms.
ROOM = [(0.12, 0.25), (2.1, 0.05), (3.6, 0.025), (6.2, 0.012)]
FISH = [(0.62, 0.03)]
FIGURE = [(0.52, 0.009), (0.70, 0.011), (0.93, 0.008)]
START = 400   # where the direct sound lands in the simulated buffer, samples


def recording(paths, offset=0.0, noise=3e-4, seed=0):
    """Simulated mic signal for one ping: the direct sound plus each (delay ms, amplitude), all moved by offset samples."""
    rng = np.random.default_rng(seed)
    c, _ = chirp(True)
    n = 2 * CYCLE
    x = placed(c, START + offset, n)
    for ms, a in paths:
        x += a * placed(c, START + offset + ms * FS / 1000, n)
    return x + noise * rng.standard_normal(n)


def profile(x):
    """Matched filter and window, as the app does it: -1 to +9 ms around the direct sound, divided by it."""
    c, _ = chirp(True)
    m = signal.correlate(x, c, mode="valid")
    w = m[START - PRE: START + POST]
    return w / w[PRE], m


def db(x):
    return 20 * np.log10(np.abs(x) + 1e-12)


def save(fig, name):
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)
    print("wrote", OUT / name)


def note(ax, text, top=False):
    ax.text(0.99, 0.98 if top else 0.02, text, transform=ax.transAxes, ha="right", va="top" if top else "bottom",
            fontsize=8.5, color=MUTED, style="italic")


def pipeline():
    """The chain as boxes, in two rows."""
    steps = [("1  Ping", "18 to 22.5 kHz chirps, 10 ms;\nspeakers alternate, up / down", "tx"),
             ("2  Capture", "both mics, 48 kHz 16 bit;\n3 s to settle, then 10 s", "tx"),
             ("3  Matched filter", "correlate each mic with\neach speaker's chirp", "ping"),
             ("4  Find the pings", "locate each direct sound;\nfind it again if lost", "ping"),
             ("5  Window, normalise", "-1 to +9 ms, divided by the\ndirect sound at the near mic", "ping"),
             ("6  Reject", "clipped, relocked, or\nfewer than 90 pings", "ping"),
             ("7  Timing fix", "fractional slide onto the\nbackground, 0.1 to 0.4 ms", "set"),
             ("8  Average 4, subtract", "four-ping blocks minus\nthe empty scene", "set"),
             ("9  Gate", "the two cross paths,\n0.2 to 1.3 ms", "model"),
             ("10  Features", "dB, own loudness removed,\n-50..+10 dB to 0..1", "model"),
             ("11  CNN, average", "one score per block,\naveraged over ~25", "model")]
    group = {"tx": ("Transmit and record", GREEN), "ping": ("Every ping", BLUE),
             "set": ("Against the background", PURPLE), "model": ("Model", ORANGE)}
    fig, ax = plt.subplots(figsize=(15, 5.6))
    ax.set(xlim=(0, 6), ylim=(0, 2.3)); ax.axis("off"); ax.grid(False)
    bw, bh = 0.86, 0.62
    pos = [(i + 0.5, 1.6) for i in range(6)] + [(5.5 - i, 0.55) for i in range(5)]
    for k, ((title, body, g), (x, y)) in enumerate(zip(steps, pos)):
        col = group[g][1]
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=col + "1a", ec=col, lw=1.4))
        ax.text(x, y + 0.17, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x, y - 0.08, body, ha="center", va="center", fontsize=8.6, color=INK2, linespacing=1.3)
        if k + 1 < len(steps):
            (x2, y2) = pos[k + 1]
            if y2 == y:
                a, b = ((x + bw / 2, y), (x2 - bw / 2, y)) if x2 > x else ((x - bw / 2, y), (x2 + bw / 2, y))
            else:
                a, b = (x, y - bh / 2), (x2, y2 + bh / 2)
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=MUTED, lw=1.3))
    for i, (name, col) in enumerate(group.values()):
        ax.add_patch(FancyBboxPatch((0.15 + 1.45 * i, 2.12), 0.12, 0.09, boxstyle="round,pad=0.01", fc=col + "1a", ec=col))
        ax.text(0.33 + 1.45 * i, 2.165, name, va="center", fontsize=9.5, color=INK2)
    ax.text(5.97, 0.05, "Sonar.java does 1 to 6 while recording; Judge.java does 7 to 11 once it stops",
            ha="right", fontsize=9, color=MUTED, style="italic")
    save(fig, "0_pipeline.png")


def ping():
    c, env = chirp(True)
    t = np.arange(CHIRP) / FS * 1000
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.1, 1.3, 1]})
    a = ax[0]
    a.plot(t, c.real, color=BLUE, lw=0.5)
    a.plot(t, env, color=INK2, lw=1.2, ls="--")
    a.plot(t, -env, color=INK2, lw=1.2, ls="--")
    a.annotate("2 ms raised-cosine fade", xy=(1.0, -0.5), xytext=(2.6, -1.25), fontsize=8.5, color=INK2,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a.set(title="1a  One chirp: 10 ms, 480 samples", xlabel="ms", ylabel="amplitude", ylim=(-1.35, 1.1))
    ins = a.inset_axes([0.62, 0.68, 0.36, 0.28])
    ins.plot(t[:60], c.real[:60], color=BLUE, lw=1); ins.set_xticks([]); ins.set_yticks([]); ins.grid(False)
    ins.set_title("first 1.25 ms", fontsize=8, fontweight="normal", color=MUTED)

    # two cycles of what the speakers play: bottom (up) at 0, top (down) half way, at 0.7 x
    n = 2 * CYCLE
    x = np.zeros(n)
    cd, _ = chirp(False)
    for k in range(2):
        x[k * CYCLE: k * CYCLE + CHIRP] += c.real
        x[k * CYCLE + CYCLE // 2: k * CYCLE + CYCLE // 2 + CHIRP] += 0.7 * cd.real
    f, tt, S = signal.spectrogram(x, FS, nperseg=128, noverlap=112, nfft=512)
    a = ax[1]
    a.pcolormesh(tt * 1000, f / 1000, 10 * np.log10(S + 1e-12), cmap="Greys", vmin=-90, vmax=-40, shading="auto")
    a.set(title="1b  What the speakers play: two 100 ms cycles", xlabel="ms", ylabel="kHz", ylim=(16, 24.5))
    a.grid(False)
    for k in range(2):
        a.text(k * 100 + 12, 18.2, "bottom spk\nup", color=BLUE, fontsize=8.5, ha="left")
        a.text(k * 100 + 62, 21.6, "top spk\ndown", color=ORANGE, fontsize=8.5, ha="left")
    a.axhline(24, color=MUTED, lw=0.8, ls=":"); a.text(198, 24.1, "24 kHz: half the sampling rate", ha="right", fontsize=8, color=MUTED)

    # why up and down: the matched filter of one barely responds to the other
    a = ax[2]
    lag = np.arange(-CHIRP + 1, CHIRP) / FS * 1000
    same = np.abs(np.correlate(c.real, c, "full")); other = np.abs(np.correlate(cd.real, c, "full"))
    a.plot(lag, db(same / same.max()), color=BLUE, lw=1.2, label="up chirp through up filter")
    a.plot(lag, db(other / same.max()), color=ORANGE, lw=1.2, label="down chirp through up filter")
    a.set(title="1c  Opposite sweeps stay apart", xlabel="lag, ms", ylabel="dB", ylim=(-60, 3), xlim=(-10, 10))
    a.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "1_ping.png")


def matched_filter():
    scene = ROOM + FISH
    x = recording(scene)
    q, m = profile(x)
    t = (np.arange(len(x)) - START) / FS * 1000
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    a = ax[0]
    a.plot(t, x, color=INK2, lw=0.4)
    a.set(title="3a  What a mic hears (simulated)", xlabel="ms after the direct sound starts", ylabel="amplitude", xlim=(-2, 14))
    note(a, "direct sound, room and object all overlap")
    a = ax[1]
    tm = (np.arange(len(m)) - START) / FS * 1000
    a.plot(tm, db(m / np.abs(m).max()), color=BLUE, lw=0.9)
    for ms, amp in scene:
        col = GREEN if (ms, amp) in FISH else MUTED
        a.annotate("object" if (ms, amp) in FISH else "room", (ms, db(amp) + 2), color=col, fontsize=8, ha="center")
    a.annotate("direct", (0, 2), fontsize=8, ha="center", color=INK2)
    a.set(title="3b  After the matched filter", xlabel="ms after the direct sound", ylabel="dB below the direct sound",
          xlim=(-1, 9), ylim=(-80, 6))
    note(a, "each 10 ms chirp is now a sharp peak", top=True)
    a = ax[2]
    k = np.arange(-24, 25)
    c, _ = chirp(True)
    ac = np.abs(np.correlate(c.real, c, "full"))
    a.plot(k / FS * 1000, ac[CHIRP - 1 + k] / ac[CHIRP - 1], color=BLUE, lw=1.4, marker="o", ms=3)
    a.axvspan(-0.5 / 4.5, 0.5 / 4.5, color=GRID, lw=0)
    a.set(title="3c  Peak width sets the resolution", xlabel="ms", ylabel="relative magnitude")
    a.text(0, 0.05, "1 / 4.5 kHz = 0.22 ms\n= 3.8 cm of range", ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "3_matched_filter.png")


def window_normalise():
    """The window, and what dividing by the direct sound does to drift from ping to ping."""
    rng = np.random.default_rng(1)
    pings = 100
    gain = 1 + 0.03 * np.cumsum(rng.standard_normal(pings)) / np.sqrt(pings)
    phase = 0.4 * np.cumsum(rng.standard_normal(pings)) / np.sqrt(pings)
    q0, m = profile(recording(ROOM + FISH))
    raw = m[START - PRE: START + POST]
    k = PRE + int(0.62 * FS / 1000)   # the object's delay
    before = np.array([g * np.exp(1j * p) * raw[k] for g, p in zip(gain, phase)])
    after = np.array([(g * np.exp(1j * p) * raw[k]) / (g * np.exp(1j * p) * raw[PRE]) for g, p in zip(gain, phase)])

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    a = ax[0]
    ms = (np.arange(W) - PRE) / FS * 1000
    a.plot(ms, db(q0), color=BLUE, lw=1)
    a.axvspan(GUARD.start / FS * 1000 - PRE / FS * 1000, GUARD.stop / FS * 1000 - PRE / FS * 1000, color=PURPLE, alpha=0.15, lw=0)
    a.axvspan(10 / FS * 1000, (10 + GATE) / FS * 1000, color=ORANGE, alpha=0.12, lw=0)
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=PURPLE, alpha=0.3, label="timing guard, 0.1 to 0.4 ms (step 7)"),
                      Patch(color=ORANGE, alpha=0.25, label="what the network sees, 0.2 to 1.3 ms (step 9)")], loc="upper right")
    a.set(title="5a  The window kept per ping: -1 to +9 ms, 480 samples", xlabel="ms after the direct sound",
          ylabel="dB relative to the direct sound", xlim=(-1, 9), ylim=(-80, 12))
    note(a, "simulated")
    a = ax[1]
    i = np.arange(pings)
    before, after = before / before[0], after / after[0]
    a.plot(i, np.degrees(np.angle(before)), color=MUTED, lw=1.2, label="as received")
    a.plot(i, np.degrees(np.angle(after)), color=BLUE, lw=1.6, label="divided by the direct sound")
    a.set(title="5b  Phase of the object's echo over 100 pings", xlabel="ping", ylabel="degrees, from the first ping")
    a.legend(loc="lower left")
    note(a, "simulated drift in gain and phase: it cancels")
    fig.tight_layout()
    save(fig, "5_window.png")


def timing():
    """Two recordings of the same scene that start a fraction of a sample apart, before and after the slide."""
    offset = 0.37
    B, _ = profile(recording(ROOM, seed=2))
    Q, _ = profile(recording(ROOM + FISH, offset=offset, seed=3))
    truth, _ = profile(recording(ROOM + FISH, seed=4))
    grid = np.arange(-2.5, 2.5001, 0.01)
    cost = [np.sum(np.abs(slid(Q, d)[GUARD] / slid(Q, d)[PRE] - B[GUARD]) ** 2) for d in grid]
    d = grid[int(np.argmin(cost))]
    fixed = slid(Q, d) / slid(Q, d)[PRE]
    ms = (np.arange(W) - PRE) / FS * 1000

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    a = ax[0]
    a.plot(ms, db(Q - B), color=MUTED, lw=1.1, label=f"object minus background, as recorded ({offset} samples late)")
    a.plot(ms, db(fixed - B), color=BLUE, lw=1.4, label="the same after the slide")
    a.plot(ms, db(truth - B), color=GREEN, lw=1, ls="--", label="the object's echo alone")
    a.axvspan(GUARD.start / FS * 1000 - PRE / FS * 1000, GUARD.stop / FS * 1000 - PRE / FS * 1000, color=PURPLE, alpha=0.15, lw=0)
    a.set(title="7a  A fraction of a sample hides the echo", xlabel="ms after the direct sound",
          ylabel="dB relative to the direct sound", xlim=(-0.5, 3), ylim=(-100, 0))
    a.legend(loc="upper right", fontsize=8.5)
    note(a, "simulated")
    a = ax[1]
    a.semilogy(grid, cost, color=PURPLE, lw=1.4)
    a.axvline(d, color=INK2, lw=1, ls="--")
    a.text(d + 0.08, max(cost) / 3, f"best slide {d:+.2f}", color=INK2, fontsize=9)
    a.set(title="7b  Mismatch in the guard, 0.1 to 0.4 ms", xlabel="slide tried, samples", ylabel="mismatch")
    note(a, "searched from -2.5 to +2.5 in 0.01 steps")
    fig.tight_layout()
    save(fig, "7_timing.png")


def blocks_and_background(names, E, y):
    """Real echoes: what each object added, per pair, and the part the network is given."""
    shown = cross(names)
    ms = (np.arange(E.shape[2]) + LO - PRE) / FS * 1000
    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.9, 1, 1])
    a = fig.add_subplot(gs[:, 0]); a.axis("off"); a.grid(False)
    a.set_title("8  Average four, subtract the empty scene", loc="left")
    # pings -> blocks of four -> minus background
    for i in range(16):
        r = i // 4
        a.add_patch(FancyBboxPatch((0.05 + 0.055 * i + 0.03 * r, 0.80), 0.04, 0.07, boxstyle="round,pad=0.003", fc=BLUE + "33", ec=BLUE, lw=0.8))
    a.text(0.5, 0.91, "pings, 10 a second from each speaker", ha="center", fontsize=9, color=INK2)
    for r in range(4):
        x0 = 0.05 + 0.25 * r
        a.add_patch(FancyArrowPatch((x0 + 0.1, 0.78), (x0 + 0.1, 0.66), arrowstyle="-|>", mutation_scale=10, color=MUTED))
        a.add_patch(FancyBboxPatch((x0 + 0.02, 0.56), 0.16, 0.09, boxstyle="round,pad=0.005", fc=PURPLE + "22", ec=PURPLE))
        a.text(x0 + 0.1, 0.605, "mean of 4", ha="center", va="center", fontsize=8.5)
    a.text(0.5, 0.50, "the phone's level repeats every four pings,\nso an average of four cancels it", ha="center", fontsize=9, color=INK2)
    a.text(0.5, 0.38, "each block, slid (step 7), minus the\nbackground recorded just before", ha="center", fontsize=9, color=INK2)
    a.add_patch(FancyBboxPatch((0.15, 0.17), 0.7, 0.12, boxstyle="round,pad=0.01", fc=GREEN + "1a", ec=GREEN))
    a.text(0.5, 0.23, "one example = what the object added\n~25 per 10 s recording", ha="center", va="center", fontsize=9.5)
    a.set(xlim=(0, 1.05), ylim=(0.1, 0.95))
    for p, name in enumerate(names):
        a = fig.add_subplot(gs[p // 2, 1 + p % 2])
        for j, o in enumerate(OBJECTS):
            v = db(E[y == j, p])
            med, lo, hi = np.median(v, 0), np.percentile(v, 15, 0), np.percentile(v, 85, 0)
            a.fill_between(ms, lo, hi, color=COLOUR[o], alpha=0.18, lw=0)
            a.plot(ms, med, color=COLOUR[o], lw=1.4, label=o)
        if p in shown:
            for edge in (10 / 48, (10 + GATE) / 48):
                a.axvline(edge, color=INK2, lw=1, ls=(0, (3, 3)))
        a.set(title=name + ("  (step 9 keeps 0.2 to 1.3 ms)" if p in shown else "  (timing only)"), xlim=(0.2, 3.1), ylim=(-85, -30))
        if p >= 2: a.set_xlabel("ms after the direct sound")
        if p % 2 == 0: a.set_ylabel("dB relative to the direct sound")
        if p == 1: a.legend(loc="upper right")
    fig.text(0.99, 0.005, f"real data: {len(E)} examples from echoes.npz; line = median, band = 15th to 85th percentile",
             ha="right", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, "8_echoes.png")


def feature_view(names, E, y):
    pairs = cross(names)
    X = features(E[:, pairs, :GATE], "each")
    ms = (np.arange(GATE) + 10) / FS * 1000
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2])
    for j, o in enumerate(OBJECTS):
        k = np.where(y == j)[0][5]
        a = fig.add_subplot(gs[j, 0]); a.grid(False)
        a.imshow(X[k], aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=(ms[0], ms[-1], 1.5, -0.5))
        a.set_yticks([0, 1], [names[i].replace(" spk > ", " > ") for i in pairs], fontsize=8.5)
        a.set_title(f"10a  One {o} example, 2 x 52", color=COLOUR[o])
        if j == 1: a.set_xlabel("ms after the direct sound")
    for i, p in enumerate(pairs):
        a = fig.add_subplot(gs[i, 1])
        for j, o in enumerate(OBJECTS):
            v = X[y == j, i]
            a.fill_between(ms, np.percentile(v, 15, 0), np.percentile(v, 85, 0), color=COLOUR[o], alpha=0.18, lw=0)
            a.plot(ms, np.median(v, 0), color=COLOUR[o], lw=1.4, label=o)
        a.set(title="10b  " + names[p].replace(" spk > ", " > "), ylim=(0, 1))
        a.set_ylabel("feature (0..1)")
        if i == 1: a.set_xlabel("ms after the direct sound")
        if i == 0: a.legend(loc="lower right")
    a = fig.add_subplot(gs[:, 2]); a.grid(False); a.axis("off")
    a.set_title("10c  From echo to feature, per path", loc="left")
    rows = [("echo power along delay", "|echo|² at each of 52 delays"),
            ("in decibels", "20 log10 |echo|"),
            ("own loudness out", "minus 10 log10 of the path's mean power"),
            ("clip and scale", "-50..+10 dB  ->  0..1")]
    for r, (head, body) in enumerate(rows):
        yy = 0.88 - 0.21 * r
        a.add_patch(FancyBboxPatch((0.05, yy - 0.07), 0.9, 0.14, boxstyle="round,pad=0.01", fc=ORANGE + "18", ec=ORANGE))
        a.text(0.5, yy + 0.025, head, ha="center", fontsize=10, fontweight="bold", color=INK)
        a.text(0.5, yy - 0.03, body, ha="center", fontsize=9, color=INK2)
        if r < len(rows) - 1:
            a.add_patch(FancyArrowPatch((0.5, yy - 0.075), (0.5, yy - 0.145), arrowstyle="-|>", mutation_scale=10, color=MUTED))
    a.text(0.5, 0.06, "removing loudness means the network must use\nhow the echo is spread over delay, not how strong it is",
           ha="center", fontsize=9, color=MUTED, style="italic")
    a.set(xlim=(0, 1), ylim=(0, 1))
    fig.text(0.99, 0.005, "real data from echoes.npz", ha="right", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, "10_features.png")


def classification(names, ids, E, y, g, r):
    """Each set held out in turn: the CNN's score for every example, and each recording's average."""
    X = features(E[:, cross(names), :GATE], "each")
    p = np.zeros(len(y))
    for held in np.unique(g):
        t = g == held
        p[t] = cnn(X[~t], y[~t], X[t], 30)
    fig, ax = plt.subplots(figsize=(15, 4.8))
    rng = np.random.default_rng(0)
    for k in np.unique(r):
        s = r == k
        o = y[s][0]
        xs = k + 0.25 * (rng.random(s.sum()) - 0.5)
        ax.scatter(xs, p[s], s=9, color=COLOUR[OBJECTS[o]], alpha=0.45, lw=0)
        m = p[s].mean()
        right = (m >= 0.5) == (o == 0)
        ax.plot([k - 0.32, k + 0.32], [m, m], color=COLOUR[OBJECTS[o]] if right else "#d03030", lw=2.6)
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.set_xticks(np.arange(0, len(np.unique(r)), 2) + 0.5, [i[-4:] for i in ids], fontsize=8.5)
    ax.set(xlim=(-0.7, len(np.unique(r)) - 0.3), ylim=(-0.04, 1.04), ylabel="P(fish)")
    ax.set_title("11  The CNN's score per four-ping example (dots) and per recording (bars, the app's answer)")
    for o in OBJECTS:
        ax.scatter([], [], color=COLOUR[o], label=o)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    rec_right = sum((p[r == k].mean() >= 0.5) == (y[r == k][0] == 0) for k in np.unique(r))
    ax.set_xlabel(f"set (each held out in turn)   -   real data: {rec_right} of {len(np.unique(r))} recordings right, {np.mean((p >= 0.5) == (y == 0)):.0%} of examples")
    fig.tight_layout()
    save(fig, "11_classify.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    ping()
    matched_filter()
    window_normalise()
    timing()
    names, ids, E, y, g, r = load()
    blocks_and_background(names, E, y)
    feature_view(names, E, y)
    classification(names, ids, E, y, g, r)
