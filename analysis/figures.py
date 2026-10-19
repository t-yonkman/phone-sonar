"""The figures for the write-up (docs/writeup.md), written to docs/figures/.    python figures.py

1_what_the_mics_hear.png   one cycle of a background recording as the two mics recorded it, and its spectrogram
2_every_cycle.png          every cycle of the empty scene, the fish and the bobblehead from one set
3_echo_shape.png           three ways to describe the object's echo that do not depend on when it arrived

Needs the recordings in recordings/sets. Prints the numbers the write-up quotes.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

import steps
from placement import echoes
from scene import PRE, W
from sets import FOLDER, OBJECTS
from shape import AROUND, LAGS, REF, aligned, autocorr, middles, stats
from steps import AXIS, BLUE, COLOUR, INK, INK2, MUTED, ORANGE

OUT = Path(__file__).parent.parent / "docs" / "figures"
SHOWN = "185816"   # the set whose background figure 1 shows


def save(fig, name):
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)
    print("wrote", OUT / name)


def what_the_mics_hear():
    path = next(FOLDER.glob(f"set_*_{SHOWN}_1_background.wav"))
    s = json.loads(path.with_suffix(".json").read_text())
    x, fs = sf.read(str(path), always_2d=True)          # channel 0 the top mic, channel 1 the bottom mic
    up, _ = steps.chirp(True)
    m = np.abs(signal.correlate(x[fs:fs + 2 * 4800, 1], up, mode="valid"))
    t0 = fs + int(np.argmax(m[:4800])) - 96               # 2 ms before the bottom speaker's ping reaches the bottom mic
    bottom, top = s["speakers"]
    far_b = bottom["direct_level"][0] / bottom["direct_level"][1]   # bottom speaker: top mic over bottom mic
    far_t = top["direct_level"][1] / top["direct_level"][0]         # top speaker: bottom mic over top mic
    print(f"figure 1: set {SHOWN} background. Bottom speaker at the top mic {1 / far_b:.0f}x weaker ({20 * np.log10(far_b):.0f} dB); "
          f"top speaker at the bottom mic {1 / far_t:.0f}x weaker ({20 * np.log10(far_t):.0f} dB); loudest samples {s['loudest_sample']}")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.25, 1]})
    a = ax[0]
    t = np.arange(4800) / 48
    # the quieter mic on top during each ping: the top mic over the bottom speaker's, the bottom mic over the top speaker's
    for part, (under, over) in ((slice(0, 2401), (1, 0)), (slice(2400, 4800), (0, 1))):
        for z, ch in enumerate((under, over)):
            a.plot(t[part], x[t0:t0 + 4800, ch][part], color=BLUE if ch == 1 else ORANGE, lw=0.5, zorder=2 + z)
    a.set(title="a  One 100 ms cycle of the empty scene, as both mics recorded it", xlabel="ms",
          ylabel="mic level (1 = the loudest it can record)", xlim=(0, 100), ylim=(-0.42, 0.42))
    pk_b, pk_t = np.abs(x[t0:t0 + 960, 1]).max(), np.abs(x[t0 + 2400:t0 + 3360, 0]).max()
    a.annotate("bottom speaker's ping,\nat the bottom mic", xy=(9, pk_b * 0.95), xytext=(17, 0.33), color=BLUE, fontsize=9,
               arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    a.annotate(f"the same ping at the top mic, 16 cm away:\n{1 / far_b:.0f}x weaker ({20 * np.log10(far_b):.0f} dB)",
               xy=(8, -0.025), xytext=(17, -0.3), color=ORANGE, fontsize=9, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    a.annotate("top speaker's ping,\nat the top mic", xy=(58, pk_t * 0.95), xytext=(66, 0.33), color=ORANGE, fontsize=9,
               arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    a.annotate(f"the same ping at the bottom mic: {1 / far_t:.0f}x weaker\n({20 * np.log10(far_t):.0f} dB), the flat line through the burst",
               xy=(58, -0.004), xytext=(66, -0.3), color=BLUE, fontsize=9, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    a.text(99, 0.17, "each echo arrives while its ping is still\nplaying, hidden inside these bursts", ha="right", va="top",
           fontsize=8.5, color=MUTED, style="italic")
    a.plot([], [], color=BLUE, lw=2.5, label="bottom mic")
    a.plot([], [], color=ORANGE, lw=2.5, label="top mic")
    a.legend(loc="lower center", ncol=2)

    a = ax[1]
    f, tt, S = signal.spectrogram(x[t0:t0 + 3 * 4800, 1], fs, nperseg=128, noverlap=112, nfft=512)
    im = a.pcolormesh(tt * 1000, f / 1000, 10 * np.log10(S + 1e-15), cmap="Greys", vmin=-120, vmax=-50, shading="auto", rasterized=True)
    a.grid(False)
    a.set(title="b  The bottom mic over three cycles of the empty scene, by frequency", xlabel="ms", ylabel="frequency, kHz",
          ylim=(0, 24), xlim=(0, 300))
    for k in (18, 22.5):
        a.axhline(k, color=BLUE, lw=0.8, ls=(0, (3, 3)), alpha=0.7)
    a.text(298, 20.25, "ping band\n18 to 22.5 kHz", ha="right", va="center", fontsize=8.5, color=BLUE,
           bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    a.annotate("bottom speaker, sweeping up,\nat the mic beside it", xy=(105, 19), xytext=(70, 11.5), fontsize=8.5, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    a.annotate("top speaker, sweeping down,\nfrom the far end", xy=(157, 21.5), xytext=(170, 14.5), fontsize=8.5, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    a.text(150, 6.2, "room noise, below about 5 kHz: far from the pings", ha="center", fontsize=8.5, color=INK,
           bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    a.text(150, 23.7, "24 kHz: the highest a 48 kHz recording can hold", fontsize=8, color=INK2, va="top", ha="center",
           bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    cb = fig.colorbar(im, ax=a, pad=0.015, fraction=0.04)
    cb.set_label("level, dB (relative)", color=INK2)
    cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(colors=MUTED)
    fig.text(0.99, 0.005, f"the empty scene of set {SHOWN}, media volume {s['media_volume']}; channel 0 of the recording is the top mic, channel 1 the bottom",
             ha="right", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    save(fig, "1_what_the_mics_hear.png")


SCENES = (("background", "the empty scene, nothing in front"), ("fish", "the fish, 10 cm in front"),
          ("bobblehead", "the bobblehead, 10 cm in front"))
ZOOM = slice(0, 14 * 48)   # the bottom speaker's ping, 0 to 14 ms into each cycle


BAND = signal.butter(6, [17000, 23500], btype="bandpass", fs=48000, output="sos")   # the pings, without the room's low noise


def cycles(label, band=False):
    """Every whole 100 ms cycle of one of SHOWN's recordings, each starting 2 ms before the bottom speaker's ping
    reaches the bottom mic: (cycles, 4800, mics). With band, only 17 to 23.5 kHz is kept, before cutting."""
    path = next(FOLDER.glob(f"set_*_{SHOWN}_*_{label}.wav"))
    x, fs = sf.read(str(path), always_2d=True)
    up, _ = steps.chirp(True)
    m = np.abs(signal.correlate(x[fs:fs + 2 * 4800, 1], up, mode="valid"))
    t0 = (fs + int(np.argmax(m[:4800])) - 96) % 4800
    if band:
        x = signal.sosfiltfilt(BAND, x, axis=0)
    return np.stack([x[t0 + k * 4800: t0 + (k + 1) * 4800] for k in range((len(x) - t0) // 4800)])


def every_cycle():
    C = {label: cycles(label) for label, _ in SCENES}
    t = np.arange(4800) / 48
    # the outline is taken in the ping band only: the fish and bobblehead recordings had more low room noise, which is
    # not the object and would raise their outlines even before the ping starts
    outline = {label: np.abs(signal.hilbert(cycles(label, band=True)[:, ZOOM, 0], axis=1)) for label, _ in SCENES}
    mean = {label: o.mean(0) for label, o in outline.items()}
    peak = mean["background"].max()
    print(f"figure 2: set {SHOWN}; far-mic outline, ping to ping: {np.mean(outline['background'].std(0)) / peak:.1%} of its peak; "
          + "; ".join(f"{label} changes it by up to {np.abs(mean[label] - mean['background']).max() / peak:.0%}" for label, _ in SCENES[1:]))

    fig, ax = plt.subplots(3, 2, figsize=(15, 11.5), sharex="col", gridspec_kw={"width_ratios": [1.6, 1]})
    for row, (label, name) in enumerate(SCENES):
        c = C[label]
        a = ax[row, 0]
        for part, order in ((slice(0, 2401), (1, 0)), (slice(2400, 4800), (0, 1))):   # the quieter mic on top, as in figure 1
            for z, ch in enumerate(order):
                a.plot(t[part], c[:, part, ch].T, color=BLUE if ch == 1 else ORANGE, lw=0.3, alpha=0.06, zorder=2 + z, rasterized=True)
        a.add_patch(plt.Rectangle((0, -0.036), 14, 0.072, fill=False, ec=INK2, lw=0.9, ls=(0, (3, 2)), zorder=5))
        a.set(title=f"{'ace'[row]}  {name[0].upper() + name[1:]}: all {len(c)} cycles, overlaid",
              ylabel="mic level (1 = the loudest it can record)", xlim=(0, 100), ylim=(-0.42, 0.42))
        if row == 0:
            a.plot([], [], color=BLUE, lw=2.5, label="bottom mic")
            a.plot([], [], color=ORANGE, lw=2.5, label="top mic")
            a.legend(loc="lower center", ncol=2)
            a.annotate("enlarged at right", xy=(14, 0.03), xytext=(19, 0.16), fontsize=8.5, color=INK2,
                       arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
            a.text(99, 0.4, "at this scale the three scenes look the same", ha="right", va="top", fontsize=8.5, color=MUTED, style="italic")

        b = ax[row, 1]
        b.plot(t[ZOOM], c[:, ZOOM, 0].T, color=ORANGE, lw=0.3, alpha=0.06, rasterized=True)
        for sign in (1, -1):
            b.plot(t[ZOOM], sign * mean[label], color=INK, lw=1.4, label="its outline in the ping band, 17 to 23.5 kHz" if sign > 0 else None)
            if label != "background":
                b.plot(t[ZOOM], sign * mean["background"], color=INK2, lw=1.2, ls=(0, (4, 2)),
                       label="the empty scene's outline, the same way" if sign > 0 else None)
        b.set(title=f"{'bdf'[row]}  Top mic, during the bottom speaker's ping", xlim=(0, 14), ylim=(-0.04, 0.04))
        if label == "background":
            b.legend(loc="lower right", fontsize=8.5)
            b.text(0.3, 0.035, "the ping 16 cm across the phone, 13x weaker\nthan at the mic beside the speaker",
                   va="top", fontsize=8.5, color=INK2)
        else:
            d = mean[label] - mean["background"]
            k = int(np.argmax(np.abs(d)))
            b.annotate(f"the {label}'s echo arrives a few tenths of a ms after\nthis sound and overlaps it; where the two add and\n"
                       f"cancel, the outline moves: up to {np.abs(d).max() / peak:.0%} of its peak",
                       xy=(t[ZOOM][k], mean[label][k]), xytext=(0.3, 0.036), va="top", fontsize=8.5, color=INK2,
                       arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
            b.legend(loc="lower right", fontsize=8.5)
    for a in ax[2]:
        a.set_xlabel("ms into the cycle")
    fig.text(0.99, 0.004, f"the three recordings of set {SHOWN}, made one after another; each cycle starts 2 ms before the bottom speaker's ping reaches the bottom mic",
             ha="right", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    save(fig, "2_every_cycle.png")


def point():
    """What a single point would give: the chirp through its own matched filter, centred where echoes are put."""
    c, _ = steps.chirp(True)
    ac = np.correlate(c.real, c, "full")       # zero lag at CHIRP - 1
    idx = (len(c) - 1) + np.arange(W) - REF
    prof = np.zeros(W, complex)
    ok = (idx >= 0) & (idx < len(ac))
    prof[ok] = ac[idx[ok]]
    return np.tile(prof / np.abs(prof).max(), (1, 2, 1))


def echo_shape():
    E, y, g, r, ids = echoes()
    P = point()
    A, R, S = aligned(E, r) * 60 - 50, autocorr(E), stats(E)
    Ap, Rp, Sp = aligned(P, np.array([0])) * 60 - 50, autocorr(P), stats(P)
    mid = middles(E, r)

    def half(curve):
        """µs at which an autocorrelation first falls to a half"""
        k = int(np.argmax(curve < 0.5))
        return 1000 * (k - 1 + (curve[k - 1] - 0.5) / (curve[k - 1] - curve[k])) / 48

    print("figure 3: bottom spk > top mic, all 15 sets at 10 cm")
    print(f"  single point: width {1000 * Sp[0, 0]:.0f} us, share within 0.1 ms of peak {Sp[0, 2]:.2f}, autocorrelation half at {half(Rp[0, 0]):.0f} us, at 0.5 ms {Rp[0, 0, -1]:.2f}")
    for j, o in enumerate(OBJECTS):
        rec = [S[r == k, 0].mean() * 1000 for k in np.unique(r[y == j])]
        print(f"  {o:10s}: middle {np.mean((mid[y == j] - PRE) / 48):.2f} ms; width median {1000 * np.median(S[y == j, 0]):.0f} us "
              f"(recordings {min(rec):.0f} to {max(rec):.0f}); share near peak {np.median(S[y == j, 2]):.2f}; "
              f"autocorrelation half at {half(np.median(R[y == j, 0], 0)):.0f} us, at 0.5 ms {np.median(R[y == j, 0, -1]):.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    labels = {}
    a = ax[0]
    ms = (np.arange(AROUND.start, AROUND.stop) - REF) / 48
    for j, o in enumerate(OBJECTS):
        v = A[y == j, 0]
        a.fill_between(ms, np.percentile(v, 15, 0), np.percentile(v, 85, 0), color=COLOUR[o], alpha=0.18, lw=0)
        a.plot(ms, np.median(v, 0), color=COLOUR[o], lw=1.8)
    a.plot(ms, Ap[0, 0], color=INK2, lw=1.2, ls="--")
    a.text(-0.235, -21, "a single point", fontsize=8.5, color=INK2)
    a.text(-0.27, 0.6, "fish", fontsize=9.5, color=BLUE, fontweight="bold")
    a.text(0.28, -1.0, "bobblehead", fontsize=9.5, color=ORANGE, fontweight="bold")
    a.set(title="a  The echo, lined up on its own middle", xlabel="ms from the echo's middle",
          ylabel="dB relative to the echo's own average", ylim=(-25, 11), xlim=(ms[0], ms[-1]))

    a = ax[1]
    lag = np.arange(LAGS) / 48
    for j, o in enumerate(OBJECTS):
        v = R[y == j, 0]
        a.fill_between(lag, np.percentile(v, 15, 0), np.percentile(v, 85, 0), color=COLOUR[o], alpha=0.18, lw=0)
        a.plot(lag, np.median(v, 0), color=COLOUR[o], lw=1.8)
    a.plot(lag, Rp[0, 0], color=INK2, lw=1.2, ls="--")
    a.annotate("a single point", xy=(0.255, 0.45), xytext=(0.12, 0.22), fontsize=8.5, color=INK2,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a.text(0.33, 0.30, "fish", fontsize=9.5, color=BLUE, fontweight="bold")
    a.text(0.40, 0.52, "bobblehead", fontsize=9.5, color=ORANGE, fontweight="bold")
    a.set(title="b  The echo compared with itself, shifted", xlabel="shift, ms", ylabel="similarity (1 = identical)",
          ylim=(0, 1.03), xlim=(0, lag[-1]))

    a = ax[2]
    for j, o in enumerate(OBJECTS):
        a.scatter(1000 * S[y == j, 0], S[y == j, 2], s=7, color=COLOUR[o], alpha=0.25, lw=0)
    for j, o in enumerate(OBJECTS):
        ks = np.unique(r[y == j])
        a.scatter([1000 * S[r == k, 0].mean() for k in ks], [S[r == k, 2].mean() for k in ks], s=46, color=COLOUR[o],
                  edgecolor="white", lw=1, zorder=4, label=f"{o}: one dot per recording")
    a.scatter([1000 * Sp[0, 0]], [Sp[0, 2]], s=140, marker="*", color=INK2, edgecolor="white", lw=0.8, zorder=5, label="a single point")
    a.scatter([], [], s=7, color=MUTED, label="faint: each 0.4 s example")
    a.set(title="c  Two numbers that describe its shape", xlabel="echo width: rms about its middle, µs",
          ylabel="share of its energy within 0.1 ms of its peak")
    a.legend(loc="upper right", fontsize=8.5)
    fig.text(0.99, 0.005, "bottom speaker > top mic; the real echoes of all 15 sets at 10 cm; lines are medians, bands the 15th to 85th percentile "
             "of the 0.4 s examples; the dashed line and star are what a single point would give",
             ha="right", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    save(fig, "3_echo_shape.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    what_the_mics_hear()
    every_cycle()
    echo_shape()
