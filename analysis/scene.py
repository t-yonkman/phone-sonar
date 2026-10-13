"""Read a recording made by the app (a WAV with one channel per mic, and the .json beside it that says
how it was made) into what each ping sounded like, for every speaker-to-mic pair.

Same processing as the app: correlate with each speaker's chirp, cut a window around each ping's direct
arrival at the mic beside that speaker, and divide by that direct arrival so gain and phase drift cancel.

Within a recording, playing and recording are locked to the same clock (the direct arrival's phase holds
to a fraction of a degree from ping to ping). Between recordings they are not: each one starts a different
fraction of a sample early or late, which match() and blocks() take out.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

PRE, POST = 48, 432               # window kept around the direct arrival, samples
W = PRE + POST
GUARD = slice(PRE + 5, PRE + 20)  # 0.1 to 0.4 ms after the direct sound: too soon for an echo from 10 cm away
_F = np.fft.fftfreq(W)


def chirp(s, up):
    i = np.arange(s["chirp"])
    t, T = i / s["fs"], s["chirp"] / s["fs"]
    fa, fb = (s["f0"], s["f1"]) if up else (s["f1"], s["f0"])
    edge = np.minimum(i, s["chirp"] - 1 - i)
    env = np.where(edge < s["fade"], 0.5 - 0.5 * np.cos(np.pi * edge / s["fade"]), 1.0)
    return env * np.exp(2j * np.pi * (fa * t + (fb - fa) * t * t / (2 * T)))


def load(path, nudge=(0, 0)):
    """-> (settings, names, time of each ping in s, profiles, direct level per speaker and ping)

    profiles is complex, shaped (pairs, pings, W), one pair per speaker and mic in the order of names
    ("bottom spk > top mic", ...), each relative to that speaker's direct sound at the mic beside it.
    Pings whose direct sound is missing are NaN.

    nudge moves the sample taken as each speaker's direct arrival (bottom, top) by whole samples. The
    direct sound has two nearly equal peaks a sample apart, so which one counts as "the" arrival can
    differ between two recordings; see match() for putting two recordings on the same footing.
    """
    path = Path(path)
    s = json.loads(path.with_suffix(".json").read_text())
    x, fs = sf.read(str(path), dtype="float64", always_2d=True)
    assert fs == s["fs"]
    cycle, mics = s["cycle"], s["channel_mics"]
    names, Q, direct, start = [], [], [], None
    for which, sp in enumerate(s["speakers"]):
        c = chirp(s, sp["sweep"] == "up")
        m = [signal.correlate(x[:, ch], c, mode="valid") for ch in range(x.shape[1])]
        n = len(m[0]) // cycle
        ref = sp["reference_channel"]
        level = np.median(np.abs(m[ref][: n * cycle]).reshape(n, cycle), axis=0)
        if start is None:
            # where in the cycle the bottom speaker's direct arrival falls: the earliest strong lobe, as the app picks it
            k = int(np.argmax(level))
            early = [j for j in range(k - 16, k) if level[j] >= 0.7 * level[k] and level[j] >= level[j - 1] and level[j] >= level[j + 1]]
            start = early[0] if early else k
            phase = start
        else:
            # the app found this speaker's direct arrival this far after the bottom speaker's
            phase = start + sp["after_bottom_samples"]
        at = phase + nudge[which] + np.arange(1, n - 2) * cycle
        at = at[at + POST < len(m[ref])]
        d = m[ref][at]
        lost = np.abs(d) < 0.5 * np.median(np.abs(d))
        for ch in range(x.shape[1]):
            q = np.array([m[ch][a - PRE: a + POST] for a in at]) / d[:, None]
            q[lost] = np.nan
            Q.append(q)
            names.append(f"{sp['name']} spk > {mics[ch]} mic")
        direct.append(2 * np.abs(d) / np.sum(np.abs(c) ** 2))
        tt = at / fs
    n = min(len(q) for q in Q)
    return s, names, tt[:n], np.array([q[:n] for q in Q]), np.array([d[:n] for d in direct])


def slid(q, d):
    """Profiles q (..., W) read off d samples later; d may be a fraction (a phase ramp in frequency)."""
    return np.fft.ifft(np.fft.fft(q, axis=-1) * np.exp(2j * np.pi * _F * d), axis=-1)


def speakers(names):
    """-> for each speaker: (which pairs are its pings, which of those has the mic beside it)"""
    out = {}
    for i, n in enumerate(names):
        spk, mic = n.split(" spk > ")[0], n.split(" spk > ")[1].replace(" mic", "")
        out.setdefault(spk, [[], None])[0].append(i)
        if spk == mic:
            out[spk][1] = i
    return list(out.values())


def match(Q, B, names):
    """Put a recording's pings Q (pairs, pings, W) on the same timing as a reference scene B (pairs, W).

    Each time the app starts, each speaker's sound reaches the recording a different fraction of a sample
    early or late, and stays that way until it stops. Two recordings therefore differ near the direct
    sound by far more than any echo, until one is slid onto the other. The slide is found per speaker,
    from the stretch just after its direct sound where no echo can have arrived yet, and applied to both
    of that speaker's pairs.  -> (the slid pings, the slides found in samples)
    """
    out, found = Q.copy(), []
    grid = np.arange(-2.5, 2.5001, 0.01)   # up to a sample each for the fraction and for which lobe was taken as the arrival
    for pairs, own in speakers(names):
        m = np.nanmean(Q[own], axis=0)
        cost = []
        for d in grid:
            s = slid(m, d)
            cost.append(np.sum(np.abs(s[GUARD] / s[PRE] - B[own][GUARD]) ** 2))
        d = grid[int(np.argmin(cost))]
        ref = slid(Q[own], d)[:, PRE:PRE + 1]      # its direct sound, which every pair is measured against
        for p in pairs:
            out[p] = slid(Q[p], d) / ref
        found.append(float(d))
    return out, found


def blocks(Q, B, names, size=4):
    """A recording's pings Q (pairs, pings, W) as echoes against a background B (pairs, W): the average of
    each `size` consecutive pings, slid onto the background's timing, minus the background.

    This is exactly what the app does on the phone, in the same order, so a model trained on these is
    looking at the same thing there: find each speaker's slide from the whole recording's average ping
    (as match does), average the pings in fours (over which the phone's four-ping pattern cancels),
    slide each average, measure it against its own slid direct sound, and take the background away.
    -> echoes shaped (blocks, pairs, W), and the slides found
    """
    n = Q.shape[1] // size
    A = Q[:, : n * size].reshape(Q.shape[0], n, size, -1).mean(2)
    out, found = np.empty_like(A), []
    grid = np.arange(-2.5, 2.5001, 0.01)   # up to a sample each for the fraction and for which lobe was taken as the arrival
    for pairs, own in speakers(names):
        m = np.nanmean(Q[own], axis=0)
        cost = []
        for d in grid:
            s = slid(m, d)
            cost.append(np.sum(np.abs(s[GUARD] / s[PRE] - B[own][GUARD]) ** 2))
        d = grid[int(np.argmin(cost))]
        ref = slid(A[own], d)[:, PRE:PRE + 1]
        for p in pairs:
            out[p] = slid(A[p], d) / ref - B[p]
        found.append(float(d))
    return out.transpose(1, 0, 2), found
