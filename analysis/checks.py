"""Four checks on the method of follow.py, on blocks A to C.    python checks.py

Needs the recordings in recordings/sets. Block D is kept in recordings/sealed and is not read here.

1. How soon: what test mode would say from the first seconds of pings, with each block held out.
2. An older empty scene: test mode reuses one for several tests, but each recording here was made against its own.
   Each recording is judged against the empty scene of 0, 1, 2 and more sets before it, and each empty scene against
   an older one, which is an empty mat with nothing done to it but time.
3. Can the blocks be told apart? Each set's empty scene against the others, and then naming the block from what
   the model is given, each set held out in turn. Three blocks, so guessing gets 14 of 42.
4. How long is an echo? Its width at half power against the direct sound's, and how much of it falls in the window.
"""
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from classify import centroid, cross, features
from distance import WINDOW, edge
from follow import AFTER, BEFORE, DETECT, MODE, cut, fit, p_fish, read_echo, recordings
from model import EchoCNN
from scene import PRE, W, blocks, match
from sets import OBJECTS, read

MS = 48
SECONDS = (2.0, 2.8, 4.0, 6.0, 8.0, 9.6)
GATE = slice(PRE + 10, PRE + 62)     # 0.2 to 1.3 ms
LATER = slice(PRE + 62, W)           # 1.3 to 9 ms


def reading(e):
    """-> the edge in ms and the loudest point in dB of the average of echoes e"""
    p = (np.abs(e.mean(0)[:, WINDOW]) ** 2).sum(0)
    return edge(p), 10 * np.log10(p.max())


def window(e, ms):
    return features(cut(e, np.full(len(e), PRE + int(round(ms * MS)) - BEFORE)), MODE)


def how_soon():
    print("\n1. How soon. Seconds of pings: recordings named, closest calls, objects found, distance rms, within 2 cm, loudest empty trial")
    truth = np.array([y[r == k][0] for k in ks])
    placed = np.array([d[r == k][0] for k in ks])
    for seconds in SECONDS:
        n = int(seconds / 0.4)
        cm, level, p = [], [], []
        for k in ks:
            t = np.flatnonzero(r == k)[:n]
            b = block[t[0]]
            ms, db = reading(E[t])
            cm.append(np.polyval(lines[b], ms)); level.append(db); p.append(p_fish(models[b], window(E[t], ms)).mean())
        cm, level, p = map(np.array, (cm, level, p))
        err = cm - placed
        blank = max(reading(Em[(rm == k)][:n])[1] for k in np.unique(rm))
        print(f"  {seconds:3.1f} s: {np.sum((p < 0.5) == truth)} of {len(ks)}, fish {p[truth == 0].min():.2f}, bobblehead {p[truth == 1].max():.2f}, "
              f"{np.sum(level > DETECT)} found, {np.sqrt(np.mean(err ** 2)):.2f} cm, {np.mean(np.abs(err) <= 2):.0%}, {blank:.1f} dB")


def when(i):
    return datetime.strptime(i, "%Y%m%d_%H%M%S")


def judged(Q, B, b):
    e = blocks(Q, B, names)[0][:, pairs]
    ms, db = reading(e)
    return np.polyval(lines[b], ms), db, p_fish(models[b], window(e, ms)).mean()


def older_scene():
    print("\n2. An older empty scene. Sets back, age in minutes: objects found, named, distance rms, within 2 cm; empty trials and then empty scenes taken for an object")
    for back in range(7):
        res, trials, scenes, ages = [], [], [], []
        for b in letters:
            ss = [s for s in sets if s["block"] == b]
            for i in range(back, len(ss)):
                s, old = ss[i], ss[i - back]
                ages.append((when(s["id"]) - when(old["id"])).total_seconds() / 60 + 0.7)   # its own is about 0.7 min old
                res += [(j, s["distance"][o]) + judged(s[o + " as recorded"], old["background"], b) for j, o in enumerate(OBJECTS)]
                if "empty as recorded" in s:
                    trials.append(judged(s["empty as recorded"], old["background"], b)[1])
                if back:
                    scenes.append(judged(s["background_pings"], old["background"], b)[1])
        obj, placed, cm, level, p = np.array(res, float).T
        err, trials, scenes = cm - placed, np.array(trials), np.array(scenes)
        print(f"  {back}, {np.median(ages):4.1f} min: {np.sum(level > DETECT)} found, {np.sum((p < 0.5) == obj)} of {len(obj)} named, {np.sqrt(np.mean(err ** 2)):.2f} cm, "
              f"{np.mean(np.abs(err) <= 2):.0%}; {np.sum(trials > DETECT)} of {len(trials)}, {np.sum(scenes > DETECT)} of {len(scenes)}")


def scene_matrix():
    ids = [s["id"] for s in sets]
    gate, later = [np.full((len(ids), len(ids)), np.nan) for _ in range(2)]
    for a, s in enumerate(sets):
        for b, u in enumerate(sets):
            if a != b:
                q = match(s["background"][:, None], u["background"], names)[0][pairs, 0]
                gate[a, b] = 10 * np.log10(np.mean(np.abs(q[:, GATE] - u["background"][pairs][:, GATE]) ** 2))
                later[a, b] = 10 * np.log10(np.mean(np.abs(q[:, LATER] - u["background"][pairs][:, LATER]) ** 2))
    which = np.array([s["block"] for s in sets])
    same = which[:, None] == which[None]
    for n, X in (("0.2 to 1.3 ms", gate), ("1.3 to 9 ms", later)):
        print(f"  {n}: same block {np.median(X[same & ~np.isnan(X)]):.1f} dB, other block {np.median(X[~same]):.1f} dB; nearest is in the same block for {np.sum(which[np.nanargmin(X, 1)] == which)} of {len(ids)} sets")


def block_net(X, y):
    torch.manual_seed(0)
    model = EchoCNN(len(letters), n_paths=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    A, labels = torch.from_numpy(X), torch.from_numpy(y)
    model.train()
    for _ in range(30):
        for idx in torch.randperm(len(A)).split(32):
            loss = F.cross_entropy(model(A[idx]), labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def tell_apart():
    print("\n3. Can the blocks be told apart?")
    scene_matrix()
    target = np.searchsorted(letters, block)
    for n, X in (("as the model is given it", features(cut(E, start), MODE)), ("loudness removed", features(cut(E, start), "each"))):
        near, net = np.zeros((len(E), len(letters))), np.zeros((len(E), len(letters)))
        for h in np.unique(g):
            t = g == h
            mean = np.array([X[~t & (target == b)].reshape((~t & (target == b)).sum(), -1).mean(0) for b in range(len(letters))])
            near[t, ((X[t].reshape(t.sum(), 1, -1) - mean[None]) ** 2).sum(-1).argmin(1)] = 1
            model = block_net(X[~t], target[~t])
            with torch.no_grad():
                net[t] = torch.softmax(model(torch.from_numpy(X[t])), 1).numpy()
        truth = np.array([target[r == k][0] for k in ks])
        print(f"  Block named from the window, {n}: nearest average {np.sum(np.array([near[r == k].mean(0).argmax() for k in ks]) == truth)} of 42, "
              f"CNN {np.sum(np.array([net[r == k].mean(0).argmax() for k in ks]) == truth)}")


def half_width(p, k):
    i, j = k, k
    while i > 0 and p[i - 1] >= p[k] / 2:
        i -= 1
    while j < len(p) - 1 and p[j + 1] >= p[k] / 2:
        j += 1
    return ((j + (p[j] - p[k] / 2) / (p[j] - p[j + 1])) - (i - 1 + (p[k] / 2 - p[i - 1]) / (p[i] - p[i - 1]))) / MS


def echo_length():
    print("\n4. How long is an echo?")
    own = names.index("bottom spk > bottom mic")
    print(f"  Direct sound, half-power width {np.median([half_width(np.abs(s['background'][own]) ** 2, PRE) for s in sets]):.2f} ms")
    rows = []
    for k in ks:
        t = r == k
        p = np.abs(E[t].mean(0)) ** 2
        x = np.clip(p - np.median(p[:, PRE + 12:PRE + 150], axis=1, keepdims=True), 0, None)
        top = PRE + 12 + int(np.argmax(p[0, PRE + 12:PRE + 76]))
        rows.append((y[t][0], d[t][0], half_width(p[0], top), x[:, start[t][0]:start[t][0] + BEFORE + AFTER].sum() / x[:, PRE + 5:PRE + 150].sum()))
    obj, placed, width, share = np.array(rows).T
    for n, q in (("Fish", obj == 0), ("Bobblehead, 16 to 20 cm", (obj == 1) & (placed >= 16)), ("Bobblehead, 10 to 14 cm", (obj == 1) & (placed <= 14))):
        print(f"  {n}: half-power width {np.median(width[q]):.2f} ms; echo energy in the window {np.median(share[q]):.0%}, lowest {share[q].min():.0%}")


if __name__ == "__main__":
    names, sets = read()
    sets = sorted((s for s in sets if s["block"]), key=lambda s: s["id"])
    pairs = cross(names)
    Em, ym, gm, rm, dm = recordings(names, sets)
    blockm = np.array([sets[i]["block"] for i in gm])
    objects = ym < 2
    E, y, g, r, d, block = Em[objects], ym[objects], gm[objects], rm[objects], dm[objects], blockm[objects]
    Em, rm = Em[~objects], rm[~objects]        # the empty trials
    delay = read_echo(E, r)[0]
    start = PRE + np.round(delay * MS).astype(int) - BEFORE
    ks, letters = np.unique(r), np.unique(block)
    models, lines = {}, {}
    for b in letters:
        t = block != b
        models[b], lines[b] = fit(E[t], y[t], start[t]), np.polyfit(delay[t], d[t], 1)
    how_soon()
    older_scene()
    tell_apart()
    echo_length()
