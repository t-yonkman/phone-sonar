"""Control: is anything left once loudness is truly gone?    python control.py    (needs the raw recordings)

Removing loudness by rescaling is not enough, because a louder echo also stands further above the noise,
and that shows as a different shape. Here, within each set and on each pair, the louder object's echo is
turned down to the quieter one's level and real background noise is added back, so both objects have the
same echo strength over the same noise. What still tells them apart is how the echo is spread in time.
"""
import numpy as np

from classify import GATE, MODE, centroid, cnn, features, given
from sets import LO, OBJECTS, read

WIN = slice(LO, LO + GATE)   # the stretch of delay the phone's network is given
rng = np.random.default_rng(0)
names, sets = read()
E, y, g, r = [], [], [], []
for i, s in enumerate(sets):
    qb = s["background_pings"]
    nb = qb.shape[1] // 4
    noise = qb[:, : 4 * nb].reshape(qb.shape[0], nb, 4, -1).mean(2) - s["background"][:, None, :]      # (pairs, blocks, W): nothing but noise
    echo = {o: s[o].mean(1) - s["background"] for o in OBJECTS}
    power = {o: np.sum(np.abs(echo[o][:, WIN]) ** 2, axis=1) for o in OBJECTS}                         # per pair
    target = np.minimum(power[OBJECTS[0]], power[OBJECTS[1]])
    for j, o in enumerate(OBJECTS):
        q = s[o]
        n = q.shape[1] // 4
        blocks = q[:, : 4 * n].reshape(q.shape[0], n, 4, -1).mean(2) - s["background"][:, None, :]
        a = np.sqrt(target / power[o])[:, None, None]                                                   # turn down, never up
        extra = noise[:, rng.integers(0, nb, n)]
        blocks = a * blocks + np.sqrt(1 - a ** 2) * extra
        E.append(blocks.transpose(1, 0, 2)[:, :, WIN])
        y += [j] * n; g += [i] * n; r += [2 * i + j] * n
E, y, g, r = np.concatenate(E), np.array(y), np.array(g), np.array(r)
check = [10 * np.log10(np.mean(np.abs(E[y == j]) ** 2, axis=(0, 2))) for j in range(2)]
print("average level per pair after equalising, dB:  fish " + " ".join(f"{v:6.1f}" for v in check[0]) + "   bobblehead " + " ".join(f"{v:6.1f}" for v in check[1]))
print("each set held out in turn:")
for label, pairs in (("the pairs the phone is given", given(names)), ("all four pairs", list(range(len(names))))):
    X = features(E[:, pairs], MODE)
    line = f"  {label:28s}"
    for name, model in (("CNN", cnn), ("nearest average", centroid)):
        pred = np.zeros_like(y)
        for held in np.unique(g):
            t = g == held
            pred[t] = model(X[~t], y[~t], X[t], 30) < 0.5   # the models answer with P(fish); class 0 is the fish
        votes = [(y[r == k][0], np.bincount(pred[r == k], minlength=2).argmax()) for k in np.unique(r)]
        wrong = [f"{sets[k // 2]['id'][-4:]}{OBJECTS[k % 2][0]}" for k, (a_, b_) in zip(np.unique(r), votes) if a_ != b_]
        line += f" | {name}: {np.mean(pred == y):4.0%} of examples, {sum(a_ == b_ for a_, b_ in votes):2d} of {len(votes)} recordings" + (f" (wrong: {' '.join(wrong)})" if wrong else "")
    print(line, flush=True)
