"""The final test: what the frozen method makes of block D, which nothing here was built on.

    python sealed.py          a dry run on block C, to show the script works. The models were trained on block C,
                              so its scores mean nothing. Block D is not read.
    python sealed.py open     block D. Run this once.

These tests and their pass marks were committed before block D was opened. The rules:

- Nothing is retrained or re-tuned afterwards. follow.pt has to be the file named by LOCKED.
- Every recording stays in, including any loud events.
- A missed pass mark is the result, and it gets written up. Anything added afterwards is labelled as exploratory.

Block D is 14 object recordings and one empty trial, so this can catch a failure but can't prove high accuracy:
even 14 of 14 only shows that true accuracy is above about 77%.

Main tests, each with a pass mark. The fifth runs in Java: after this, app/sim/sealed.ps1.

1. Naming: the strength-kept model, recordings right at 10 s
2. Distance: within 2 cm, the line fixed, not knowing the object
3. Detection: the empty trial rejected and objects found, at the level fixed in follow.py
4. First guess: the same answer at 2.8 s as at 10 s, "nothing there" counting as an answer
5. App match: the same answers from the app's own code, and gaps no larger than on blocks A to C

Comparisons, reported without pass marks: the strength-blurred model, strength and delay alone, nearest average, the
phone's old model, and the mean error of each object.

Robustness, also without pass marks: each recording against the previous set's empty scene, as test mode's reused
scene is; empty scenes against the previous set's and against themselves split in two, for false alarms; the echo
3 dB louder and quieter; the window moved about 2 cm either way; and the same mark placed twice.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from classify import centroid, cross, features
from distance import WINDOW, edge
from follow import BEFORE, MODE, cut, p_fish, read_echo, recordings, standardised
from model import EchoCNN
from scene import PRE, blocks
from sets import FOLDER, OBJECTS, read
from shape import logistic

HERE = Path(__file__).parent
SEALED = HERE.parent / "recordings" / "sealed"
LOCKED = "32bfe50c2855a3f4e0525847d3671beb73cdb21c476742e15c8436df88c37288"    # sha256 of follow.pt, from commit 613ce00
FIRST = 28                # pings at the first guess: 2.8 s
NAMING, DISTANCE, FOUND, SAME = 12, 13, 13, 13       # the pass marks, of 14 recordings
GAIN, MOVE = 3, 5         # dB and samples, for the robustness checks


def network(state):
    model = EchoCNN(len(OBJECTS), n_paths=2)
    model.load_state_dict(state)
    return model.eval()


def reading(Q, B, model, gain=0, shift=0):
    """What test mode makes of pings Q against an empty scene B."""
    e = blocks(Q, B, names)[0][:, pairs] * 10 ** (gain / 20)
    p = (np.abs(e.mean(0)[:, WINDOW]) ** 2).sum(0)
    ms, level = edge(p), 10 * np.log10(p.max())
    X = features(cut(e, np.full(len(e), PRE + int(round(ms * 48)) - BEFORE + shift)), MODE)
    return dict(ms=ms, cm=saved["cm_per_ms"] * ms + saved["cm_at_zero"], level=level, found=level > saved["detect_db"], p=float(p_fish(model, X).mean()), X=X)


def answer(r):
    return "nothing" if not r["found"] else OBJECTS[0] if r["p"] >= 0.5 else OBJECTS[1]


def line(text, passed=None):
    out = f"{text}" + ("" if passed is None else f"   {'PASS' if passed else 'FAIL'}")
    print(out, flush=True)
    results.append(out)


if __name__ == "__main__":
    opened = "open" in sys.argv
    if hashlib.sha256((HERE / "follow.pt").read_bytes()).hexdigest() != LOCKED:
        sys.exit("follow.pt is not the frozen model")
    saved = torch.load(HERE / "follow.pt", weights_only=False)
    frozen, blurred = network(saved["model"]), network(saved["strength_blurred"])
    names, measured = read()
    pairs = cross(names)
    train = sorted((s for s in measured if s["block"]), key=lambda s: s["id"])
    if opened:
        folder = SEALED
        sets = sorted(read(SEALED)[1], key=lambda s: s["id"])
        if len(sets) != 7 or {s["block"] for s in sets} != {"D"}:
            sys.exit("recordings/sealed is not block D")
    else:
        folder = FOLDER
        sets = [s for s in train if s["block"] == "C"]
    results = []
    line(f"{'BLOCK D' if opened else 'DRY RUN ON BLOCK C: THESE SCORES MEAN NOTHING'}: {len(sets)} sets")

    rows = [(s, j, o, reading(s[o + " as recorded"], s["background"], frozen), reading(s[o + " as recorded"][:, :FIRST], s["background"], frozen))
            for s in sets for j, o in enumerate(OBJECTS)]
    empties = [reading(s["empty as recorded"], s["background"], frozen) for s in sets if "empty as recorded" in s]
    truth = np.array([j for _, j, _, _, _ in rows])
    placed = np.array([s["distance"][o] for s, _, o, _, _ in rows], float)
    final = [r for _, _, _, r, _ in rows]
    n = len(rows)

    print("\nEach recording:")
    for (s, j, o, r, f) in rows:
        print(f"  {s['id'][-6:]} {o:10s} at {s['distance'][o]:.0f} cm: P(fish) {r['p']:.3f}, {r['cm']:.1f} cm, {r['level']:.1f} dB; at 2.8 s {answer(f)}, {f['cm']:.1f} cm")

    line("\nMain tests")
    right = np.array([(r["p"] < 0.5) == j for r, j in zip(final, truth)])
    line(f"1. Naming: {right.sum()} of {n} right, pass mark {NAMING}", right.sum() >= NAMING)
    err = np.array([r["cm"] for r in final]) - placed
    near = int(np.sum(np.abs(err) <= 2))
    line(f"2. Distance: {near} of {n} within 2 cm, pass mark {DISTANCE}; rms {np.sqrt(np.mean(err ** 2)):.2f} cm, within 1 cm {int(np.sum(np.abs(err) <= 1))}", near >= DISTANCE)
    found = sum(r["found"] for r in final)
    quiet = sum(not r["found"] for r in empties)
    line(f"3. Detection: {found} of {n} objects found, pass mark {FOUND}; {quiet} of {len(empties)} empty trials rejected, levels {[round(float(r['level']), 1) for r in empties]}",
         found >= FOUND and quiet == len(empties))
    same = sum(answer(r) == answer(f) for _, _, _, r, f in rows)
    line(f"4. First guess: {same} of {n} the same at 2.8 s as at 10 s, pass mark {SAME}", same >= SAME)
    line("5. App match: run app/sim/sealed.ps1; pass is the same answers, and gaps of at most 0.05 in P(fish), 0.51 cm and 0.44 dB")

    line("\nComparisons, no pass marks")
    line(f"  Strength-blurred model: {sum((float(p_fish(blurred, r['X']).mean()) < 0.5) == j for r, j in zip(final, truth))} of {n}")
    E, y, g, r, _ = recordings(names, train)
    keep = y < 2
    delay, level = read_echo(E, r)
    heard = features(cut(E, PRE + np.round(delay * 48).astype(int) - BEFORE), MODE)
    two = np.c_[level, delay]
    line(f"  Strength and delay alone: {sum((standardised(logistic)(two[keep], y[keep], np.array([[x['level'], x['ms']]])).mean() < 0.5) == j for x, j in zip(final, truth))} of {n}")
    line(f"  Nearest average: {sum((centroid(heard[keep], y[keep], x['X']).mean() < 0.5) == j for x, j in zip(final, truth))} of {n}")
    filed = [json.loads(next(Path(folder).glob(f"set_{s['id']}_*_{o}.json")).read_text()) for s, _, o, _, _ in rows]
    line(f"  The phone's old model, as it guessed at the time: {sum(f.get('guess') == OBJECTS[j] for f, j in zip(filed, truth))} of {n}")
    line(f"  Mean distance error: fish {err[truth == 0].mean():+.2f} cm, bobblehead {err[truth == 1].mean():+.2f} cm")

    line("\nRobustness, no pass marks")
    old = [(s, j, o, reading(s[o + " as recorded"], before["background"], frozen)) for before, s in zip(sets, sets[1:]) for j, o in enumerate(OBJECTS)]
    gap = np.array([x["cm"] - s["distance"][o] for s, _, o, x in old])
    line(f"  Against the previous set's empty scene: {sum((x['p'] < 0.5) == j for _, j, _, x in old)} of {len(old)} named, {int(np.sum(np.abs(gap) <= 2))} within 2 cm, {sum(x['found'] for _, _, _, x in old)} found")
    trials = [reading(s["empty as recorded"], before["background"], frozen) for before, s in zip(sets, sets[1:]) if "empty as recorded" in s]
    scenes = [reading(s["background_pings"], before["background"], frozen) for before, s in zip(sets, sets[1:])]
    line(f"  Taken for an object on the previous set's empty scene: {sum(x['found'] for x in trials)} of {len(trials)} empty trials, {sum(x['found'] for x in scenes)} of {len(scenes)} empty scenes")
    halves = [reading(s["background_pings"][:, s["background_pings"].shape[1] // 2:], s["background_pings"][:, :s["background_pings"].shape[1] // 2].mean(1), frozen) for s in sets]
    line(f"  Empty scenes split in two, taken for an object: {sum(x['found'] for x in halves)} of {len(halves)}")

    def named(**stress):
        return sum((reading(s[o + " as recorded"], s["background"], frozen, **stress)["p"] < 0.5) == j for s, j, o, _, _ in rows)

    line(f"  Echo {GAIN} dB louder {named(gain=GAIN)} of {n}, quieter {named(gain=-GAIN)}; window moved {MOVE} samples later {named(shift=MOVE)}, earlier {named(shift=-MOVE)}")
    twice = [(a[2], a[3]["cm"], b[3]["cm"]) for a in rows[:2] for b in rows[-2:] if a[1] == b[1]]
    line("  Same mark placed twice: " + ", ".join(f"{o} {a:.1f} then {b:.1f} cm" for o, a, b in twice))

    ref = {}
    for (s, _, o, r, _) in rows:
        ref.update({f"{s['id']}_{o}_p": round(r["p"], 4), f"{s['id']}_{o}_cm": round(r["cm"], 3), f"{s['id']}_{o}_db": round(r["level"], 2)})
    for s in sets:
        if "empty as recorded" in s:
            r = reading(s["empty as recorded"], s["background"], frozen)
            ref.update({f"{s['id']}_empty_p": round(r["p"], 4), f"{s['id']}_empty_cm": round(r["cm"], 3), f"{s['id']}_empty_db": round(r["level"], 2)})
    (HERE / "parity_sealed.json").write_text(json.dumps(ref, indent=1))
    if opened:
        (HERE / "sealed_result.txt").write_text("\n".join(results) + "\n")
        print("\nwrote sealed_result.txt and parity_sealed.json")
