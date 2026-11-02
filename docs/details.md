# Phone sonar: the details

*The full account behind the [README](../README.md): every test and its numbers, both field tests, what
the phone does to the data, the signal chain step by step, and the reasoning behind the limits. A walk
through the first model's figures is in [writeup.md](writeup.md).*

The method has two stages. The first model (26 sets, a fixed window, each object about 10 cm out by eye)
classified an object known to be there. It missed in the field when echoes arrived unusually early or
late, so the current method first finds the object, reads its distance, and only then names it, on
recordings at measured distances. The current method comes first here; the first model follows, as the
record of how it got there.

## The measured blocks

A Galaxy Note10+ stands on its long edge on a kickstand, the back of its case facing a paper mat marked
from 10 to 30 cm. Distance runs from the back of the case to the centre of the object's base. Each end of
the phone has a speaker and a mic, 16 cm apart.

| Level | Count | What it is |
|---|---|---|
| Block | 4 | One spot: 7 sets at 10, 12, 14, 16, 18 and 20 cm in a shuffled order, then the first again as a drift check |
| Set | 28 | An empty scene, then both objects at one mark, in a random order; one set in each block adds an empty trial |
| Recording | 56 with an object, 4 empty trials | 10 s after 3 s to get clear, with a photo after each |

Blocks A, B and C were used for everything below. Block D was recorded the same way and kept sealed. The
app shows what to place, so no label is typed, and each recording starts itself after a countdown, so the
phone isn't touched between the empty scene and the objects. None of the 66 A to C recordings was flagged
or lost the pings. One, a fish in block A's last set, has a loud event in it; it stays in, and its distance
read 16.1 cm at a mark of 16.

**What it showed about the first model.** The 26-set model scored 33 of 42 on these blocks on the phone,
and only 1 of 6 fish at 18 to 20 cm. It called all 3 empty trials the bobblehead. The earlier sets, meant
to be 10 cm out, typically read 15.5 cm for the fish and 17.3 cm for the bobblehead, from about 8 to 23 cm:
they were never measured from a fixed point.

## Reading distance

`analysis/distance.py`. Each block is held out in turn, with both objects pooled, so nothing knows which
object it is measuring. 42 recordings:

| Read from the echo | rms error | Within 1 cm | Within 2 cm |
|---|---|---|---|
| Edge: where it first rises to half of its peak power | 1.46 cm | 74% | 90% |
| Centre: its power-weighted middle | 1.59 cm | 64% | 88% |
| Peak | 1.60 cm | 60% | 88% |
| Ridge regression on the whole echo | 1.43 cm | 55% | 86% |

The edge was chosen for the best share within 1 and 2 cm. Resampling sets gives 0.84 to 2.11 cm rms and 62
to 86% within 1 cm. The search starts at 0.25 ms: starting at 0.46 ms, as `shape.py` does, clipped seven
echoes and flattered the 10 cm results.

| Recordings | rms error | Within 1 cm | Within 2 cm |
|---|---|---|---|
| Fish, 10 to 20 cm | 0.79 cm | 76% | 100% |
| Both objects, 16 to 20 cm | 0.76 cm | 77% | 100% |
| Fish, 10 to 14 cm | 0.83 cm | 80% | 100% |
| Bobblehead, 10 to 14 cm | 2.65 cm | 60% | 60% |

The mean error is +0.04 cm for the fish and −0.02 cm for the bobblehead. All four errors over 2 cm are the
bobblehead at 10 to 14 cm: +6.5, −3.1, +3.1 and −2.8 cm, the rest of the recordings coming to about 0.7 cm rms.

**What limits it.** Not noise: one 0.4 s reading varies by about 0.3 cm. The same object placed twice at the
same mark reads about 1 cm apart per reading, which rules out the first target of ±1 cm on 9 recordings in
10. The target became ±2 cm, which is what the window needs. That was decided after seeing these results,
so block D has to confirm it. Error doesn't grow through a block (+0.08 cm a set, −0.10 to +0.27).

**Detection.** The loudest point of the echo stayed under −55.6 dB on all 3 empty trials and passed −52.3 dB
on every object but one, block C's bobblehead at 14 cm at −59.1 dB. The threshold is −54 dB.

| Loudest point | 10 to 14 cm | 16 cm | 18 to 20 cm |
|---|---|---|---|
| Fish | −37 dB | −40 dB | −42 dB |
| Bobblehead | −51 dB | −44 dB | −44 dB |

The fish weakens with distance, as echoes should. The bobblehead is weak and erratic up close and steady from
18 cm. That is not movement: its echo is as steady from ping to ping as the fish's in 20 of 21 recordings. The
cause is unknown. One guess is the angle, which is about 40° either side at 10 cm and 22° at 20 cm.

## Naming the object

`analysis/follow.py`. A 1 ms window starts 10 samples before the echo's edge, in dB against the direct sound.
In training it is moved at random by up to 5 samples, about 2 cm, so a misread does little harm. Recordings
right of 42, each block held out:

| Input | Right |
|---|---|
| Follows the echo, strength kept (the frozen model; all three seeds) | 42 |
| Nearest class mean, same input | 41 |
| Fixed window, loudness removed, retrained on these blocks | 39 |
| Strength and delay alone, by logistic regression | 38 |
| Follows the echo, strength blurred by up to 10 dB in training | 34 |
| Shape statistics, by logistic regression | 33 |
| Follows the echo, loudness removed | 31 |

![Recordings right for each input, and what the model said of each recording](follow.png)

The frozen model gets 98% of 0.4 s examples right, and its closest calls are P(fish) 0.75 and 0.17. Trained
on all three blocks and tested on the 26 earlier sets, from other days and spots, it names 49 of 52. Adding
those sets to the training did no good, so they stay out of it. With detection in front, 44 of the 45 trials
come out right; the miss is the bobblehead reported as nothing there.

**It relies on strength.** Making every echo louder or quieter moves all its inputs together, and the
model doesn't survive that:

| Change to the echo | −6 dB | −3 dB | +3 dB | +6 dB |
|---|---|---|---|---|
| Frozen model | 22 | 33 | 33 | 23 |
| Strength blurred in training | 36 | 35 | 36 | 40 |

Moving the window instead hardly matters to the frozen model: 42 of 42 at about 2 cm either way and 39 to 41
at about 5 cm. The blurred model is steadier against level but weaker overall (34 of 42, 46 of 52 on the
earlier sets) and worse against a moved window (29 to 37). The frozen model goes on the phone because it holds
across days here, but it is mostly naming each object by how strongly it reflects, and a different stand angle
or a third object could break it. The blurred model is kept beside it, and block D tests both.

**Why shape is weak.** The speakers deliver about 19.5 to 22 kHz, not the 18 to 22.5 kHz sent. That blurs each
echo over about 5 cm of distance, and both objects are smaller than that. At half power the fish's echo is 0.29
ms wide and the bobblehead's 0.31 ms, against 0.35 ms for the direct sound: the object adds nothing measurable
to the width of the ping. The remaining shape information probably lives in how parts of the object interfere,
which changes with small turns. More bandwidth is the thing that would change this.

## Checks of the method

`analysis/checks.py`, on blocks A to C.

**How soon.** With each block held out, the first seconds do nearly as well as all ten, and 2.8 s is where test
mode gives its first guess:

| Seconds of pings | Named | Distance rms | Within 2 cm |
|---|---|---|---|
| 2.0 | 42 of 42 | 1.53 cm | 90% |
| 2.8 | 42 of 42 | 1.52 cm | 90% |
| 6.0 | 42 of 42 | 1.48 cm | 90% |
| 9.6 | 42 of 42 | 1.47 cm | 90% |

**An older empty scene.** Test mode reuses one empty scene for several tests, so each recording was judged
against the scene of an earlier set, and each empty scene against an older one, which is an empty mat with
nothing done to it but time:

| Age of empty scene | Named | Distance rms | Empty mat taken for an object |
|---|---|---|---|
| About 0.7 min, its own | 42 of 42 | 1.46 cm | 0 of 3 empty trials |
| About 2.3 min | 36 of 36 | 1.35 cm | 6 of 18 |
| About 3.9 min | 30 of 30 | 1.93 cm | 8 of 15 |
| About 5.8 min | 23 of 24 | 1.85 cm | 10 of 12 |
| About 7.6 min | 18 of 18 | 2.50 cm | 9 of 9 |
| About 9.2 min | 12 of 12 | 2.75 cm | 6 of 6 |
| About 10.8 min | 4 of 6 | 3.11 cm | 3 of 3 |

Real echoes stand well above the drift, but an empty mat has nothing else, so "nothing there" is the first thing
to fail. Test mode stops starting new tests once the scene is 2 minutes old, so a last test can end at about 2.5
minutes, where a false report is possible. Drift is fastest in the first minutes after setting up.

**Can the blocks be told apart?** Their empty scenes can: every one of the 21 sets is nearest another set in its
own block, and sets differ by −58.5 dB within a block against −49.4 dB across blocks, on 0.2 to 1.3 ms. What the
model is given can't: naming the block from it, with each set held out, gets 8 to 15 of 42, where guessing gets
14. Subtracting the empty scene removes what marks the spot, which is why holding out a block is a fair test. It
doesn't show the model works in a new room: all the blocks look to be one room.

**How long is an echo?** The window holds 94% of a typical echo's energy for the fish and for the bobblehead at 16
to 20 cm, and 79% for the bobblehead at 10 to 14 cm, 66% at worst. Longer windows gained nothing in naming.

## Test mode

The app opens in test mode (`TestActivity`, `Follow`, `LiveView`), which saves nothing. After an empty scene each
test starts itself after a countdown. The echo shows along the mat from the first pings, the first guess comes
2.8 s in, and a new one every 0.4 s from all the pings so far, with a trace of how it has moved. A reading of all
the pings takes about 0.2 s on the phone, and live runs skipped no updates. The percentage is the network's
average P of the object it names, not a measured chance of being right.

## The sealed test

`analysis/sealed.py` and `app/sim/sealed.ps1` hold the tests, pass marks and rules. They were committed before
block D was opened: nothing is retrained afterwards, every recording stays in, and a missed mark stands as the
result. Block D has 14 object recordings and 1 empty trial, so even 14 of 14 would only show accuracy above about
77%. A dry run on block C scored 12 of 14 on distance, one short of the mark, on a block the line was fitted to.

| Test | Pass mark | Result |
|---|---|---|
| Named at 10 s | 12 of 14 | __ |
| Within 2 cm | 13 of 14 | __ |
| Objects found, empty trial rejected | 13 of 14, and rejected | __ |
| Same answer at 2.8 s as at 10 s | 13 of 14 | __ |
| App matches Python | Same answers, gaps as on blocks A to C | __ |

## The first 26 sets

Each object stood about 10 cm in front of the phone, by eye. The first model uses the two crossing paths: each
speaker to the mic at the far end.

| Level | Count | What it is |
|---|---|---|
| Set | 26 | One spot (laminate or tile floor, different distances from walls and furniture): the empty scene, then each object, recorded in turn |
| Recording | 78 (52 with an object) | 10 s after 3 s for the hand to get clear; about 100 pings per speaker |
| Example | 1,248 | 4 consecutive pings averaged (0.4 s); about 24 per recording |

Four sessions on three days: 8 sets, then 6, on 12 October 2026; 1 on 13 October; and 12 on 27
October, of which one is left out (see the second field test).

**Labels.** The app records each object without being told which, shows its guess, then asks. Each
trial yields a label and a prediction made before it.

**Quality checks.** Recordings that clip, lose the pings part way, or keep fewer than 90 are
discarded and repeated; the repeat overwrites them, so none are counted here.

**Splits.** Always by whole set: a recording's examples are near-duplicates, and the objects share a
spot with their empty scene, so a random split would leak. Two tests: each set held out in turn, and
trained on the 15 sets up to 13 October, tested on the 11 recorded on 27 October.

## The first model's results

| Test | CNN | Nearest class mean |
|---|---|---|
| Each of the 26 sets held out in turn (52 recordings) | **47 of 52** (89% of examples) | 46 of 52 (86%) |
| Trained on the 15 sets up to 13 October, tested on the 11 from 27 October (22 recordings) | 18 of 22 (79%) | 16 of 22 (73%) |
| Control: both echoes equally loud over the same noise, each set held out | 44 of 52 (81%) | 45 of 52 (82%) |

**Errors by object**, counted separately because in a real search they would not cost the same.
Held out by set, 4 of the CNN's 5 misses are fish named the bobblehead (sets 190605, 195619, 195930
and 200048); the fifth is the bobblehead of 195822, named the fish. The baseline misses 4 fish and 2
bobbleheads. On the phone, the second field test missed 3 fish and 1 bobblehead, and the first missed
2 fish and 1 bobblehead. Missing the fish is the more common error, but 52 recordings are too few for
a rate.

**Simple against complex**, each set held out in turn:

| Model | Input | Recordings right | Examples right |
|---|---|---|---|
| CNN, about 9,400 parameters | Both crossing paths, 2 × 52 values | 47 of 52 | 89% |
| Nearest class mean | Both crossing paths | 46 of 52 | 86% |
| Nearest class mean | Bottom speaker to top mic only, 52 values | 47 of 52 | 89% |
| CNN | Bottom speaker to top mic only | 46 of 52 | 85% |
| Nearest class mean | 8 statistics of the echo's shape, no timing | 48 of 52 | 86% |
| Logistic regression | 8 statistics of the echo's shape, no timing | 46 of 52 | 83% |

With 52 recordings these cannot be ranked; one or two recordings is within chance. The simplest, a
class mean on one path or on eight shape statistics, does as well as the CNN. The CNN runs on the
phone, but these data do not make the case for it over the baseline.

**Caveat.** The input (paths, delays, how loudness is removed) was chosen after seeing the 6 sets
recorded second. The 11 sets of 27 October were recorded after it was chosen, so the second test
above is the fair one.

## The first field test, and what it changed

The first model on the phone used all four paths out to 3.1 ms. Holding out each of the first 8 sets
it named 16 of 16; on the phone, on the next 6 sets, **9 of 12**. The PC reproduced the same three
errors from the same recordings, so the fault was the model's, not the port's.

In two of the three misses both objects had added nearly the same thing to the empty scene, on every
path, peaking 1 to 3 ms out: something up to half a metre away had changed between the recordings,
inside the model's window. The input was cut to the delays the object can produce, 0.2 to 1.3 ms
after the direct sound, and to the two crossing paths. The one set recorded next, on 13 October, went
1 of 2 both with the first model, still on the phone, and with the new input on the PC; both missed
the fish.

On the 11 sets recorded on 27 October, the narrowed input names 18 of 22 with the CNN, against 14 of
22 for the first input. The baseline goes the other way (16 against 18), so the case for the
narrowing rests on the CNN.

## The second field test

From 18 October the phone ran the model trained on the first 15 sets. On 27 October, 12 new sets were
recorded with it, each object guessed before the phone was told what it was.

![Each recording's P(fish): the phone's answer at the time, and the 15-set model on the PC](steps/real/11_guesses.png)

On the 11 sets kept it named **18 of 22**: 8 of 11 fish and 10 of 11 bobbleheads. The PC, running
the same model on the same recordings, makes the same four mistakes.

The misses are where the echo arrived at an unusual time. In the first 15 sets the fish's echo was
centred 0.63 to 0.97 ms after the direct sound, and the bobblehead's 0.70 to 1.19 ms.

| Set | Recording | Echo centred at | Named |
|---|---|---|---|
| 195930 | Fish | 1.29 ms, later than any before; about what a fish 20 cm out would give | The bobblehead |
| 195822 | Both | 0.61 and 0.63 ms, both early and alike | Each as the other |
| 195619 | Fish | 0.65 ms, at the early edge | The bobblehead |

In 195822 both objects added the same amount in the model's window (−32.7 dB), which fits something
close to the phone changing after the empty scene, or both objects placed closer, more than the
objects themselves. Nothing beyond 1.3 ms changed more than in the earlier sets, so this was not the
half-metre room change of the first field test.

**Set 200754 is left out.** Its recording labelled bobblehead added −30.7 dB in the model's window,
louder than any bobblehead before (at most −49.7 dB) and like a fish. The phone named it the fish at
P(fish) = 0.99, and the recording labelled fish the bobblehead at 0.01. In the app only the first
object is named; the second is filled in as the other, so one mis-tap swaps both labels. Both echoes
also arrived early (0.64 and 0.74 ms), so both objects having stood closer would explain it as well.
Since neither can be confirmed, the set is excluded rather than relabelled, and its files are kept in
`recordings/excluded/`. With it, the phone's score would have been 18 of 24.

## What could be fooling the first model

The measured blocks settled the first of these: real distance mattered, and it is why the window now follows
the echo. The loudness control below was on the fixed window with loudness removed, a different input from
the current model's.

**Distance instead of shape.** Subtracting the empty scene and removing loudness leave the echo's
form, but not when it arrived, and that is the object's distance: each centimetre closer or further
moves the echo 0.046 ms (2.2 samples) on a crossing path. A 2 cm sideways move shifts it only 0.27
samples, as one leg of the path lengthens and the other shortens.

![Recordings named right as the object is moved, each set held out](placement.png)

With the recorded echoes moved in delay and strength as if each held-out object stood 3 cm closer or
further, the CNN names 40 and 31 of 52 instead of 47; the baseline 32 and 29 instead of 46. By 14 cm
the CNN is at chance. Trained on copies moved between 7 and 13 cm, it holds 41 to 48 from 5 to 14 cm.
At the same nominal 10 cm, the fish's echo is centred 0.83 ms after the direct sound and the
bobblehead's 0.96 ms. That may be shape (parts of the bobblehead on longer paths) or where each was
put down; the input can use both, and these data cannot tell them apart. The second field test's
misses are this weakness met for real.

Inputs without timing were tested as an alternative (figure 3; [writeup.md](writeup.md)). Moved from
5 to 15 cm, the current input names 25 to 50 of 52; the echo's autocorrelation 42 to 47, with 47 at
10 cm; eight shape statistics 44 to 49 with logistic regression (46 at 10 cm) and 42 to 48 with a class
mean (48 at 10 cm). Moving recorded echoes does not change the angle the object is seen from, so this
is the least a real move would do.

**Loudness.** The fish's echo is about 10 dB stronger than the bobblehead's, so the model could be
reading level. Each path's loudness is removed, and in the control, with both echoes at the same
level over the same real noise, the CNN still names 44 of 52. What is left is how the energy spreads
over delay: the fish's echo is compact, the bobblehead's spread (median width 105 µs against 143 µs
on the louder crossing path), and the fish's is the narrower in 23 of the 26 sets.

**The room changing** between the empty scene and the object: the cause of the first field failures.
The input now stops at the latest delay an object 10 cm away can produce.

**Near-duplicate examples.** A recording's 24 or so examples share one placement, so every test
holds out whole sets.

**The phone itself:** below.

## What the phone does to the data

These properties of this phone had to be measured before any two recordings could be compared.

| Effect | Size | What is done about it |
|---|---|---|
| Each time audio starts, each speaker's sound lands a fraction of a sample early or late, and stays so | Up to about a sample between recordings (−1.2 to +0.9 found); left alone, larger than the echo | Fitted per recording 0.1 to 0.4 ms after the direct sound, before any echo, and removed with a phase ramp in the frequency domain |
| Level and timing repeat every four pings | Level 1.9 to 2.0% rms (4.7 to 5.0% peak to trough); timing about 1.2 µs | Dividing each ping by its direct sound leaves less than the noise (below) |
| The direct sound has two nearly equal peaks a sample apart | — | The earliest strong one is always taken, so recordings agree |
| The earpiece is quiet at the far mic | −45 dB below its own end; the loudspeaker −22 dB | Its path alone names 38 or 43 of 52 recordings, the loudspeaker's 46 or 47: in effect one transmitter and one receiver |
| Only one recording setting gives two different mics | 24 combinations of source, channel mask and input tried; the mic by the cameras is unreachable | `MIC` source, stereo mask: top and bottom mics |
| The mic stops rising in step with the ping above about a quarter of its range | Doubling the drive from 0.1 to 0.2 gives 0.41 of full scale, not 0.5; a drive of 0.4 clips | Pings at a quarter of the range (media volume 10 or 11); the louder top speaker driven at 0.7; clipped recordings discarded |
| 48 kHz sampling holds nothing above 24 kHz, and the speakers fall away below about 19.5 kHz | The 18 to 22.5 kHz sent would resolve about 3.8 cm; what arrives blurs echoes over about 5 cm | Sets the resolution; rules out a beam (Limits) |

![The four-ping pattern, what it leaves, and accuracy against pings averaged](four_ping.png)

**The four-ping pattern** (`analysis/four_ping.py`). Each speaker's level steps through four values,
and its arrival moves with it (correlation 0.96 for the bottom speaker, −0.92 for the top), as if the
audio output placed each ping a slightly different fraction of a sample late on a four-ping cycle.
Dividing by the direct sound removes the level but not the timing, yet nothing measurable survives:
on the empty scenes, what is left in the model's window falls with the pings averaged exactly as
noise would (−57.7 dB for one, −63.7 dB for four, against −57.6 and −63.6), and three or five pings,
which do not span the pattern, leave no more than four. Held out, one to eight pings per example do
about equally well (CNN 46 to 48 of 52). Averaging four is not needed for the pattern; it lowers the
noise, though at this sample size accuracy barely depends on it.

## How it works

![The signal chain, step by step](steps/0_pipeline.png)

1. **Ping.** 10 ms linear chirps, 18 to 22.5 kHz, with a 2 ms raised-cosine fade at each end. The
   speakers alternate, one sweeping up and one down so they can be told apart, each ten times a second.
2. **Record.** Both mics at 48 kHz, 16 bit: 10 s, after 3 s for the hand to get clear.
3. **Matched filter.** Correlate each mic with each speaker's chirp (template matching): each 10 ms
   sweep becomes a peak about 0.3 ms wide, giving echo strength against delay for every ping.
4. **Find the pings.** Lock on to each speaker's direct sound at the mic beside it; if lost, find it
   again and count it.
5. **Window and reference.** Keep 1 ms before to 9 ms after the direct sound, and divide each ping by
   it (a complex division, so drift in gain and phase cancels).
6. **Reject** recordings that clipped, lost the pings, or kept fewer than 90.
7. **Timing fix.** Slide each speaker's pings by the fraction of a sample that lines them up with the
   empty scene's.
8. **Average and subtract.** Average the pings in fours and coherently subtract the empty scene
   recorded just before, leaving what the object added.
9. **Find the echo.** On the two crossing paths, average the echoes so far and add their power over 0.25 to
   1.6 ms. Under −54 dB nothing is there. Otherwise the edge, where the echo first reaches half of its peak
   power, gives the distance through a straight line, and a window runs from 10 samples before it to 38 after.
10. **Features.** Decibels against the direct sound, −80 to −20 dB scaled onto 0 to 1, with nothing removing
    the level: a 2 × 48 array per example.
11. **Classify and vote.** The model scores each example; the app averages over those so far.

The first model had no step 9: it took a fixed 0.2 to 1.3 ms, each path's own loudness removed, in a 2 × 52 array.

![One cycle of the empty scene in time, and three by frequency](figures/1_what_the_mics_hear.png)

*Figure 1. The empty scene. Each speaker is loudest at the mic beside it; at the far mic its ping is
13 times weaker from the loudspeaker and 175 times from the earpiece. The echoes arrive while the
ping still plays, hidden inside the bursts.*

![Every cycle of the empty scene, the fish and the bobblehead, overlaid](figures/2_every_cycle.png)

*Figure 2. All 99 cycles of one set's three recordings, overlaid. At full scale they look the same.
At the far mic, where the crossing ping is weak, the object's echo overlaps it and reshapes the
burst: by up to 36% of its peak with the fish and 10% with the bobblehead, against 4.7% from ping to
ping with nothing in front.*

![Three descriptions of the echo that do not depend on when it arrived](figures/3_echo_shape.png)

*Figure 3. Three descriptions of the echo without its timing, used in the distance test. The dashed
line and star are a single point, the sharpest echo this sonar can produce: the fish's echo is close
to it, the bobblehead's spread.*

A longer walk through these figures is in [writeup.md](writeup.md); rough figures for every step,
simulated and real, are in `docs/steps/`.

**The model.** A 1-D CNN over delay with the two paths as channels: three blocks of convolution,
batch norm, ReLU and max-pool (16, 32 and 64 channels; kernels 7, 5 and 3), a maximum over delay, and
a linear layer to two classes. About 9,000 parameters; it trains in seconds on a laptop CPU. The
baseline is the nearest class mean on the same input.

## On the phone

The PyTorch model is exported with `torch.export` to ExecuTorch (XNNPACK backend, a 48 KB program)
and runs in a native Android app, with the signal chain reimplemented in Java. Three checks keep
phone and PC in step, for test mode's model:

- ExecuTorch against PyTorch on the PC, over all 1,080 examples of blocks A to C: the largest
  difference in probability is 8e-7.
- At every start the app re-runs one stored example per recording against PyTorch's answer, and test
  mode won't run if they disagree. On the phone all 45 match.
- Replayed on the PC through the app's Java code (`app/sim/check.ps1`), all 45 recordings get the same
  answer as the Python pipeline, within 0.05 in probability, 0.51 cm and 0.44 dB.

The first model's own checks gave 2e-6 and 0.07. It still runs in the background of block collection, so
its guesses are filed with each recording. The phone ran the 15-set model in the second field test, and
the repository also holds that model retrained on all 26 sets.

## Limits

What carries over to a search sonar is the processing, not the physics. The differences, most
important first:

**The targets are acoustically hard; a body in water is not.** Acoustic impedance, density times
sound speed, is about 410 rayl for air, 1.5 million for water, and millions for dense solids. In air
any dense solid reflects about 99.9% of the sound energy at its surface, whatever it is made of, so
both printed objects are near-perfect mirrors and the echo carries their outline and nothing else.
In water, soft tissue (about 1.6 million rayl) is nearly matched to its surroundings and reflects
well under 1% at the skin. A body returns sound mainly from the air in its lungs and from bone, a
real fish mainly from its swim bladder. Those echoes are weaker and far more variable: they depend on
how much gas there is (depth compresses it, and drowning can replace it with water) and on what is
inside the target, not on its shape. Nothing here measures how separable such targets are.

**There is no beam.** Beam width is set by the aperture in wavelengths. At 18 to 22.5 kHz the
wavelength in air is 15 to 19 mm, larger than the phone's speaker and mic openings, so each sends
and receives in nearly all directions; a 10 degree beam would need an aperture about 10 cm across.
The phone cannot go higher, because 48 kHz sampling stops at 24 kHz. A search sonar typically works
at hundreds of kilohertz, where the wavelength in water is a few millimetres and a transducer a few
centimetres across is many wavelengths wide. Its beam gives bearing, keeps everything outside it out
of the echo, and concentrates the energy; one transducer, swept, is enough.

Without a beam, everything in range in every direction lands in one echo: floor, walls, whoever is
nearby. The phone substitutes three things: a transmitter and receiver 16 cm apart, subtraction of
the empty scene, and ignoring delays the object cannot produce.

**A search has no empty scene to subtract.** Subtraction needs the same scene, unchanged, without the
target. Two of the three wrong guesses in the first field test appear to come from something up to
half a metre away changing between two recordings on an indoor floor. Open water is never still, and
there is no recording of it from before the person went in.

**It listens while it transmits.** The 10 ms chirp is 3.4 m long in air, against about 25 cm from
speaker to object to mic, so the echo arrives while the ping is still being sent. That works only
because speaker and mic are separate, at the price of the direct sound reaching the mic stronger than
the echo. A single transducer cannot listen while it transmits: it sends short pings and is blind
close in.

**Scale and scope.** 10 to 20 cm, not tens of metres; one stationary phone on a stand, not a moving
handheld device; two objects, one at a time, on the centre line. It reads distance but not sideways
position: the two crossing paths have the same length, so they can't show it. It has no answer for an
object that is neither of the two: anything strong enough reads as one of them. And 42 recordings from
one evening is a small sample.

## Next data to collect

Each answers a question the current data cannot:

- **Unmarked positions**, measured after the guess: the last part of the final test.
- **More bandwidth:** a chirp of about 10 to 22 kHz could cut the blur from about 5 cm to about 1 cm, and
  show whether shape becomes readable, at the price of being audible. The speakers are unmeasured below 18 kHz.
- **More objects, including two of similar size and different shape:** is the separation shape, or
  only that one object is strong and the other weak?
- **New rooms, floors and days:** the measured blocks come from one evening, and look to be one room. A few
  dB of change in echo strength is enough to hurt this model.
- **The bobblehead at 10 to 14 cm, at measured angles:** why is its echo weak and erratic there?
- **The empty scene twice, with something moved half a metre away in between:** how large is what a
  changed room leaves, next to a real echo? That bounds what subtraction can be trusted with.
- **The phone held in the hand:** movement breaks both the per-recording timing fix and the
  subtraction of the empty scene.

## Reproduce

The raw recordings and photos are not included. For the first model, `echoes.npz` holds what the signal
chain extracts from the 26 sets, enough to rerun its tests from the model input on:

```
pip install -r requirements.txt
cd analysis
python classify.py                          # each set held out in turn; saves the model trained on every set
python classify.py since 20261027_195308     # trained on the 15 sets up to 13 October, tested on the 11 from 27 October
python classify.py grid                     # every window, path and loudness choice that was compared (slow)
```

The current method's scripts need the raw recordings in `recordings/sets`, so they can't be rerun from this
repository, though their saved model and the app's checks' reference answers are in it:

```
python distance.py     # reading distance, each block held out
python follow.py       # naming, each block held out; saves follow.pt
python checks.py       # how soon, older empty scenes, telling blocks apart, echo length
python sealed.py       # a dry run of the final test on block C; `python sealed.py open` is block D
```

`sets.py`, `control.py`, `placement.py`, `shape.py`, `four_ping.py`, `figures.py` and
`steps_real.py` also need the raw recordings; `export.py` needs the `executorch` package in its own
environment (see its docstring). With the recordings in place, `classify.py` rewrites `echoes.npz` and its
model from every set there, measured blocks included. `steps_real.py` compares the phone's answers with
whichever model is saved, and marks the sets that model was not trained on.

The app builds without Gradle or Android Studio, given JDK 17 and the Android SDK (platform 34,
build-tools 34.0.0) in `%LOCALAPPDATA%\Android` as `jdk-17` and `Sdk`:

```
powershell -ExecutionPolicy Bypass -File app\get-libs.ps1    # ExecuTorch 1.5.1 from Maven Central
powershell -ExecutionPolicy Bypass -File app\build.ps1
powershell -ExecutionPolicy Bypass -File app\sim\check.ps1   # the app's own code on the PC, against the recordings
```

## Repository

| | |
|---|---|
| `analysis/scene.py` | A recording to echo profiles: matched filter, reference, timing fix, averaging |
| `analysis/sets.py` | Sets of recordings to echoes |
| `analysis/distance.py` | Reading distance from the echo |
| `analysis/follow.py` | The current model: the window that follows the echo, the tests, and `follow.pt` |
| `analysis/checks.py` | How soon, older empty scenes, telling blocks apart, echo length |
| `analysis/sealed.py` | The final test on block D, committed before it was opened |
| `analysis/classify.py`, `model.py` | The first model's input, the CNN and the baseline, training, held-out tests |
| `analysis/control.py` | Control: both objects equalised in loudness and noise |
| `analysis/placement.py` | The object moved from 5 to 15 cm, imitated on the recordings |
| `analysis/shape.py` | Inputs without timing: aligned echo, autocorrelation, shape statistics |
| `analysis/four_ping.py` | The four-ping pattern, what it leaves, and pings per example |
| `analysis/figures.py` | Figures 1 to 3 |
| `analysis/steps.py`, `steps_real.py` | A rough figure for every step, simulated and real (`docs/steps/`), and the phone's answers against the saved model |
| `analysis/export.py` | PyTorch to ExecuTorch, with the examples the app checks itself against |
| `analysis/echoes.npz` | The echoes of the first 26 sets |
| `app/src/` | The Android app: `Sonar` (audio), `Follow`, `TestActivity` and `LiveView` (test mode), `Net` (ExecuTorch), `Judge` (the first model's chain), `BlockActivity`, `RunSheet` and `Photo` (collecting a block) |
| `app/sim/` | PC checks of the app's code, and the app's part of the final test |
| `app/pull.ps1` | Copies new recordings from the phone |
| `docs/` | Figures, a longer walk through them (`writeup.md`), and how to record a block (`recording.md`) |

## Glossary

| | |
|---|---|
| Ping | One chirp from one speaker: a 10 ms tone sweeping between 18 and 22.5 kHz |
| Path | One speaker to one mic; the two crossing paths go from a speaker at one end to the mic at the other |
| Direct sound | The ping arriving straight from the speaker, before any echo; every ping is measured against it |
| Empty scene | The recording made with nothing in front, just before the objects; subtracted from both |
| Empty trial | A recording with nothing placed, after reaching in as if placing something |
| Set | An empty scene and the two objects, recorded in one spot at one distance |
| Block | Seven sets in one spot, at distances from a shuffled run sheet |
| Example | The average of four consecutive pings: 0.4 s of one recording |
| Edge | Where the echo first rises to half of its peak power: the front of the object |
| Held out | Left out of training and used only for testing; here always a whole block or set at a time |
| Nearest class mean | The baseline: an example is called whichever object's average example it is closer to |
| dB | Ten times the log of a power ratio: −20 dB is a tenth of the amplitude, −40 dB a hundredth |
