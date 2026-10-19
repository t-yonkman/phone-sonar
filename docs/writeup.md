# What the phone hears, and the shape of an echo

*A longer walk through the figures in [details.md](details.md) and the [README](../README.md). They are made by
`analysis/figures.py`, `analysis/shape.py` and `analysis/four_ping.py` from the recordings in
`recordings/sets`, which are not in the repository; the numbers below are the ones they print.*

## 1. What the microphones record

![One cycle of a recording in time, and three cycles by frequency](figures/1_what_the_mics_hear.png)

The phone stands on its long edge. Each end has a speaker and a microphone, 16 cm from the other
end's. Every 100 ms the bottom speaker plays a 10 ms chirp sweeping up from 18 to 22.5 kHz, and 50 ms
later the top speaker plays one sweeping down. Both mics record all the time, 48,000 samples a second.
The figure is the empty scene of one set, with nothing in front of the phone.

**(a) In time.** The vertical axis is the level each sample is stored at: 16-bit numbers, scaled so
that 1 is the loudest the recording can hold. The phone's mics are not calibrated, so there is no
physical unit such as pascals. The pings use about a third of the range (the loudest sample here is
0.34), well short of clipping.

Each speaker is loudest at the mic beside it. The same ping reaches the mic at the far end, 16 cm away,
much weaker: 13 times weaker (−22 dB) from the bottom speaker, and 175 times (−45 dB) from the top
one, so that the bottom mic's trace runs almost flat through the top speaker's burst. The quieter mic
is drawn on top during each ping so that both can be seen. These far-end paths are the ones the model is given, because they cross the space in
front of the phone where the object stands. The paths to the mic beside each speaker serve as its
reference for timing and level.

Nothing in the raw sound looks like an echo. An object 10 cm away echoes about 0.75 ms after its ping
starts, while the 10 ms ping is still playing. On the louder crossing path it is 35 to 55 dB (60 to
600 times) weaker than the ping at the mic beside the speaker. The echoes are inside these bursts. The next step, correlating each mic
with each speaker's chirp, is what pulls them apart, by when they arrive.

**(b) By frequency.** The same recording from the bottom mic, over three cycles, as frequency against
time; darker is louder. The pings sit between 18 and 22.5 kHz: strong up-sweeps from the bottom
speaker beside the mic, faint down-sweeps from the top speaker at the far end. The two sweep in
opposite directions so that each speaker's matched filter answers to its own ping and largely ignores
the other's. The room's own noise is below about 5 kHz, far from the pings, and the matched filter
ignores it.

The top of the plot, 24 kHz, is a hard ceiling: a recording made 48,000 times a second cannot hold
anything above half that. That is why the ping cannot go higher, and why the phone cannot form a beam
(see Limits). 18 kHz is above most adults' hearing.

### With an object in front

![Every cycle of the empty scene, the fish and the bobblehead, overlaid](figures/2_every_cycle.png)

These are the three recordings of one set, made one after another about a minute apart: the empty
scene, then the fish, then the bobblehead, 10 cm in front. Each has 99 cycles of 100 ms, drawn here on
top of each other and lined up on the bottom speaker's ping. Within a recording the cycles fall almost
exactly on top of each other, because the phone plays and records on the same clock.

At full scale (a, c, e) the three look the same. At the mic beside each speaker the loud burst
changes by only a few per cent between the three recordings, too little to see at this scale.

It does show at the far end. The panels on the right enlarge the top mic while the bottom speaker
plays: the ping that has crossed the phone, 13 times weaker than at its own end. The object's echo
reaches the same mic a few tenths of a millisecond later. Both are the same 10 ms sweep, so they
overlap almost completely, and because the sweep's pitch keeps rising, the two drift in and out of
step during the ping: in some stretches they add, in others they cancel. So the burst's outline
changes shape, by up to 36% of its peak with the fish and 10% with the bobblehead, whose echo is
weaker. With nothing in front, the outline varies by 4.7% from one ping to the next.

The outlines are measured in the ping band only, 17 to 23.5 kHz. The fish and bobblehead recordings
happened to have more low room noise than the empty one, which would otherwise raise their outlines
even before the ping starts.

The signal chain does the same thing, more precisely. Correlating with the chirp separates the ping
from the echo by when each arrives, and subtracting the empty scene removes everything that was
already there.

## 2. The shape of the echo, without its distance

![Three descriptions of the echo that do not depend on when it arrived](figures/3_echo_shape.png)

Subtracting the empty scene leaves only what the object added, and removing each path's loudness
leaves the form of its echo. Neither removes *when* the echo arrived, and that is the object's
distance. On a path that crosses the phone, every centimetre closer or further moves the echo 0.046 ms
(2.2 samples) earlier or later and makes it about 1 dB stronger or weaker. A sideways move hardly
matters: 2 cm to one side moves the echo by 0.27 samples, because one leg of the path lengthens as
the other shortens.

The model is given a fixed stretch of delay, 0.2 to 1.3 ms, so it can use where in that stretch the
echo falls, and it does. With each set held out, and its echo moved as if the object had stood 3 cm
closer or further, the CNN names 24 to 26 of the 30 recordings instead of 29 (`analysis/placement.py`).

The figure shows three ways to describe the echo that do not depend on when it arrived, for the bottom
speaker to top mic path, all 15 sets, at 10 cm. Each uses only the stretch where an object can echo:
from 0.46 ms (the 16 cm straight across the phone, the shortest way any sound off the object can
take) to 1.6 ms (an object about 18 cm out). In each panel the dashed line or star is what a single
point would give: the sharpest echo this sonar can produce with its 4.5 kHz of bandwidth.

**(a) Lined up on its own middle.** Each recording's echo middle is found: the power-weighted centre of
that stretch, above the noise, on both crossing paths and over all its examples together. The echo is
slid there by a fraction of a sample, and a window from 0.29 ms before to 0.5 ms after is kept, in dB
relative to its own average. The fish's echo rises and falls much like a single point, with higher
tails. The bobblehead's is flatter and runs on later.

**(b) Compared with itself, shifted.** The autocorrelation asks how much the echo resembles itself
shifted by a given amount: 1 with no shift, falling as the shift grows. Moving an echo in time leaves
it unchanged, so it carries no distance at all. By a shift of 0.5 ms a single point has fallen to 0.10
and the fish to 0.14. The bobblehead stays at 0.34, because its echo comes from parts at different
distances, and they keep resembling each other at shifts equal to their spacing.

**(c) Two numbers.** The echo's width (its rms spread about its middle), and the share of its energy
within 0.1 ms of its peak. A single point gives 95 µs and 0.62. The fish's examples have a median of
105 µs and 0.51: close to a single point, one compact reflecting surface. The bobblehead's have 170 µs
and 0.36, and vary far more. Averaged over each recording, the fish ranges from 100 to 197 µs and the
bobblehead from 109 to 383, so no single number separates all 30 recordings.

What these show is how far the object's reflecting parts are spread along the path, roughly its depth
as the sonar sees it, not its outline. The fish is close to the limit of what this sonar can resolve.

### How well each works

Each set held out in turn (`analysis/shape.py`). The last column moves the held-out set's echo as if the
object had stood anywhere from 5 to 15 cm out.

| Input | Model | Recordings right at 10 cm (examples) | Moved 5 to 15 cm |
|---|---|---|---|
| Fixed stretch, 0.2 to 1.3 ms (now) | CNN | 29 of 30 (96%) | 23 to 30 |
| (a) Lined up on its middle | CNN | 25 (81%) | 23 to 27 |
| (b) Autocorrelation, shifts 0 to 0.5 ms | CNN | 29 (88%) | 24 to 29 |
| (c) Shape statistics, 8 numbers | Logistic regression | 26 (87%) | 26 to 28 |

The shape statistics are four per path: the width, the share near the peak, how far the peak stands
above the average, and how soon the autocorrelation falls to a half.

- **Lining up did worse.** Probably because finding the middle is noisy, and because the window around
  the bobblehead's later middle reaches to about 1.5 ms: the stretch where changes elsewhere in the
  room caused two of the first phone model's three wrong guesses.
- **The autocorrelation** names as many recordings at 10 cm as the input used now, and more of them
  once the object is 3 cm or more off. Single 0.4 s examples are right less often (88% against 96%); the average over a
  recording's 25 or so hides that.
- **The statistics** are the steadiest across distance and the easiest to read, at some cost at 10 cm.

**The catch.** At the same nominal 10 cm, the fish's echo middle is at 0.81 ms and the bobblehead's at
0.98 ms. Part of that 0.17 ms may be shape: parts of the bobblehead, such as its head, on longer paths.
Part may be where each tended to be put down. The input used now can use both; the descriptions above
give up both, which is why they score lower at 10 cm. Only recordings at measured distances can tell
the two apart.

**What the test cannot show.** Moving the recorded echoes imitates the change in delay and strength
only, which is exactly what these descriptions ignore, so their staying level is expected. What slope
remains most likely comes from the edges of the stretch: closer than about 7 cm the echo runs into what is left of
the phone's own sound just after the direct arrival, and past about 13 cm the bobblehead's echo runs
out of the 1.6 ms window. A real move also changes the angle each end of the phone sees the object
from, 49° off the phone's face at 7 cm against 32° at 13 cm, and no recording here shows what that
does. With 30 recordings, a difference of one or two is within chance. The next sets worth recording
are at measured distances, such as 7, 10 and 13 cm from a marked spot.

## 3. The four-ping pattern, and why pings are averaged in fours

![The pattern ping by ping, what it leaves, and accuracy against the number of pings averaged](four_ping.png)

Each speaker's sound comes out of the phone a little differently on every fourth ping, in all 45
recordings (`analysis/four_ping.py`).

**(a) What it is.** Measured at the mic beside each speaker, the level steps through four values and
repeats: 1.6 to 1.7% rms, 4.2 to 4.4% from peak to trough, and anywhere from 0.3 to 6.6% between
recordings. With the pattern taken out, the level varies by only 0.4 to 0.5% from one ping to the
next. The ping's arrival moves with it, by 0.05 to 0.06 of a sample (about 1 µs) from earliest to
latest, against about 0.01 of a sample from ping to ping. Level and timing move together (correlation
0.98 for the bottom speaker, −0.92 for the top), as they would if the phone's audio output were
placing each ping a slightly different fraction of a sample late on a four-ping cycle. That cause is a
guess; the measurement is not.

The earlier README's "about 1.5%" was the rms. The "±2.5%" read off one recording earlier is half its peak to
trough. Both describe the same pattern.

**(b) What it leaves.** Dividing each ping by its own direct sound takes out a change of level but not
a change of timing, so some of the pattern could survive into what the model is given. It does not, in
any amount that can be measured. On the empty scenes, what is left on the crossing paths over the
model's window falls with the number of pings averaged exactly as noise alone would: −58.1 dB for one
ping, −64.2 dB for four, against −58.0 and −64.0 for noise. Averaging three or five pings, which do not
cover the pattern, leaves no more than averaging four. Lining each ping up on its own before averaging
changes nothing.

**(c) What it means for the model.** With each set held out, three to eight pings per example do equally
well (CNN 29 or 30 recordings of 30, 96 to 97% of examples); one or two pings do worse per example (89
to 92%), as more noise would predict. Lining each ping up first does not help. The differences between
three, four and eight are within chance.

So the reason the README used to give for averaging in fours, that it cancels the pattern, is not the reason it
works: four works because it brings the noise down. The model is unchanged, and the explanation in
`scene.py`, `classify.py` and `Judge.java` now says so.
