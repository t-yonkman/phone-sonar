# Phone sonar

An unmodified phone used as an active sonar in air. It pings above hearing from
its own speakers, listens with its own microphones, and a small PyTorch model
running on the phone says which of two objects is standing in front of it: a
3D-printed fish or a bobblehead figure.

It is a deliberately small version of the problem a handheld search sonar
solves: say what a target is from its echo, on the device, from data collected
in the field.

![What each object adds to the empty scene, for all 14 sets](docs/echoes.png)

*Each line is one recording: what the object added to the empty scene on one
speaker-to-mic path. Between the dashed lines, on the two paths that cross the
phone, the fish (blue) and the figure (orange) separate. Elsewhere they do
not, and the model is no longer shown it.*

## Setup

A Galaxy Note10+ stands on its long edge with the object 10 cm in front, so
the two ends of the phone sit either side of it. Each end has a speaker and a
microphone. Two speakers and two mics give four speaker-to-mic paths, each
seeing the object from a different angle.

## Signal chain

1. **Ping.** Linear chirps from 18 to 22.5 kHz, 10 ms long, with a 2 ms
   raised-cosine fade at each end. The two speakers alternate, one sweeping up
   and the other down so they can be told apart, each ten times a second. The
   4.5 kHz bandwidth resolves range to about 3.8 cm.
2. **Capture.** Both mics at 48 kHz, 16 bit. A recording is 10 s, about 100
   pings from each speaker, after 3 s for the hand to get clear.
3. **Pulse compression.** Each mic signal is correlated with each speaker's
   chirp, giving echo strength against delay for every ping.
4. **Reference and background.** Each profile is divided by the speaker's direct
   arrival at the mic beside it (a complex division, so gain and phase drift
   cancel). A recording of the empty scene, made just before, is subtracted
   coherently. What is left is what the object added.
5. **Model input.** The two paths that cross the phone, from 0.2 to 1.3 ms after
   the direct sound (as late as an object 10 cm away can echo), as log
   magnitude with each path's own loudness removed: a 2 x 52 array per example.
   An example is the average of four pings (0.4 s). The same-end paths serve
   as the timing and level reference and are not shown to the model.

Two properties of this phone had to be found and corrected before any two
recordings could be compared:

- Every time audio starts, each speaker's output lands a different fraction of
  a sample early or late. Left alone, that mismatch is larger than the echo. It
  is fitted per recording in the first 0.4 ms after the direct sound, before
  any echo can arrive, and removed with a phase ramp in the frequency domain.
- The speaker level repeats in a four-ping pattern of about 1.5%, which is why
  pings are averaged in fours.

## Model

A 1-D CNN over delay with the paths as channels: three blocks of convolution,
batch norm, ReLU and max-pool (16, 32, 64 channels; kernels 7, 5, 3), a global
maximum over delay, and a linear layer to two classes. About 9,400 parameters;
it trains in seconds on a laptop CPU. The baseline beside it is the nearest
class mean on the same input.

A 10 s recording gives about 25 examples. The model scores each and the app
averages them.

## Data

A *set* is three recordings made in one spot: the empty scene, then each
object. The spots differ (laminate and tile floor, different distances from
walls and furniture). There are 14 sets so far: 28 object recordings, 672
examples.

The app collects its own labelled data. After the background it records an
object without being told which, shows its guess, and only then asks what it
was. Every trial therefore yields a labelled recording, the model's
prediction, and a running field score. Recordings that clip or lose the pings
are not counted and are repeated.

Splits are always by whole set. The examples within a recording are
near-duplicates, so a random split would leak.

## Results

| | CNN | Nearest class mean |
|---|---|---|
| Each of the 14 sets held out in turn | 28 of 28 recordings, 99% of examples | 26 of 28, 93% |
| Trained on the first 8 sets, tested on the 6 recorded later | 12 of 12, 99% | 12 of 12, 96% |
| Control: both objects' echoes made equally strong over the same noise, each set held out | 27 of 28, 88% | 25 of 28, 86% |

The fish's echo is about 10 dB stronger than the figure's, so the control
checks that level is not all the model reads. What is left is how the echo's
energy is spread over delay. One transmitter and one receiver still give a
time series: parts of the scene at different path lengths answer at different
times, resolved here to about 0.2 ms, so roughly five independent values per
path. The fish returns one compact echo (0.16 ms rms on the loudspeaker path)
and the figure a more spread one (0.27 ms). With both at the same level over
the same noise, the fish is still the more compact in all 14 sets.

The first two rows need a caveat, and the caveat is the most useful thing the
project produced.

The first model put on the phone was given all four paths out to 3.1 ms. Held
out a set at a time on the first eight sets it scored 16 of 16. On the phone,
on six new sets, it scored **9 of 12**. The PC reproduced the same three
errors from the same recordings, so the fault was the model's and not the
port's.

In two of the three misses the fish and the figure had both added nearly the
same thing to the background, on every path, peaking 1 to 3 ms out. Something
up to half a metre away had changed since the background was taken, and it
was inside the model's window. The input was then cut down to where the object
can be, and to the two paths that cross the phone (the figure above). That is
the model in the table.

It was chosen after looking at the six later sets, so the table flatters it.
It has not yet been tested on sets recorded after it was built.

## On the phone

The trained PyTorch model is exported with `torch.export` to ExecuTorch with
the XNNPACK backend (a 48 KB program) and runs inside a native Android app.
The signal chain above is reimplemented in Java for the phone. Three checks
keep the phone and the PC honest with each other:

- The exported program is run on the PC over every training example and
  compared with PyTorch: the largest difference in probability is 1e-7.
- The app carries one example per recording with PyTorch's answer to it. It
  re-runs them every time it opens and will not guess if they disagree.
- The recorded sets are replayed on the PC through the app's own Java code and
  compared with the Python pipeline: all 28 recordings come out the same, to
  within 0.03 in probability.

## Limits

This exercises the pipeline of a search sonar, not its physics. The
differences, most important first:

**The targets here are acoustically hard. A body in water is not.** Acoustic
impedance is density times sound speed: about 410 rayl for air, 1.5 million
for water, and millions for dense solids. Against air, any dense solid is a
wall: about 99.9% of the sound energy reflects at its surface, whatever it is
made of. Both printed objects are near-perfect mirrors, and the echo carries
their outline and nothing else. In water, soft tissue (about 1.6 million rayl) is nearly
matched to its surroundings and well under 1% of the energy reflects at the
skin. A body returns sound mainly from the air in its lungs and from bone, and
a real fish mainly from its swim bladder. Those echoes are weaker and far more
variable than a hard object's: they depend on how much gas there is (depth
compresses it, and drowning can replace it with water) and on what is inside
the target, not on its shape. Nothing here measures how separable such targets
are.

**There is no beam.** How narrow a beam can be is set by the aperture in
wavelengths. At 18 to 22.5 kHz the wavelength in air is 15 to 19 mm and the
phone's speaker and mic openings are smaller than that, so each sends and
receives in nearly all directions; a 10 degree beam would need an aperture
about 10 cm across. The phone cannot go higher, because 48 kHz sampling stops
at 24 kHz. A search sonar typically works at hundreds of kilohertz, where the
wavelength in water is a few millimetres and a transducer a few centimetres
across is many wavelengths wide. Its beam gives bearing (the target is where
the beam points), keeps everything outside the beam out of the echo, and
concentrates the energy. One transducer, swept, is then enough.

Without a beam, everything in range in every direction lands in the same
echo: floor, walls, whoever is standing nearby. The phone gets by on three
substitutes. The transmitter and receiver are separate and 16 cm apart, a
speaker at one end and a mic at the other. A recording of the empty scene is
subtracted. And delays the object cannot produce are ignored.

**The second speaker turned out to add little.** It was there to give a
second crossing path from the other side. Given the loudspeaker-to-far-mic
path alone, the CNN names 27 of 28 held-out recordings and the nearest class
mean 28 of 28; given the earpiece path alone, 24 and 22. The earpiece is too
quiet at the far mic. So this is in effect one transmitter and one receiver
with no beam, and with 28 recordings the CNN cannot be told apart from the
baseline.

**A search has no empty scene to subtract.** Background subtraction needs the
same scene, unchanged, without the target. Two of the three wrong guesses in
the field test appear to have come from something up to half a metre away
changing between two recordings on an indoor floor. Open water is never still, and there is no
recording of it from before the person went in.

**It listens while it transmits.** The 10 ms chirp is 3.4 m long in air,
against about 25 cm from speaker to object to mic, so the echo arrives while
the ping is still being sent. That is possible only because speaker and mic
are separate, and the price is the speaker's direct sound arriving at the same
mic, stronger than the echo. A single transducer cannot listen while it
transmits: it sends short pings and is blind close in.

**Scale and scope.** 10 cm, not tens of metres. One stationary phone on a
stand, not a moving handheld device. Two objects at one distance and position.
It classifies an object known to be there; it does not detect or locate. And
14 sets is a small sample, with the one true field test so far being of the
earlier model.

## Repository

| | |
|---|---|
| `analysis/scene.py` | A recording to echo profiles: pulse compression, reference, timing correction |
| `analysis/sets.py` | Sets of recordings to echoes; draws the figure above |
| `analysis/classify.py`, `model.py` | Model input, the CNN, training, held-out tests |
| `analysis/control.py` | Control: both objects equalised in loudness and noise |
| `analysis/export.py` | PyTorch to ExecuTorch, with the examples the app checks itself against |
| `analysis/echoes.npz` | The echoes of all 14 sets |
| `app/src/` | The Android app: `Sonar` (audio), `Judge` (signal chain), `Net` (ExecuTorch), `MainActivity` (collection flow) |
| `app/sim/` | PC checks of the app's code |

The raw recordings are audio of the room they were made in and are not
included. `echoes.npz` holds what the signal chain extracts from them, which
is enough to rerun everything from the model input onwards:

```
pip install -r requirements.txt
cd analysis
python classify.py                  # each set held out in turn
python classify.py since 203955     # train on the first 8 sets, test on the later 6
python classify.py grid             # every window, path and loudness choice that was compared (slow)
```

The app builds without Gradle or Android Studio. With JDK 17 and the Android
SDK (platform 34, build-tools 34.0.0) in `%LOCALAPPDATA%\Android` as `jdk-17`
and `Sdk`:

```
powershell -ExecutionPolicy Bypass -File app\get-libs.ps1    # ExecuTorch 1.5.1 from Maven Central
powershell -ExecutionPolicy Bypass -File app\build.ps1
```
