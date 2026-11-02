# Recording a block

A block is 7 sets in one spot: the distances 10, 12, 14, 16, 18 and 20 cm in a shuffled order, then the first
one again to show any drift. It takes about 10 minutes. The app shows what to do at every step, so nothing is
labelled by hand.

## Setting up

1. Disconnect any Bluetooth audio device, or the pings go to it.
2. Tape the mat down in a new spot for each block: another room, floor or corner.
3. Stand the phone on its kickstand with the back of the case, at the floor, on the mat's zero line. The back
   faces the mat. Distances run from there to the centre of the object's base.
4. Set the media volume to 10 or 11. The app warns if it isn't.
5. Open Phone Sonar, press **Collect**, and pick the block. A block already begun carries on where it stopped.

Blocks A, B and C show the guesses at the end of each set. Block D is the sealed test: it never shows them, and
nobody looks at its results until the method is frozen. **Practice** is 3 sets and is not used for anything.

## Each set

Press **Start** once, for the empty scene. The rest of the set runs by itself: the screen shows each step and
counts down 20 seconds, then 3 seconds more for you to be clear before the recording.

1. **"Clear the mat, then press Start."** Take the last object away, press Start, and step back.
2. **"The fish at 12 cm."** Put it there, with the centre of its base on the mark, and step back.
3. **The other object**, at the same mark, after taking the first away.
4. In one set of each block, **"Nothing this time."** Reach in as if placing something, step back to where you
   would stand after placing an object, and leave nothing. The recording only starts afterwards, so this isn't
   about your hand. It repeats everything else that happens when an object is placed, so that a set with nothing
   added is handled like one with an object.

Each recording takes 13 seconds: 3 to get clear and 10 of sonar. A photo follows, then the next instruction. A
set takes about a minute and a half, and there's no hurry between sets.

**Don't touch or nudge the phone during a set.** Everything in a set is compared with its empty scene. Between
sets the phone can be touched, because each set has its own.

A Bluetooth shutter remote also works, on any set: it starts a recording straight away.

## If something goes wrong

- A recording that clips or loses the pings isn't kept. The screen says why and shows the same step again.
- **Hold** stops the countdown if you won't be ready, and **Start** carries on. Holding means touching the phone,
  so tap gently and flag the set afterwards.
- **Flag set** marks the current or last set as having gone wrong: bumped, wrong object, someone walked past.
  Flagged recordings are kept, and any exclusion follows rules set before looking at the results.
- **Restart set** flags what has been recorded of the set and starts it again.

## Test mode

The app opens in test mode, which says what is on the mat, how far away and how sure, while it pings. It saves
nothing. Set up as for a block, with the mat clear, and press **Start**. After the empty scene the phone isn't
touched again, so each test starts itself after a 20 second countdown: 3 seconds to get clear, then 10 seconds of
pings. The echo shows along the mat from the first pings. The first guess comes about 3 seconds in and updates
every 0.4 seconds from all the pings so far, and a trace shows how it has moved.

Tests run one after another until **Stop**, or until the empty scene is two minutes old, when it asks for a new one.
Leaving the mat empty is a fair test too: the answer should be "Nothing there". The percentage is the network's own
average P of the object it names, which runs high and isn't a measured chance of being right.

## Afterwards

With the phone plugged in, copy everything new to the PC:

```
powershell -ExecutionPolicy Bypass -File app\pull.ps1
```

Recordings go to `recordings/sets`, block D to `recordings/sealed` and practice runs to `recordings/practice`.
None of it goes into the repository, photos included.
