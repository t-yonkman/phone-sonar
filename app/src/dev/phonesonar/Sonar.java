package dev.phonesonar;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Locale;

/**
 * Makes one recording of fixed length: pings alternately from the bottom and top speakers and records
 * the top and bottom microphones. The phone stands on its long edge with the target in front of it, so
 * the two ends sit either side of the target and each speaker-to-mic pair hears it from a different
 * angle. Recordings come in sets of three made one after another in the same surroundings: the
 * background with nothing in front, then each object. A background recording hands back its average
 * ping; an object recording is judged against that (see Judge) and hands back a guess.
 * Runs on its own thread: a few seconds for the hand that pressed Start to get clear, then the
 * recording, then it saves, guesses and reports back.
 */
final class Sonar extends Thread {
    static final String TAG = "sonar";
    static final int FS = 48000;
    static final double F0 = 18000, F1 = 22500;   // chirp band, Hz: above most adults' hearing; resolves about 3.8 cm
    static final int CHIRP = 480;                 // chirp length in samples (10 ms)
    static final int FADE = 96;                   // fade in and out over 2 ms, so it starts and stops without a click
    static final int CYCLE = 4800;                // each speaker pings every 100 ms: bottom first, top half way through
    static final int SLACK = 96;                  // the top speaker's ping is looked for within 2 ms of where it should be
    static final int PRE = 48, POST = 432;        // window shown around a ping's direct arrival: -1 ms .. +9 ms
    static final int W = PRE + POST;
    static final int BUF = FS * 4;
    static final int SETTLE = 3, LENGTH = 10;     // seconds thrown away after Start, then seconds kept
    // Source MIC with a stereo mask is the one combination the recording API offers that gives two
    // different mics on a Note10+: the top one on channel 0 and the bottom one on channel 1. Every other
    // source, mask and input was tried (24 combinations): each gives one mic copied to both channels,
    // the input called "back" is the top mic again, and a third channel comes back silent. The mic by
    // the cameras is not reachable this way.
    static final int SOURCE = MediaRecorder.AudioSource.MIC, MASK = AudioFormat.CHANNEL_IN_STEREO;
    static final String[] MICS = {"top mic", "bottom mic"};
    // The top speaker is heard louder at its own mic than the bottom one is at its (0.32 against 0.22
    // for the same drive), so it is driven less, to keep both near a quarter of the mics' range.
    static final float TOP_TRIM = 0.7f;

    interface Listener {
        void onUpdate(Update u);
    }

    static final class Update {
        String text = "", diag = "";
        String[] names = {};      // one per line of the picture: which speaker to which mic
        float[][] profile = {};   // per line: what the latest ping sounded like across the window, dB below its direct sound
        boolean finished, ok;     // the last update of a run; ok if the recording was saved and nothing disturbed it
        // with the last update of a good run:
        String file, settings;    // the recording's name without ending, and the settings it was made with (the inside of a JSON object)
        float[][][] scene;        // from a background run: its average ping, [real, imaginary][pair][delay]
        Judge.Result guess;       // from a run with something in front: what the network makes of it
    }

    /** One speaker's pings: its chirp, and what each mic made of the latest one. */
    private static final class Ping {
        final String name;
        final float[] re = new float[CHIRP], im = new float[CHIRP];   // re is played, re + i*im is the matched filter
        int after;          // its direct arrival comes this many samples after the bottom speaker's
        int mic = -1;       // the mic that hears it loudest, beside it
        float[][] profile;
        float[] level;
        int[] offset;

        /** The two speakers sweep opposite ways, so a recording shows which ping came from which. */
        Ping(String name, boolean up) {
            this.name = name;
            double T = (double) CHIRP / FS, fa = up ? F0 : F1, fb = up ? F1 : F0;
            for (int i = 0; i < CHIRP; i++) {
                double t = (double) i / FS, ph = 2 * Math.PI * (fa * t + (fb - fa) * t * t / (2 * T));
                int edge = Math.min(i, CHIRP - 1 - i);
                double env = edge < FADE ? 0.5 - 0.5 * Math.cos(Math.PI * edge / FADE) : 1;
                re[i] = (float) (env * Math.cos(ph));
                im[i] = (float) (env * Math.sin(ph));
            }
        }
    }

    volatile boolean running = true;   // cleared from outside to abandon the run without saving

    private final String set, label;
    private final int place;
    private final float amp;
    private final int volume;
    private final float[][][] scene;   // the set's background, to judge this recording against; null if this is the background
    private final Classifier net;
    private final File outDir;
    private final Listener listener;

    private final Ping[] pings = {new Ping("bottom spk", true), new Ping("top spk", false)};
    private final float[] pRe = new float[W], pIm = new float[W];
    private final float[] aRe = new float[CYCLE], aIm = new float[CYCLE];
    private int nch;
    private float[][] buf;
    private float[] peak, loudest;
    private float[][][] qRe, qIm;   // every kept ping, [pair][ping][delay], measured against its speaker's direct sound
    private short[] raw;
    private long rawFrames;
    private double energy, ref;
    private int n, idx = -1, hold = FS / 2, relocks, count, misses, kept;
    private boolean keeping;

    /**
     * @param set    what the three recordings made together are filed under
     * @param place  1, 2 or 3: this recording's place in the set
     * @param label  what to call it on screen while it records
     * @param volume the phone's media volume step, only noted alongside the recording
     * @param scene  the average ping of this set's background, for guessing what is in front; null to record the background itself
     * @param net    the network that does the guessing, or null to only record
     */
    Sonar(String set, int place, String label, float amp, int volume, float[][][] scene, Classifier net, File outDir, Listener listener) {
        super("sonar");
        this.set = set;
        this.place = place;
        this.label = label;
        this.amp = amp;
        this.volume = volume;
        this.scene = scene;
        this.net = net;
        this.outDir = outDir;
        this.listener = listener;
        for (int i = 0; i < CHIRP; i++) energy += pings[0].re[i] * pings[0].re[i] + pings[0].im[i] * pings[0].im[i];
    }

    @Override
    public void run() {
        AudioTrack track = null;
        AudioRecord rec = null;
        String error = null;
        try {
            track = makeTrack();
            int min = AudioRecord.getMinBufferSize(FS, MASK, AudioFormat.ENCODING_PCM_16BIT);
            rec = new AudioRecord(SOURCE, FS, MASK, AudioFormat.ENCODING_PCM_16BIT, Math.max(min, CYCLE * 2 * 2 * 2));
            if (rec.getState() != AudioRecord.STATE_INITIALIZED) throw new IllegalStateException("the microphones could not be opened");
            configure(rec.getChannelCount());
            rec.startRecording();
            track.play();
            Log.i(TAG, "started: set " + set + ", " + place + " " + label + ", amp " + amp + ", media volume " + volume);
            short[] in = new short[CYCLE / 4 * nch];   // 25 ms
            while (running && rawFrames < (long) (SETTLE + LENGTH) * FS) {
                int got = rec.read(in, 0, in.length);
                if (got <= 0) throw new IllegalStateException("microphone read returned " + got);
                feed(in, got / nch);
            }
        } catch (Throwable e) {
            Log.e(TAG, "stopped on error", e);
            error = String.valueOf(e.getMessage());
        } finally {
            if (track != null) track.release();
            if (rec != null) rec.release();
        }
        finish(error);
    }

    /** Decide whether the recording is good, save it if there is anything to save, and report. */
    private void finish(String error) {
        Update u = new Update();
        u.finished = true;
        if (error != null) {
            u.text = "error: " + error;
        } else if (!running) {
            u.text = "stopped early: nothing saved";
        } else if (kept == 0) {
            u.text = "couldn't hear the pings: nothing saved\nis sound going to Bluetooth, or media volume at zero?";
        } else {
            boolean clipped = false;
            for (int c = 0; c < nch; c++) clipped |= loudest[c] > 0.99;
            u.ok = kept >= 9 * LENGTH && relocks == 0 && !clipped;   // ten pings a second from each speaker
            u.file = save();
            u.settings = settings();
            if (u.file == null) {
                u.ok = false;
                u.text = "could not save the recording";
            } else if (!u.ok) {
                u.text = clipped ? "TOO LOUD for a mic: turn media volume down a step" : "lost the pings part way: something disturbed it";
            } else if (scene == null) {
                u.text = "saved " + label;
                u.scene = average();
            } else {
                u.text = "saved " + label;
                if (net != null) {
                    say("working out what it is");
                    // the pairs, speaker by speaker, the one with the mic beside the speaker first
                    int[][] pairs = new int[pings.length][nch];
                    for (int s = 0; s < pings.length; s++) {
                        pairs[s][0] = s * nch + pings[s].mic;
                        for (int c = 0, i = 1; c < nch; c++) if (c != pings[s].mic) pairs[s][i++] = s * nch + c;
                    }
                    u.guess = Judge.judge(qRe, qIm, kept, scene[0], scene[1], pairs, net);
                }
            }
        }
        Log.i(TAG, "finished: " + u.text.replace('\n', ' ') + (u.ok ? "" : "  (not counted)"));
        listener.onUpdate(u);
    }

    /**
     * One cycle of audio in a static buffer, looped by the mixer so the pings are exactly CYCLE samples
     * apart: the bottom speaker (right channel) at the start, the top speaker (left channel) half way.
     */
    private AudioTrack makeTrack() {
        short[] out = new short[CYCLE * 2];
        for (int i = 0; i < CHIRP; i++) {
            out[2 * i + 1] = (short) Math.round(32767 * amp * pings[0].re[i]);
            out[2 * (CYCLE / 2 + i)] = (short) Math.round(32767 * amp * TOP_TRIM * pings[1].re[i]);
        }
        AudioTrack t = new AudioTrack.Builder()
                .setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build())
                .setAudioFormat(new AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(FS)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                        .build())
                .setTransferMode(AudioTrack.MODE_STATIC)
                .setBufferSizeInBytes(out.length * 2)
                .build();
        t.write(out, 0, out.length);
        t.setLoopPoints(0, CYCLE, -1);
        return t;
    }

    /** The average of the kept pings: [real, imaginary][pair][delay]. */
    private float[][][] average() {
        float[][][] out = new float[2][qRe.length][W];
        for (int p = 0; p < qRe.length; p++) {
            for (int i = 0; i < kept; i++) {
                for (int k = 0; k < W; k++) {
                    out[0][p][k] += qRe[p][i][k] / kept;
                    out[1][p][k] += qIm[p][i][k] / kept;
                }
            }
        }
        return out;
    }

    private void configure(int channels) {
        nch = channels;
        buf = new float[nch][BUF];
        peak = new float[nch];
        loudest = new float[nch];
        qRe = new float[pings.length * nch][LENGTH * FS / CYCLE][W];
        qIm = new float[pings.length * nch][LENGTH * FS / CYCLE][W];
        raw = new short[LENGTH * FS * nch];
        for (Ping p : pings) {
            p.profile = new float[nch][W];
            p.level = new float[nch];
            p.offset = new int[nch];
        }
    }

    private void feed(short[] in, int frames) {
        if (!keeping && rawFrames >= (long) SETTLE * FS) {   // the part that is kept starts here: judge it on its own
            keeping = true;
            Arrays.fill(loudest, 0);
            relocks = 0;
        }
        for (int i = 0; i < frames * nch; i++) raw[(int) ((rawFrames * nch + i) % raw.length)] = in[i];
        rawFrames += frames;

        if (n + frames > BUF) {   // keep the newest half
            for (float[] b : buf) System.arraycopy(b, BUF / 2, b, 0, n - BUF / 2);
            n -= BUF / 2;
            if (idx >= 0) idx -= BUF / 2;
        }
        for (int c = 0; c < nch; c++) {
            peak[c] *= 0.95f;
            for (int i = 0; i < frames; i++) {
                float v = Math.abs(in[nch * i + c] / 32768f);
                buf[c][n + i] = in[nch * i + c] / 32768f;
                if (v > peak[c]) peak[c] = v;
                if (v > loudest[c]) loudest[c] = v;
            }
        }
        n += frames;

        if (idx < 0 && (hold -= frames) <= 0) acquire();
        while (idx >= 0 && idx + CYCLE / 2 + SLACK + POST + CHIRP <= n) step();
    }

    /** Matched filter output, for one speaker's chirp, at count buffer positions starting at j0. */
    private static void mfRange(float[] x, Ping p, int j0, int count, float[] outRe, float[] outIm) {
        for (int k = 0; k < count; k++) {
            float a = 0, b = 0;
            for (int t = 0, j = j0 + k; t < CHIRP; t++, j++) {
                a += x[j] * p.re[t];
                b -= x[j] * p.im[t];
            }
            outRe[k] = a;
            outIm[k] = b;
        }
    }

    /**
     * Find the pings. First the bottom speaker's: its chirp's strongest arrival on any mic in a full
     * cycle. Then the top speaker's, which was played exactly half a cycle later: its strongest arrival
     * on any mic close to where that puts it.
     */
    private void acquire() {
        Ping bottom = pings[0], top = pings[1];
        int s = n - 2 * CYCLE - CHIRP;
        float best = 0;
        for (int c = 0; c < nch; c++) {
            mfRange(buf[c], bottom, s, CYCLE, aRe, aIm);
            for (int k = 0; k < CYCLE; k++) {
                float pw = aRe[k] * aRe[k] + aIm[k] * aIm[k];
                if (pw > best) {
                    best = pw;
                    idx = s + k;
                    bottom.mic = c;
                }
            }
        }
        // The direct sound can arrive as two nearly equal lobes a few samples apart. Take the earliest strong
        // one every time, so the time reference does not depend on which happened to be larger just now.
        int from = Math.max(idx - 16, 0), cnt = idx - from + 2;
        mfRange(buf[bottom.mic], bottom, from, cnt, aRe, aIm);
        for (int k = 1; k < cnt - 1; k++) {
            float pw = aRe[k] * aRe[k] + aIm[k] * aIm[k];
            if (pw >= 0.5f * best && pw >= aRe[k - 1] * aRe[k - 1] + aIm[k - 1] * aIm[k - 1]
                    && pw >= aRe[k + 1] * aRe[k + 1] + aIm[k + 1] * aIm[k + 1]) {
                idx = from + k;
                best = pw;
                break;
            }
        }
        ref = Math.sqrt(best);
        float loud = 0;
        for (int c = 0; c < nch; c++) {
            mfRange(buf[c], top, idx + CYCLE / 2 - SLACK, 2 * SLACK, aRe, aIm);
            for (int k = 0; k < 2 * SLACK; k++) {
                float pw = aRe[k] * aRe[k] + aIm[k] * aIm[k];
                if (pw > loud) {
                    loud = pw;
                    top.after = CYCLE / 2 - SLACK + k;
                    top.mic = c;
                }
            }
        }
        Log.i(TAG, String.format(Locale.US, "acquired: bottom speaker loudest on mic %d at level %.4f; top speaker on mic %d at %.4f, %+d samples from half way",
                bottom.mic, 2 * ref / energy, top.mic, 2 * Math.sqrt(loud) / energy, top.after - CYCLE / 2));
    }

    /** Look at one ping from each speaker; the bottom speaker's direct arrival is expected at buffer position idx. */
    private void step() {
        Ping bottom = pings[0];
        mfRange(buf[bottom.mic], bottom, idx - PRE, W, pRe, pIm);
        float bestPow = 0, mag2 = pRe[PRE] * pRe[PRE] + pIm[PRE] * pIm[PRE];
        for (int k = 0; k < W; k++) bestPow = Math.max(bestPow, pRe[k] * pRe[k] + pIm[k] * pIm[k]);
        double mag = Math.sqrt(mag2);
        if (2 * mag / energy < 0.002 || bestPow > 6 * mag2 || mag < 0.5 * ref || mag > 2 * ref) {
            // nothing there, or the direct arrival is not where it should be (audio glitch): find it again
            idx = -1;
            hold = CYCLE;
            relocks++;
            misses++;
            publish();
            return;
        }
        ref += 0.1 * (mag - ref);
        misses = 0;
        count++;
        boolean keep = keeping && kept < qRe[0].length;
        for (int s = 0; s < pings.length; s++) {
            Ping p = pings[s];
            int at = idx + p.after - PRE;
            mfRange(buf[p.mic], p, at, W, pRe, pIm);
            float a = pRe[PRE], b = pIm[PRE], own = Math.max(a * a + b * b, 1e-12f);   // its direct sound at the mic beside it
            for (int c = 0; c < nch; c++) {
                mfRange(buf[c], p, at, W, pRe, pIm);   // the same stretch of time at every mic
                float top = 0;
                for (int k = 0; k < W; k++) {
                    float pw = pRe[k] * pRe[k] + pIm[k] * pIm[k];
                    p.profile[c][k] = (float) (10 * Math.log10(pw / own + 1e-12));
                    if (pw > top) {
                        top = pw;
                        p.offset[c] = k - PRE;
                    }
                    if (keep) {   // divided by the direct sound (a + ib), so gain and phase drift cancel
                        qRe[s * nch + c][kept][k] = (pRe[k] * a + pIm[k] * b) / own;
                        qIm[s * nch + c][kept][k] = (pIm[k] * a - pRe[k] * b) / own;
                    }
                }
                p.level[c] = (float) (2 * Math.sqrt(top) / energy);
            }
        }
        if (keep) kept++;
        idx += CYCLE;
        publish();
    }

    /** Put a line on screen without touching the picture. */
    private void say(String text) {
        Update u = new Update();
        u.text = text;
        listener.onUpdate(u);
    }

    private static String mic(int c) {
        return c < MICS.length ? MICS[c] : "mic " + c;
    }

    private void publish() {
        Update u = new Update();
        long left = (long) (SETTLE + LENGTH) * FS - rawFrames;
        u.text = keeping ? String.format(Locale.US, "recording %s\n%d s left", label, left / FS + 1)
                : String.format(Locale.US, "%s: hands away\nstarting in %d", label, (left - (long) LENGTH * FS) / FS + 1);
        StringBuilder d = new StringBuilder();
        if (idx < 0) {
            u.text += misses < 20 ? "\n(looking for the pings)" : "\n(can't hear the pings)";
        } else {
            u.names = new String[pings.length * nch];
            u.profile = new float[pings.length * nch][];
            int i = 0;
            for (Ping p : pings) {
                for (int c = 0; c < nch; c++, i++) {
                    u.names[i] = p.name + " > " + mic(c);
                    u.profile[i] = p.profile[c].clone();
                    d.append(String.format(Locale.US, "%-23s level %.3f, %+3d samples\n", u.names[i], p.level[c], p.offset[c]));
                }
            }
        }
        for (int c = 0; c < nch; c++) d.append(String.format(Locale.US, "%s input peak %.2f   ", mic(c), peak[c]));
        d.append(String.format(Locale.US, "\nrelocks %d   media volume %d   amp %.2f", relocks, volume, amp));
        u.diag = d.toString();
        if (count % 10 == 0) Log.i(TAG, u.text.replace('\n', ' ') + " | " + u.diag.replace('\n', '|'));
        listener.onUpdate(u);
    }

    /**
     * Write the kept part of the recording as a WAV with one channel per mic. It is named for its set and
     * its place in it; the activity adds what was in front once that is known, and writes the settings
     * (see {@link #settings}) beside it.
     *
     * @return the name it was saved under, without ending, or null if it could not be
     */
    private String save() {
        int frames = (int) Math.min(rawFrames, (long) LENGTH * FS);
        if (frames == 0 || outDir == null) return null;
        String name = String.format(Locale.US, "set_%s_%d", set, place);
        ByteBuffer bb = ByteBuffer.allocate(44 + frames * nch * 2).order(ByteOrder.LITTLE_ENDIAN);
        bb.put("RIFF".getBytes()).putInt(36 + frames * nch * 2).put("WAVEfmt ".getBytes());
        bb.putInt(16).putShort((short) 1).putShort((short) nch).putInt(FS).putInt(FS * nch * 2).putShort((short) (nch * 2)).putShort((short) 16);
        bb.put("data".getBytes()).putInt(frames * nch * 2);
        long first = (rawFrames - frames) * nch;
        for (int i = 0; i < frames * nch; i++) bb.putShort(raw[(int) ((first + i) % raw.length)]);
        try (FileOutputStream os = new FileOutputStream(new File(outDir, name + ".wav"))) {
            os.write(bb.array());
            Log.i(TAG, "saved " + name + " (" + frames / FS + " s)");
            return name;
        } catch (Exception e) {
            Log.e(TAG, "could not save recording", e);
            return null;
        }
    }

    /** How the recording was made, as the inside of a JSON object. */
    private String settings() {
        StringBuilder sp = new StringBuilder(), pk = new StringBuilder();
        for (Ping p : pings) {
            StringBuilder lv = new StringBuilder();
            for (int c = 0; c < nch; c++) lv.append(c > 0 ? ", " : "").append(String.format(Locale.US, "%.4f", p.level[c]));
            sp.append(sp.length() > 0 ? ", " : "").append(String.format(Locale.US,
                    "{\"name\": \"%s\", \"sweep\": \"%s\", \"amp\": %.3f, \"after_bottom_samples\": %d, \"reference_channel\": %d, \"direct_level\": [%s]}",
                    p.name.replace(" spk", ""), p == pings[0] ? "up" : "down", p == pings[0] ? amp : amp * TOP_TRIM, p.after, p.mic, lv));
        }
        for (int c = 0; c < nch; c++) pk.append(c > 0 ? ", " : "").append(String.format(Locale.US, "%.3f", loudest[c]));
        return String.format(Locale.US, "\"set\": \"%s\", \"place_in_set\": %d, \"seconds\": %.1f, \"settle_seconds\": %d, "
                        + "\"fs\": %d, \"f0\": %.0f, \"f1\": %.0f, \"chirp\": %d, \"fade\": %d, \"cycle\": %d, \"amp\": %.2f, \"media_volume\": %d, "
                        + "\"channels\": %d, \"channel_mics\": [\"top\", \"bottom\"], \"speakers\": [%s], \"loudest_sample\": [%s], \"relocks\": %d",
                set, place, Math.min(rawFrames, (long) LENGTH * FS) / (double) FS, SETTLE, FS, F0, F1, CHIRP, FADE, CYCLE, amp, volume, nch, sp, pk, relocks);
    }
}
