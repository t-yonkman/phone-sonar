package dev.phonesonar;

/**
 * Turns a recording made with something in front of the phone, and the background recorded just before
 * it, into a guess at what the something is. The steps, and their order, are those of blocks() in
 * analysis/scene.py followed by features() in analysis/classify.py, so the network sees here exactly
 * what it was trained on:
 *
 * 1. Each time the app starts, each speaker's sound lands a different fraction of a sample early or
 *    late. Find, per speaker, the slide that best lays this recording's average ping over the
 *    background's in the stretch just after the direct sound, where no echo can have arrived yet. This
 *    uses the pair from the speaker to the mic beside it.
 * 2. For the pair from each speaker to the mic at the far end of the phone: average the pings in fours
 *    (the phone's own sound repeats in a four-ping pattern, which cancels over one turn), slide each
 *    average, measure it against the speaker's own slid direct sound, and take the background away.
 *    What is left is the echo.
 * 3. Express each of those two echoes in decibels along delay, with its own loudness taken out, as far
 *    as an object 10 cm away can echo and no further, and give each four-ping pair of echoes to the
 *    network. The guess is the average of its answers.
 *
 * The network is not given the two same-end pairs or anything later than 1.3 ms. On the first sets the
 * phone guessed at, both let changes elsewhere in the room in, and cost it three recordings in twelve.
 */
final class Judge {
    static final int W = Sonar.W, PRE = Sonar.PRE;
    static final int LO = PRE + 10, HI = PRE + 62;           // delays given to the network: 0.2 to 1.3 ms
    static final int GUARD0 = PRE + 5, GUARD1 = PRE + 20;    // 0.1 to 0.4 ms: too soon for an echo
    static final int BLOCK = 4;
    private static final double[] COS = new double[W], SIN = new double[W];

    static {
        for (int j = 0; j < W; j++) {
            COS[j] = Math.cos(2 * Math.PI * j / W);
            SIN[j] = Math.sin(2 * Math.PI * j / W);
        }
    }

    static final class Result {
        float[] p = {};          // how likely each of Classifier.CLASSES is
        double[] slide = {};     // per speaker, samples
        double level;            // the loudness of the echoes the network was given, dB below the direct sound
        int examples;
    }

    /** A profile's spectrum, so that it can be read off at a fraction of a sample. */
    private static void spectrum(float[] re, float[] im, double[] outRe, double[] outIm) {
        for (int f = 0; f < W; f++) {
            double a = 0, b = 0;
            for (int k = 0, j = 0; k < W; k++, j = (j + f) % W) {
                a += re[k] * COS[j] + im[k] * SIN[j];
                b += im[k] * COS[j] - re[k] * SIN[j];
            }
            outRe[f] = a;
            outIm[f] = b;
        }
    }

    /** Turn a spectrum into that of the same profile read off d samples later. */
    private static void slide(double[] xRe, double[] xIm, double d, double[] outRe, double[] outIm) {
        for (int f = 0; f < W; f++) {
            double th = 2 * Math.PI * (f < W / 2 ? f : f - W) * d / W, c = Math.cos(th), s = Math.sin(th);
            outRe[f] = xRe[f] * c - xIm[f] * s;
            outIm[f] = xRe[f] * s + xIm[f] * c;
        }
    }

    /** The profile with this spectrum, at delay k: {real, imaginary}. */
    private static void at(double[] xRe, double[] xIm, int k, double[] out) {
        double a = 0, b = 0;
        for (int f = 0, j = 0; f < W; f++, j = (j + k) % W) {
            a += xRe[f] * COS[j] - xIm[f] * SIN[j];
            b += xRe[f] * SIN[j] + xIm[f] * COS[j];
        }
        out[0] = a / W;
        out[1] = b / W;
    }

    /**
     * @param re,im   this recording's pings: [pair][ping][delay], each measured against its speaker's direct sound
     * @param count   how many pings there are
     * @param bgRe,bgIm the background's average ping: [pair][delay]
     * @param pairs   per speaker, the indices of its pairs: the one with the mic beside it, then the one with the mic at the far end
     */
    static Result judge(float[][][] re, float[][][] im, int count, float[][] bgRe, float[][] bgIm, int[][] pairs, Classifier net) {
        Result r = new Result();
        int blocks = count / BLOCK;
        r.slide = new double[pairs.length];
        r.examples = blocks;
        float[][][] x = new float[blocks][pairs.length][HI - LO];   // each speaker's far echo: its power, then the network's input
        double[] sRe = new double[W], sIm = new double[W], mRe = new double[W], mIm = new double[W], v = new double[2], ref = new double[2];
        float[] avRe = new float[W], avIm = new float[W];
        for (int s = 0; s < pairs.length; s++) {
            int own = pairs[s][0], far = pairs[s][1];
            // 1. the slide, from the whole recording's average ping on the pair with the mic beside the speaker
            average(re[own], im[own], 0, count, avRe, avIm);
            spectrum(avRe, avIm, sRe, sIm);
            double best = Double.MAX_VALUE;
            for (int step = -250; step <= 250; step++) {   // up to a sample each for the fraction and for which lobe was taken as the arrival
                double d = step / 100.0, cost = 0;
                slide(sRe, sIm, d, mRe, mIm);
                at(mRe, mIm, PRE, ref);
                double mag2 = ref[0] * ref[0] + ref[1] * ref[1];
                for (int k = GUARD0; k < GUARD1; k++) {
                    at(mRe, mIm, k, v);
                    double qRe = (v[0] * ref[0] + v[1] * ref[1]) / mag2 - bgRe[own][k], qIm = (v[1] * ref[0] - v[0] * ref[1]) / mag2 - bgIm[own][k];
                    cost += qRe * qRe + qIm * qIm;
                }
                if (cost < best) {
                    best = cost;
                    r.slide[s] = d;
                }
            }
            // 2. each four-ping average of the far pair, slid, against the speaker's own slid direct sound, less the background
            for (int b = 0; b < blocks; b++) {
                average(re[own], im[own], b * BLOCK, BLOCK, avRe, avIm);
                spectrum(avRe, avIm, sRe, sIm);
                slide(sRe, sIm, r.slide[s], mRe, mIm);
                at(mRe, mIm, PRE, ref);
                double mag2 = ref[0] * ref[0] + ref[1] * ref[1];
                average(re[far], im[far], b * BLOCK, BLOCK, avRe, avIm);
                spectrum(avRe, avIm, sRe, sIm);
                slide(sRe, sIm, r.slide[s], mRe, mIm);
                for (int k = LO; k < HI; k++) {
                    at(mRe, mIm, k, v);
                    double qRe = (v[0] * ref[0] + v[1] * ref[1]) / mag2 - bgRe[far][k], qIm = (v[1] * ref[0] - v[0] * ref[1]) / mag2 - bgIm[far][k];
                    x[b][s][k - LO] = (float) (qRe * qRe + qIm * qIm);
                }
            }
        }
        // 3. decibels with each echo's own loudness taken out, scaled to 0..1 over 60 dB, and the network's answer
        r.p = new float[Classifier.CLASSES.length];
        double total = 0;
        for (int b = 0; b < blocks; b++) {
            for (float[] pair : x[b]) {
                double sum = 0;
                for (float pw : pair) sum += pw;
                total += sum;
                double loud = 10 * Math.log10(sum / pair.length);
                for (int k = 0; k < pair.length; k++) {
                    double db = 20 * Math.log10(Math.sqrt(pair[k]) + 1e-12) - loud;
                    pair[k] = (float) ((Math.max(-50, Math.min(10, db)) + 50) / 60);
                }
            }
            float[] p = net.classify(x[b]);
            for (int c = 0; c < p.length; c++) r.p[c] += p[c] / blocks;
        }
        r.level = 10 * Math.log10(total / Math.max(blocks, 1) + 1e-30);
        return r;
    }

    private static void average(float[][] re, float[][] im, int from, int n, float[] outRe, float[] outIm) {
        for (int k = 0; k < W; k++) {
            float a = 0, b = 0;
            for (int i = from; i < from + n; i++) {
                a += re[i][k];
                b += im[i][k];
            }
            outRe[k] = a / n;
            outIm[k] = b / n;
        }
    }
}
