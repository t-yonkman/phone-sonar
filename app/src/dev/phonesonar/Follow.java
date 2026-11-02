package dev.phonesonar;

import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;
import java.util.Scanner;

/**
 * Finds what is in front, reads how far away it is, and only then names it, from however many pings there are so
 * far. The steps and their order are those of analysis/follow.py, so the network sees here what it was trained on:
 *
 * 1. As in Judge, turn each four pings into an echo against the empty scene, on the two paths that cross the phone.
 * 2. Average those echoes. Their loudest point has to pass the level the empty trials stayed under, or nothing is there.
 * 3. The average's edge, where it first rises to half of its peak power, gives the distance through a straight
 *    line fitted on the measured blocks. Both come from the export, in follow.txt.
 * 4. A 1 ms window starting just before that edge, in dB against the direct sound with nothing taken out, goes to
 *    the network, one four-ping echo at a time. The answer is the average of its answers.
 *
 * Everything is worked out afresh from all the pings each time, so with all of a recording's pings the answer is
 * the one analysis/follow.py gives for it.
 */
final class Follow {
    static final int W = Sonar.W, PRE = Sonar.PRE, BLOCK = Judge.BLOCK;
    static final int FROM = PRE + 12, TO = PRE + 76;   // 0.25 to 1.6 ms: where an object 10 to 20 cm away can echo
    static final int BEFORE = 10, AFTER = 38;          // the window given to the network, around the echo's edge
    static final int LAST = TO + AFTER;                // the latest delay ever needed
    static final int MS = Sonar.FS / 1000;

    static final class Reading {
        boolean found;      // there is an echo strong enough to be an object
        double cm;          // how far away, from the echo's edge
        double level;       // the echo's loudest point, dB below the direct sound
        float[] p = {};     // how likely each of Classifier.CLASSES is, if found
        float[] echo = {};  // the echo's power along delay from FROM, dB below the direct sound
        int examples;
    }

    private final float[][] bgRe, bgIm;
    private final Classifier net;
    private final double cmPerMs, cmAtZero, detect;

    /** @param scene the empty scene's average ping, [real, imaginary][pair][delay]; @param line what line() read */
    Follow(float[][][] scene, Classifier net, double[] line) {
        bgRe = scene[0];
        bgIm = scene[1];
        this.net = net;
        cmPerMs = line[0];
        cmAtZero = line[1];
        detect = line[2];
    }

    /** follow.txt: cm per ms of delay, cm at no delay, the level an echo has to pass in dB. */
    static double[] line(InputStream in) throws IOException {
        try (Scanner s = new Scanner(in).useLocale(Locale.US)) {
            return new double[]{s.nextDouble(), s.nextDouble(), s.nextDouble()};
        }
    }

    /**
     * @param re,im my pings so far: [pair][ping][delay], each measured against its speaker's direct sound
     * @param count how many there are
     * @param pairs per speaker, the indices of its pairs: the one with the mic beside it, then the one at the far end
     */
    Reading read(float[][][] re, float[][][] im, int count, int[][] pairs) {
        int blocks = count / BLOCK;
        Reading r = new Reading();
        r.examples = blocks;
        double[][][] eRe = new double[blocks][pairs.length][LAST], eIm = new double[blocks][pairs.length][LAST];
        double[] sRe = new double[W], sIm = new double[W], mRe = new double[W], mIm = new double[W], v = new double[2], ref = new double[2];
        float[] avRe = new float[W], avIm = new float[W];
        for (int s = 0; s < pairs.length; s++) {
            int own = pairs[s][0], far = pairs[s][1];
            double d = Judge.fit(re[own], im[own], count, bgRe[own], bgIm[own]);
            for (int b = 0; b < blocks; b++) {
                Judge.average(re[own], im[own], b * BLOCK, BLOCK, avRe, avIm);
                Judge.spectrum(avRe, avIm, sRe, sIm);
                Judge.slide(sRe, sIm, d, mRe, mIm);
                Judge.at(mRe, mIm, PRE, ref);
                double mag2 = ref[0] * ref[0] + ref[1] * ref[1];
                Judge.average(re[far], im[far], b * BLOCK, BLOCK, avRe, avIm);
                Judge.spectrum(avRe, avIm, sRe, sIm);
                Judge.slide(sRe, sIm, d, mRe, mIm);
                for (int k = FROM - BEFORE; k < LAST; k++) {
                    Judge.at(mRe, mIm, k, v);
                    eRe[b][s][k] = (v[0] * ref[0] + v[1] * ref[1]) / mag2 - bgRe[far][k];
                    eIm[b][s][k] = (v[1] * ref[0] - v[0] * ref[1]) / mag2 - bgIm[far][k];
                }
            }
        }
        double[] p = new double[TO - FROM];
        for (int s = 0; s < pairs.length; s++) {
            for (int k = FROM; k < TO; k++) {
                double a = 0, b = 0;
                for (int i = 0; i < blocks; i++) {
                    a += eRe[i][s][k] / blocks;
                    b += eIm[i][s][k] / blocks;
                }
                p[k - FROM] += a * a + b * b;
            }
        }
        double ms = edge(p);
        r.cm = cmPerMs * ms + cmAtZero;
        double top = 0;
        r.echo = new float[p.length];
        for (int k = 0; k < p.length; k++) {
            top = Math.max(top, p[k]);
            r.echo[k] = (float) (10 * Math.log10(p[k] + 1e-30));
        }
        r.level = 10 * Math.log10(top + 1e-30);
        r.found = r.level > detect;
        if (!r.found) return r;

        int start = PRE + (int) Math.round(ms * MS) - BEFORE;
        r.p = new float[Classifier.CLASSES.length];
        float[][] x = new float[pairs.length][BEFORE + AFTER];
        for (int b = 0; b < blocks; b++) {
            for (int s = 0; s < pairs.length; s++) {
                for (int k = 0; k < x[s].length; k++) {
                    double a = eRe[b][s][start + k], c = eIm[b][s][start + k];
                    double db = 20 * Math.log10(Math.sqrt(a * a + c * c) + 1e-12);
                    x[s][k] = (float) ((Math.max(-80, Math.min(-20, db)) + 80) / 60);
                }
            }
            float[] one = net.classify(x);
            for (int c = 0; c < one.length; c++) r.p[c] += one[c] / blocks;
        }
        return r;
    }

    /** Where the echo first rises to half of its peak power, walking back from the peak, in ms after the direct sound. */
    static double edge(double[] p) {
        int k = 0;
        for (int i = 1; i < p.length; i++) if (p[i] > p[k]) k = i;
        int i = k;
        while (i > 0 && p[i - 1] >= 0.5 * p[k]) i--;
        double at = i == 0 ? 0 : i - 1 + (0.5 * p[k] - p[i - 1]) / (p[i] - p[i - 1]);
        return (at + FROM - PRE) / MS;
    }
}
