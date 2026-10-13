package dev.phonesonar;

import java.io.File;
import java.lang.reflect.Method;
import java.util.Random;

/**
 * Feeds the app's Sonar class two made-up microphone channels on the PC, with both speakers pinging, and
 * checks that it finds each speaker's pings, reports each speaker-to-mic level and arrival, writes the
 * recording and hands back its settings.   args: [folder to save into]
 */
public class RecorderCheck {
    static final int FS = Sonar.FS, CYCLE = Sonar.CYCLE, CHIRP = Sonar.CHIRP;

    static double chirp(int i, boolean up) {   // the played waveform: the bottom speaker sweeps up, the top one down
        double t = (double) i / FS, T = (double) CHIRP / FS, fa = up ? Sonar.F0 : Sonar.F1, fb = up ? Sonar.F1 : Sonar.F0;
        int edge = Math.min(i, CHIRP - 1 - i);
        double env = edge < Sonar.FADE ? 0.5 - 0.5 * Math.cos(Math.PI * edge / Sonar.FADE) : 1;
        return env * Math.cos(2 * Math.PI * (fa * t + (fb - fa) * t * t / (2 * T)));
    }

    public static void main(String[] args) throws Exception {
        int nch = 2, frames = (Sonar.SETTLE + Sonar.LENGTH) * FS;
        // per speaker (bottom, top), per mic (top, bottom): how loud its direct sound is, and how many samples late
        double[][] gain = {{0.02, 0.25}, {0.15, 0.012}};
        int[][] late = {{26, 0}, {0, 24}};
        int[] played = {100, 100 + CYCLE / 2 + 7};   // the top speaker comes out 7 samples later than half way
        float[][] x = new float[nch][frames + CYCLE];
        Random rnd = new Random(1);
        for (int c = 0; c < nch; c++) {
            for (int i = 0; i < frames; i++) x[c][i] = (float) (0.0005 * rnd.nextGaussian());
            for (int sp = 0; sp < 2; sp++) {
                for (int s = played[sp]; s + CHIRP + 300 < frames; s += CYCLE) {
                    for (int i = 0; i < CHIRP; i++) {
                        x[c][s + late[sp][c] + i] += gain[sp][c] * chirp(i, sp == 0);
                        x[c][s + late[sp][c] + 60 + i] += 0.02 * gain[sp][c] * chirp(i, sp == 0);   // a faint echo 1.25 ms later
                    }
                }
            }
        }
        final Sonar.Update[] last = {new Sonar.Update()}, live = {new Sonar.Update()};
        Sonar sonar = new Sonar("check", 1, "background", 0.1f, 11, null, null, new File(args.length > 0 ? args[0] : "."), u -> {
            last[0] = u;
            if (!u.finished) live[0] = u;
        });
        Method configure = Sonar.class.getDeclaredMethod("configure", int.class);
        Method feed = Sonar.class.getDeclaredMethod("feed", short[].class, int.class);
        Method finish = Sonar.class.getDeclaredMethod("finish", String.class);
        for (Method m : new Method[]{configure, feed, finish}) m.setAccessible(true);
        configure.invoke(sonar, nch);
        short[] chunk = new short[1200 * nch];
        for (int f = 0; f + 1200 <= frames; f += 1200) {
            for (int i = 0; i < 1200; i++) for (int c = 0; c < nch; c++) chunk[nch * i + c] = (short) Math.round(32767 * x[c][f + i]);
            feed.invoke(sonar, chunk, 1200);
            if (f == 2 * FS || f == 8 * FS) System.out.println("at " + f / FS + " s: " + last[0].text.replace("\n", " / "));
        }
        finish.invoke(sonar, (String) null);
        System.out.println("finished: " + last[0].text.replace("\n", " / ") + "   ok=" + last[0].ok + "   file=" + last[0].file
                + "   background kept: " + (last[0].scene != null));
        System.out.println("{" + last[0].settings + "}");
        last[0] = live[0];
        System.out.println(last[0].diag);
        for (int i = 0; i < last[0].profile.length; i++) {
            float[] p = last[0].profile[i];
            int peak = 0;
            for (int k = 0; k < p.length; k++) if (p[k] > p[peak]) peak = k;
            System.out.printf("%-24s peak %.1f dB at %+d; 60 samples after it %.1f dB%n", last[0].names[i], p[peak], peak - Sonar.PRE, p[Math.min(peak + 60, p.length - 1)]);
        }
    }
}
