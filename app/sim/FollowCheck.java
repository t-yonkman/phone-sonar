package dev.phonesonar;

import java.io.File;
import java.io.FileInputStream;
import java.nio.file.Files;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Plays the measured sets through the app's own Sonar and Follow, ping by ping as test mode meets them, and sets
 * what comes out with all of a recording's pings beside what analysis/export.py got from it with PyTorch. Also says
 * what it would have shown at the first guess, and how long a reading takes on the PC. The network is the one these
 * sets trained, so this checks that the app's sums are the PC's, not how good the guesses are: see analysis/follow.py.
 *
 * args: folder of sets, the network file, parity_follow.json, follow.txt, a folder to save into
 */
public class FollowCheck {
    private static final int FIRST = 28;

    public static void main(String[] args) throws Exception {
        File sets = new File(args[0]), out = new File(args[4]);
        Classifier net;
        try (FileInputStream in = new FileInputStream(args[1])) {
            net = new JavaNet(in);
        }
        double[] line;
        try (FileInputStream in = new FileInputStream(args[3])) {
            line = Follow.line(in);
        }
        Map<String, Double> pc = new TreeMap<>();
        Matcher m = Pattern.compile("\"(\\w+)\": ([0-9.eE-]+)").matcher(new String(Files.readAllBytes(new File(args[2]).toPath())));
        while (m.find()) pc.put(m.group(1), Double.parseDouble(m.group(2)));

        String[] labels = {"background", "fish", "bobblehead", "empty"};
        Map<String, File[]> found = new TreeMap<>();
        for (File f : sets.listFiles((d, n) -> n.startsWith("set_") && n.endsWith(".wav"))) {
            String[] part = f.getName().replace(".wav", "").split("_");
            for (int i = 0; i < labels.length; i++) {
                if (labels[i].equals(part[part.length - 1])) found.computeIfAbsent(part[1] + "_" + part[2], k -> new File[labels.length])[i] = f;
            }
        }
        double worstP = 0, worstCm = 0, worstDb = 0, spent = 0;
        int reads = 0, recordings = 0, agree = 0, same = 0;
        System.out.println("set              object      app P(fish)  PC P(fish)   app cm   PC cm   app dB   PC dB   at 2.8 s");
        for (Map.Entry<String, File[]> e : found.entrySet()) {
            File[] f = e.getValue();
            if (f[0] == null || !pc.containsKey(e.getKey() + "_fish_p")) continue;
            Sonar.Update bg = Parity.run(f[0], 1, "background", null, null, out);
            if (!bg.ok) {
                System.out.println(e.getKey() + "  background not usable: " + bg.text);
                continue;
            }
            Follow follow = new Follow(bg.scene, net, line);
            for (int o = 1; o < labels.length; o++) {
                if (f[o] == null) continue;
                Follow.Reading[] first = {null};
                float[][][][] q = {null, null};
                int[][][] pairs = {null};
                int[] count = {0};
                Parity.run(f[o], o + 1, "test", bg.scene, null, out, (re, im, n, p) -> {
                    q[0] = re;
                    q[1] = im;
                    pairs[0] = p;
                    count[0] = n;
                    if (n == FIRST) first[0] = follow.read(re, im, n, p);
                });
                long t0 = System.nanoTime();
                Follow.Reading r = follow.read(q[0], q[1], count[0], pairs[0]);
                spent += (System.nanoTime() - t0) / 1e9;
                reads++;
                String key = e.getKey() + "_" + labels[o];
                double p = r.found ? r.p[0] : Double.NaN, refP = pc.get(key + "_p"), refCm = pc.get(key + "_cm"), refDb = pc.get(key + "_db");
                recordings++;
                worstCm = Math.max(worstCm, Math.abs(r.cm - refCm));
                worstDb = Math.max(worstDb, Math.abs(r.level - refDb));
                if (r.found) {
                    worstP = Math.max(worstP, Math.abs(p - refP));
                    if ((p < 0.5) == (refP < 0.5)) agree++;
                }
                String early = first[0] == null ? "-" : first[0].found ? String.format("%s %.0f cm", first[0].p[0] >= 0.5 ? "fish" : "bobblehead", first[0].cm) : "nothing there";
                if (first[0] != null && first[0].found == r.found && (!r.found || (first[0].p[0] >= 0.5) == (r.p[0] >= 0.5))) same++;
                System.out.printf("%s  %-10s  %8.4f     %8.4f   %6.2f  %6.2f   %6.1f  %6.1f   %s%n", e.getKey(), labels[o], p, refP, r.cm, refCm, r.level, refDb, early);
            }
        }
        System.out.printf("%d recordings: largest gap from the PC: P(fish) %.4f, distance %.2f cm, level %.2f dB; %d named as the PC names them; "
                + "%d said the same at 2.8 s as at the end; %.0f ms for a reading of all the pings%n", recordings, worstP, worstCm, worstDb, agree, same, 1000 * spent / Math.max(reads, 1));
    }
}
