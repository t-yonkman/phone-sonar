package dev.phonesonar;

import java.io.File;
import java.io.FileInputStream;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Plays the recorded sets back through the app's own Sonar and Judge on the PC, exactly as the phone
 * would have met them (background first, then each object judged against it), and sets the answer beside
 * the one analysis/export.py got from the same recordings with PyTorch. The network itself is run here
 * by JavaNet, since the phone's PyTorch runtime cannot run on the PC; the app checks that runtime against
 * PyTorch's answers itself, each time it opens.
 *
 * A saved recording holds only the ten seconds that were kept, so its first three seconds are played
 * once more in front to stand in for the settling time. That is a whole number of ping cycles, so the
 * pings stay in step.
 *
 * args: folder of sets, the network file, parity.json, a folder to save into
 */
public class Parity {
    static final int FS = Sonar.FS;

    static short[] wav(File f) throws Exception {
        byte[] all = Files.readAllBytes(f.toPath());
        ByteBuffer b = ByteBuffer.wrap(all, 44, all.length - 44).order(ByteOrder.LITTLE_ENDIAN);   // the app's own files: a plain 44-byte header
        short[] x = new short[(all.length - 44) / 2];
        b.asShortBuffer().get(x);
        return x;
    }

    static Sonar.Update run(File f, int place, String label, float[][][] scene, Classifier net, File out) throws Exception {
        short[] x = wav(f);
        int nch = 2, frames = x.length / nch, step = 1200;
        final Sonar.Update[] last = {null};
        Sonar sonar = new Sonar("parity", place, label, 0.1f, 11, scene, net, out, u -> {
            if (u.finished) last[0] = u;
        });
        Method configure = Sonar.class.getDeclaredMethod("configure", int.class);
        Method feed = Sonar.class.getDeclaredMethod("feed", short[].class, int.class);
        Method finish = Sonar.class.getDeclaredMethod("finish", String.class);
        for (Method m : new Method[]{configure, feed, finish}) m.setAccessible(true);
        configure.invoke(sonar, nch);
        short[] chunk = new short[step * nch];
        for (int pass = 0; pass < 2; pass++) {
            int end = pass == 0 ? Sonar.SETTLE * FS : frames;
            for (int at = 0; at + step <= end; at += step) {
                System.arraycopy(x, at * nch, chunk, 0, step * nch);
                feed.invoke(sonar, chunk, step);
            }
        }
        finish.invoke(sonar, (String) null);
        new File(out, last[0].file + ".wav").delete();
        return last[0];
    }

    public static void main(String[] args) throws Exception {
        File sets = new File(args[0]), out = new File(args[3]);
        Classifier net;
        try (FileInputStream in = new FileInputStream(args[1])) {
            net = new JavaNet(in);
        }
        Map<String, Double> pc = new TreeMap<>();
        Matcher m = Pattern.compile("\"(\\w+)\": ([0-9.eE-]+)").matcher(new String(Files.readAllBytes(new File(args[2]).toPath())));
        while (m.find()) pc.put(m.group(1), Double.parseDouble(m.group(2)));

        Map<String, File[]> found = new TreeMap<>();
        for (File f : sets.listFiles((d, n) -> n.startsWith("set_") && n.endsWith(".wav"))) {
            String[] part = f.getName().replace(".wav", "").split("_");
            String label = part[part.length - 1];
            int which = label.equals("background") ? 0 : label.equals("fish") ? 1 : label.equals("bobblehead") ? 2 : -1;
            if (which >= 0) found.computeIfAbsent(part[2], k -> new File[3])[which] = f;
        }
        int right = 0, total = 0;
        double worst = 0;
        long spent = 0;
        System.out.println("set     object      app P(fish)  PC P(fish)   app says    slides (samples)   echo dB  examples");
        for (Map.Entry<String, File[]> e : found.entrySet()) {
            File[] f = e.getValue();
            if (f[0] == null) continue;
            Sonar.Update bg = run(f[0], 1, "background", null, net, out);
            if (!bg.ok) {
                System.out.println(e.getKey() + "  background not usable: " + bg.text);
                continue;
            }
            for (int o = 1; o <= 2; o++) {
                if (f[o] == null) continue;
                String object = Classifier.CLASSES[o - 1];
                long t0 = System.nanoTime();
                Sonar.Update u = run(f[o], o + 1, "object", bg.scene, net, out);
                spent += System.nanoTime() - t0;
                if (!u.ok || u.guess == null) {
                    System.out.println(e.getKey() + "  " + object + " not usable: " + u.text);
                    continue;
                }
                Judge.Result g = u.guess;
                Double ref = pc.get(e.getKey() + "_" + object);
                String says = Classifier.CLASSES[g.p[0] >= g.p[1] ? 0 : 1];
                total++;
                if (says.equals(object)) right++;
                if (ref != null) worst = Math.max(worst, Math.abs(ref - g.p[0]));
                System.out.printf("%s  %-10s  %8.4f     %8s     %-10s  %+.2f  %+.2f      %6.1f   %d%n", e.getKey(), object, g.p[0],
                        ref == null ? "-" : String.format("%.4f", ref), says + (says.equals(object) ? "" : " X"), g.slide[0], g.slide[1], g.level, g.examples);
            }
        }
        System.out.printf("%d of %d named correctly from the app's echo extraction; largest gap from the PC's P(fish): %.4f; %.1f s per recording on this PC, playback included%n",
                right, total, worst, spent / 1e9 / Math.max(total, 1));
    }
}
