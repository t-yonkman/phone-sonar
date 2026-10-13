package dev.phonesonar;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * The fish-or-bobblehead network written out in plain Java, from the weights analysis/export.py saves as
 * fish_or_bobblehead.bin. On the phone the network is run by PyTorch's own runtime (see Net in the app);
 * that runtime is native phone code and cannot run on the PC, so this stands in for it when the app's
 * Sonar and Judge are checked here. Three convolutions along delay with the speaker-to-mic pairs as
 * channels, each followed by keeping what is positive and the larger of every two neighbours; then the
 * largest value each of the 64 features reaches anywhere along delay; then one weighted sum per class.
 */
final class JavaNet implements Classifier {
    private static final int[] OUT = {16, 32, 64}, TAPS = {7, 5, 3};
    private final int[] IN = {0, 16, 32};   // the first is how many pairs the network is given, read off the file's length
    private final float[][][][] weight = new float[3][][][];   // [layer][out][in][tap]
    private final float[][] bias = new float[3][];
    private final float[][] head = new float[CLASSES.length][OUT[2]];
    private final float[] headBias = new float[CLASSES.length];

    JavaNet(InputStream in) throws IOException {
        ByteArrayOutputStream all = new ByteArrayOutputStream();
        byte[] chunk = new byte[8192];
        for (int got; (got = in.read(chunk)) > 0; ) all.write(chunk, 0, got);
        ByteBuffer b = ByteBuffer.wrap(all.toByteArray()).order(ByteOrder.LITTLE_ENDIAN);
        int numbers = CLASSES.length * OUT[2] + CLASSES.length, perPair = OUT[0] * TAPS[0];
        for (int l = 0; l < 3; l++) numbers += OUT[l] * IN[l] * TAPS[l] + OUT[l];
        IN[0] = (b.remaining() / 4 - 1 - numbers) / perPair;
        numbers += IN[0] * perPair;
        if (IN[0] < 1 || b.remaining() != 4 + 4 * numbers || b.get() != 'F' || b.get() != 'I' || b.get() != 'S' || b.get() != 'H') {
            throw new IOException("not the network this app was built for");
        }
        for (int l = 0; l < 3; l++) {
            weight[l] = new float[OUT[l]][IN[l]][TAPS[l]];
            bias[l] = new float[OUT[l]];
            for (float[][] o : weight[l]) for (float[] i : o) for (int k = 0; k < i.length; k++) i[k] = b.getFloat();
            for (int o = 0; o < OUT[l]; o++) bias[l][o] = b.getFloat();
        }
        for (float[] row : head) for (int i = 0; i < row.length; i++) row[i] = b.getFloat();
        for (int c = 0; c < headBias.length; c++) headBias[c] = b.getFloat();
    }

    /**
     * @param x one example: [pairs][delays], each value 0..1
     * @return how likely each class is, adding up to 1, in the order of CLASSES
     */
    @Override
    public float[] classify(float[][] x) {
        float[][] a = x;
        for (int l = 0; l < 3; l++) {
            int n = a[0].length, half = TAPS[l] / 2;
            float[][] y = new float[OUT[l]][n / 2];
            for (int o = 0; o < OUT[l]; o++) {
                float first = 0;
                for (int t = 0; t < n; t++) {
                    float s = bias[l][o];
                    for (int i = 0; i < IN[l]; i++) {
                        float[] w = weight[l][o][i], v = a[i];
                        for (int k = Math.max(0, half - t), end = Math.min(TAPS[l], n + half - t); k < end; k++) s += w[k] * v[t + k - half];
                    }
                    s = Math.max(s, 0);
                    if (t % 2 == 0) first = s;
                    else y[o][t / 2] = Math.max(first, s);
                }
            }
            a = y;
        }
        double[] score = new double[CLASSES.length];
        double top = Double.NEGATIVE_INFINITY, sum = 0;
        for (int c = 0; c < score.length; c++) {
            score[c] = headBias[c];
            for (int o = 0; o < a.length; o++) {
                float most = a[o][0];
                for (float v : a[o]) most = Math.max(most, v);
                score[c] += head[c][o] * most;
            }
            top = Math.max(top, score[c]);
        }
        float[] p = new float[score.length];
        for (double s : score) sum += Math.exp(s - top);
        for (int c = 0; c < p.length; c++) p[c] = (float) (Math.exp(score[c] - top) / sum);
        return p;
    }
}
