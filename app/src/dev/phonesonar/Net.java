package dev.phonesonar;

import org.pytorch.executorch.EValue;
import org.pytorch.executorch.ExecuTorchRuntime;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * The fish-or-bobblehead network. It is the PyTorch model trained on the PC (analysis/classify.py),
 * exported by analysis/export.py as an ExecuTorch program, and run here by ExecuTorch, PyTorch's
 * runtime for phones. Nothing of the network is written out in this app: the program file holds its
 * layers and weights, ending in a softmax, so what comes back is how likely each class is.
 */
final class Net implements Classifier {
    private final Module module;

    /** @param program the exported model (.pte). The runtime reads it from a file, not from inside the APK. */
    Net(File program) {
        module = Module.load(program.getAbsolutePath());
    }

    @Override
    public float[] classify(float[][] x) {
        int pairs = x.length, delays = x[0].length;
        float[] flat = new float[pairs * delays];
        for (int p = 0; p < pairs; p++) System.arraycopy(x[p], 0, flat, p * delays, delays);
        Tensor in = Tensor.fromBlob(flat, new long[]{1, pairs, delays});
        return module.forward(EValue.from(in))[0].toTensor().getDataAsFloatArray();
    }

    /**
     * Run the examples analysis/export.py saved alongside the program, and compare with what PyTorch made
     * of the same examples on the PC.
     *
     * @return how many examples there were, and the largest difference in P(fish) from the PC's answer
     */
    double[] check(InputStream examples) throws IOException {
        ByteArrayOutputStream all = new ByteArrayOutputStream();
        byte[] chunk = new byte[8192];
        for (int got; (got = examples.read(chunk)) > 0; ) all.write(chunk, 0, got);
        ByteBuffer b = ByteBuffer.wrap(all.toByteArray()).order(ByteOrder.LITTLE_ENDIAN);
        if (b.get() != 'T' || b.get() != 'E' || b.get() != 'S' || b.get() != 'T') throw new IOException("not a file of test examples");
        int n = b.getInt(), pairs = b.getInt(), delays = b.getInt();
        double worst = 0;
        float[][] x = new float[pairs][delays];
        for (int i = 0; i < n; i++) {
            for (float[] pair : x) for (int k = 0; k < delays; k++) pair[k] = b.getFloat();
            worst = Math.max(worst, Math.abs(classify(x)[0] - b.getFloat()));
        }
        return new double[]{n, worst};
    }

    /** What the runtime is using to do the sums, for the log. */
    static String backends() {
        return String.join(", ", ExecuTorchRuntime.getRegisteredBackends());
    }
}
