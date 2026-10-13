package dev.phonesonar;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.media.AudioManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Collects recordings in sets of three made in the same surroundings, and has a go at naming what is in
 * front. First the background with nothing there. Then one of the two objects, without being told which:
 * the app guesses, and only then asks which it was. Then the other object: it guesses again, from the
 * recording and the background alone, and is marked against what must have been left. Each recording
 * stops by itself after a fixed time, so nothing is touched while it records.
 *
 * For testing over adb the launch intent may carry: amp (0..1), set (a name for the set, instead of
 * the time), auto (any number above 0 presses Start once).
 */
public class MainActivity extends Activity implements Sonar.Listener {
    private static final String[] STEPS = {"background", "first object", "the other object"};
    private final Handler ui = new Handler(Looper.getMainLooper());
    private Net net;
    private Sonar sonar;
    private String set;             // what the set under way is filed as: the time its background was started
    private float[][][] scene;      // that set's background, which the two objects are judged against
    private int step, done;         // which of the three is next; how many sets have been completed since the app was opened
    private int first = -1;         // which object the first one turned out to be, once told. Never passed to the guessing.
    private int right, guesses;
    private Sonar.Update pending;   // a recording that has been guessed at and is waiting to be told what it was
    private Button go, fresh;
    private Button[] was = new Button[Classifier.CLASSES.length];
    private LinearLayout answer;
    private TextView pos, diag;
    private ScopeView scope;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        String ready = loadNet();

        int pad = (int) (16 * getResources().getDisplayMetrics().density);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);
        root.setBackgroundColor(0xff111111);

        LinearLayout row = new LinearLayout(this);
        go = new Button(this);
        go.setOnClickListener(v -> start());
        fresh = new Button(this);
        fresh.setText("New set");
        fresh.setOnClickListener(v -> {
            step = 0;
            prompt("Set abandoned.");
        });
        row.addView(go);
        row.addView(fresh);

        answer = new LinearLayout(this);
        for (int c = 0; c < was.length; c++) {
            int which = c;
            was[c] = new Button(this);
            was[c].setText("It was the " + Classifier.CLASSES[c]);
            was[c].setOnClickListener(v -> told(which));
            answer.addView(was[c]);
        }
        answer.setVisibility(View.GONE);

        pos = new TextView(this);
        pos.setTextSize(20);
        pos.setTextColor(0xffdddddd);
        scope = new ScopeView(this);
        diag = new TextView(this);
        diag.setTypeface(Typeface.MONOSPACE);
        diag.setTextSize(11);
        diag.setTextColor(0xff999999);

        root.addView(row);
        root.addView(answer);
        root.addView(pos);
        root.addView(scope, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        root.addView(diag);
        setContentView(root);
        prompt(ready + (net == null ? " Recording only." : " Phone on its stand, media volume 10 or 11."));

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 1);
        } else if (getIntent().getIntExtra("auto", 0) > 0) {
            start();
        }
    }

    /**
     * Hand the exported PyTorch model to the ExecuTorch runtime, and have it redo the examples that were
     * saved with PyTorch's own answers when the model was exported. The guessing is only switched on if
     * the phone gets the same answers as the PC did.
     *
     * @return a line for the screen saying how that went
     */
    private String loadNet() {
        String name = "fish_or_bobblehead.pte";
        try {
            File program = new File(getFilesDir(), name);   // the runtime wants a file, and assets are not files
            try (InputStream in = getAssets().open(name); FileOutputStream out = new FileOutputStream(program)) {
                byte[] chunk = new byte[8192];
                for (int got; (got = in.read(chunk)) > 0; ) out.write(chunk, 0, got);
            }
            Net loaded = new Net(program);
            double[] check;
            try (InputStream in = getAssets().open("selftest.bin")) {
                check = loaded.check(in);
            }
            Log.i(Sonar.TAG, String.format(Locale.US, "PyTorch model check: %.0f examples, largest difference from the PC %.6f; runtime backends: %s",
                    check[0], check[1], Net.backends()));
            if (check[1] > 0.01) return String.format(Locale.US, "PyTorch model check FAILED: differs from the PC by %.3f.", check[1]);
            net = loaded;
            return String.format(Locale.US, "PyTorch model check: %.0f of %.0f examples match the PC.", check[0], check[0]);
        } catch (Throwable e) {   // a missing native library is an Error, not an Exception
            Log.e(Sonar.TAG, "no network to guess with", e);
            return "The PyTorch model could not be loaded.";
        }
    }

    /** Say what just happened and what the next recording is. */
    private void prompt(String news) {
        String[] put = {"Leave nothing in front", "Put the fish or the bobblehead 10 cm in front, either one",
                "Swap it for the " + (first < 0 ? "other one" : Classifier.CLASSES[1 - first])};
        go.setText("Start: " + STEPS[step]);
        go.setEnabled(true);
        fresh.setEnabled(step > 0);
        answer.setVisibility(View.GONE);
        pos.setText(String.format(Locale.US, "%s\nSet %d, recording %d of 3. %s, press Start, hands away. It stops by itself after %d s.%s",
                news, done + 1, step + 1, put[step], Sonar.SETTLE + Sonar.LENGTH,
                guesses > 0 ? String.format(Locale.US, "\nGuesses right so far: %d of %d.", right, guesses) : ""));
    }

    private void start() {
        if (sonar != null || pending != null) return;
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            pos.setText("Microphone permission is needed.");
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 1);
            return;
        }
        if (step == 0) {
            String named = getIntent().getStringExtra("set");
            set = named != null ? named : new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            first = -1;
        }
        AudioManager audio = (AudioManager) getSystemService(AUDIO_SERVICE);
        // With media volume at 11 of 15, 0.1 puts the ping at a quarter of the bottom mic's range. The mic's
        // level stops rising in step with the ping above that (0.2 gives 0.41, not 0.5) and clips by 0.4.
        // The guessing is given this set's background and the network, and nothing about what came before.
        sonar = new Sonar(set, step + 1, STEPS[step], getIntent().getFloatExtra("amp", 0.1f),
                audio.getStreamVolume(AudioManager.STREAM_MUSIC), step == 0 ? null : scene, net, getExternalFilesDir(null), this);
        sonar.start();
        go.setEnabled(false);
        fresh.setEnabled(false);
    }

    @Override
    public void onUpdate(Sonar.Update u) {
        ui.post(() -> {
            if (!u.finished) {
                pos.setText(u.text);
                if (!u.diag.isEmpty()) diag.setText(u.diag);
                scope.set(u);
                return;
            }
            sonar = null;
            if (!u.ok) {
                prompt(u.text + ". This one will be repeated.");
            } else if (step == 0) {
                scene = u.scene;
                file(u, "background", "");
                step = 1;
                prompt("Saved the background.");
            } else if (u.guess == null) {   // no network: just ask
                pending = u;
                ask("Recorded.");
            } else if (step == 1) {
                pending = u;
                ask("My guess: " + guess(u.guess) + ".");
            } else {
                marked(u, 1 - first);
            }
        });
    }

    private static int pick(Judge.Result g) {
        return g.p[0] >= g.p[1] ? 0 : 1;
    }

    private static String guess(Judge.Result g) {
        int c = pick(g);
        return String.format(Locale.US, "%s%s (%.0f%%)", g.p[c] < 0.6 ? "not sure, leaning " : "the ", Classifier.CLASSES[c].toUpperCase(Locale.US), 100 * g.p[c]);
    }

    /** Show the guess, if any, and wait to be told which object it was. */
    private void ask(String news) {
        go.setEnabled(false);
        fresh.setEnabled(false);
        answer.setVisibility(View.VISIBLE);
        for (int c = 0; c < was.length; c++) was[c].setEnabled(step == 1 || c == 1 - first);
        pos.setText(news + "\nWhich was it?");
    }

    private void told(int which) {
        Sonar.Update u = pending;
        if (u == null) return;
        pending = null;
        if (step == 1) first = which;
        marked(u, which);
    }

    /** Mark a guess against what the object really was, file the recording under that, and move on. */
    private void marked(Sonar.Update u, int truth) {
        String news = "Recorded the " + Classifier.CLASSES[truth] + ".", extra = "";
        if (u.guess != null) {
            boolean correct = pick(u.guess) == truth;
            guesses++;
            if (correct) right++;
            news = String.format(Locale.US, "My guess: %s. It was the %s: %s.", guess(u.guess), Classifier.CLASSES[truth], correct ? "RIGHT" : "WRONG");
            extra = String.format(Locale.US, "\"guess\": \"%s\", \"p_fish\": %.4f, \"guess_right\": %b, \"echo_level_db\": %.1f, \"slides\": [%.2f, %.2f], ",
                    Classifier.CLASSES[pick(u.guess)], u.guess.p[0], correct, u.guess.level, u.guess.slide[0], u.guess.slide[1]);
        }
        file(u, Classifier.CLASSES[truth], extra);
        step = (step + 1) % STEPS.length;
        if (step == 0) done++;
        prompt(news + (step == 0 ? " Set complete: move somewhere else for the next one." : ""));
    }

    /** Give the recording its final name, now that what was in front is known, and write its settings beside it. */
    private void file(Sonar.Update u, String label, String extra) {
        File dir = getExternalFilesDir(null);
        String name = u.file + "_" + label;
        if (!new File(dir, u.file + ".wav").renameTo(new File(dir, name + ".wav"))) Log.e(Sonar.TAG, "could not rename " + u.file);
        try (PrintWriter pw = new PrintWriter(new File(dir, name + ".json"))) {
            pw.println("{\"label\": \"" + label + "\", \"ok\": true, " + extra + u.settings + "}");
        } catch (Exception e) {
            Log.e(Sonar.TAG, "could not write settings for " + name, e);
        }
        Log.i(Sonar.TAG, "filed " + name + (extra.isEmpty() ? "" : "  " + extra));
    }

    @Override
    protected void onPause() {
        super.onPause();
        Sonar s = sonar;
        if (s != null) s.running = false;   // leaving the app abandons the recording under way; it reports back and is repeated
    }
}
