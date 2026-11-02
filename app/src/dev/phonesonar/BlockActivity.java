package dev.phonesonar;

import android.Manifest;
import android.app.Activity;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.media.AudioManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.KeyEvent;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Records a block of sets to a RunSheet. The app shows what to put where, so nothing is labelled by hand;
 * one mistaken tap in the older flow could swap both labels of a set. Nobody should touch the phone between
 * the empty scene and the objects, so only the empty scene is started by hand. Each recording after it starts
 * itself after a countdown, or from a Bluetooth shutter or adb (any of the keys in remote()). A photo is taken
 * as each recording ends.
 *
 * The app still guesses at each recording before it could know, and files the guess, but says nothing of it
 * until the set is over. Block D is the sealed one: its guesses are filed and never shown.
 *
 * How far a block has got is kept, so closing the app loses at most the set under way.
 */
public class BlockActivity extends Activity implements Sonar.Listener {
    private static final String PRACTICE = "practice", SEALED = "D";
    private static final String[] BLOCKS = {"A", "B", "C", SEALED, PRACTICE};
    private static final int PRACTICE_SETS = 3;
    private static final int COUNTDOWN = 20;   // seconds to change what is on the mat and step back

    private final Handler ui = new Handler(Looper.getMainLooper());
    private final List<String> filed = new ArrayList<>();          // the recordings of the set under way, or the last one
    private final StringBuilder guesses = new StringBuilder();     // what was made of them, shown when the set is over
    private SharedPreferences progress;
    private File dir;
    private Net net;
    private Photo photo;
    private String block, set;
    private long seed;
    private RunSheet sheet;
    private int at, trial;          // which set of the sheet is next or under way; which of its recordings
    private float[][][] scene;      // this set's empty scene, which the rest of it is judged against
    private Sonar sonar;
    private boolean busy;           // a recording has ended and its photo is being taken
    private String step = "";       // what the screen is asking for
    private Runnable tick;          // the next second of the countdown, if one is running
    private Button go;
    private TextView now, log, diag;
    private ScopeView scope;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        progress = getSharedPreferences("blocks", MODE_PRIVATE);
        dir = getExternalFilesDir(null);
        StringBuilder ready = new StringBuilder();
        net = Net.load(this, ready);
        photo = new Photo(this);

        int pad = (int) (16 * getResources().getDisplayMetrics().density);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);
        root.setBackgroundColor(0xff111111);

        LinearLayout blocks = new LinearLayout(this);
        for (String name : BLOCKS) {
            Button b = new Button(this);
            b.setText(name.equals(PRACTICE) ? "Practice" : "Block " + name);
            b.setOnClickListener(v -> begin(name));
            blocks.addView(b);
        }
        LinearLayout row = new LinearLayout(this);
        go = new Button(this);
        go.setText("Start");
        go.setEnabled(false);
        go.setOnClickListener(v -> start());
        Button hold = new Button(this);
        hold.setText("Hold");
        hold.setOnClickListener(v -> hold());
        Button restart = new Button(this);
        restart.setText("Restart set");
        restart.setOnClickListener(v -> restart());
        Button flag = new Button(this);
        flag.setText("Flag set");
        flag.setOnClickListener(v -> log.setText(flag() + " recordings flagged."));
        row.addView(go);
        row.addView(hold);
        row.addView(restart);
        row.addView(flag);

        now = new TextView(this);
        now.setTextSize(30);
        now.setTextColor(0xffffffff);
        log = new TextView(this);
        log.setTextSize(14);
        log.setTextColor(0xff999999);
        scope = new ScopeView(this);
        diag = new TextView(this);
        diag.setTypeface(Typeface.MONOSPACE);
        diag.setTextSize(11);
        diag.setTextColor(0xff999999);

        root.addView(blocks);
        root.addView(row);
        root.addView(now);
        root.addView(log);
        root.addView(scope, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        root.addView(diag);
        setContentView(root);
        now.setText("Pick a block. A block already begun carries on where it stopped.");
        log.setText(ready);

        List<String> missing = new ArrayList<>();
        for (String p : new String[]{Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA}) {
            if (checkSelfPermission(p) != PackageManager.PERMISSION_GRANTED) missing.add(p);
        }
        if (!missing.isEmpty()) requestPermissions(missing.toArray(new String[0]), 1);
    }

    private void begin(String name) {
        if (sonar != null || busy) return;
        stopCounting();
        block = name;
        if (name.equals(PRACTICE)) progress.edit().remove(name + ".seed").remove(name + ".at").apply();
        seed = progress.getLong(name + ".seed", 0);
        if (seed == 0) {
            seed = System.nanoTime();
            progress.edit().putLong(name + ".seed", seed).apply();
        }
        sheet = new RunSheet(seed, name.equals(PRACTICE) ? PRACTICE_SETS : RunSheet.FULL);
        at = progress.getInt(name + ".at", 0);
        trial = 0;
        filed.clear();
        guesses.setLength(0);
        log.setText("");
        prompt();
    }

    private static String side(int sideways) {
        return sideways == 0 ? "" : ", " + Math.abs(sideways) + " towards the " + (sideways > 0 ? "bottom" : "top") + " end";
    }

    /** Show what to do next. */
    private void prompt() {
        if (at == sheet.sets.size()) {
            go.setEnabled(false);
            now.setText("Block " + block + " is finished.");
            return;
        }
        RunSheet.Plan p = sheet.sets.get(at);
        String label = p.trials.get(trial), what;
        if (label.equals(RunSheet.BACKGROUND)) {
            what = "Clear the mat, then press Start.";
        } else if (label.equals(RunSheet.EMPTY)) {
            what = "Nothing this time. Reach in as if placing something, then step back.";
        } else {
            what = "The " + label + " at " + p.distance + " cm" + side(p.sideways) + ".";
        }
        int volume = ((AudioManager) getSystemService(AUDIO_SERVICE)).getStreamVolume(AudioManager.STREAM_MUSIC);
        step = String.format(Locale.US, "Block %s, set %d of %d, recording %d of %d%s\n%s", block, at + 1, sheet.sets.size(), trial + 1, p.trials.size(),
                volume == 10 || volume == 11 ? "" : ". Media volume is " + volume + ": it should be 10 or 11", what);
        now.setText(step);
        go.setEnabled(true);
        if (trial > 0) count(COUNTDOWN);
    }

    /** Count down on the screen, then start the recording. */
    private void count(int seconds) {
        if (seconds == 0) {
            start();
            return;
        }
        now.setText(step + "\nStarting in " + seconds + " s");
        tick = () -> count(seconds - 1);
        ui.postDelayed(tick, 1000);
    }

    private void stopCounting() {
        if (tick != null) ui.removeCallbacks(tick);
        tick = null;
    }

    /** Stop the countdown, for when the object isn't placed in time. Start carries on. */
    private void hold() {
        if (tick == null) return;
        stopCounting();
        now.setText(step + "\nHeld. Press Start when you are clear.");
    }

    private void start() {
        if (sheet == null || at == sheet.sets.size() || sonar != null || busy) return;
        stopCounting();
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            now.setText("Microphone permission is needed.");
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 1);
            return;
        }
        if (trial == 0) {
            set = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            filed.clear();
            guesses.setLength(0);
            log.setText("");
        }
        String label = sheet.sets.get(at).trials.get(trial);
        int volume = ((AudioManager) getSystemService(AUDIO_SERVICE)).getStreamVolume(AudioManager.STREAM_MUSIC);
        sonar = new Sonar(set, trial + 1, label, Sonar.AMP, volume, trial == 0 ? null : scene, net, dir, this);
        sonar.start();
        go.setEnabled(false);
    }

    @Override
    public void onUpdate(Sonar.Update u) {
        ui.post(() -> {
            if (!u.finished) {
                now.setText(u.text);
                if (!u.diag.isEmpty()) diag.setText(u.diag);
                scope.set(u);
                return;
            }
            sonar = null;
            if (!u.ok) {
                log.setText(u.text + ". This one will be repeated.");
                prompt();
                return;
            }
            busy = true;
            String label = sheet.sets.get(at).trials.get(trial);
            photo.take(new File(dir, u.file + "_" + label + ".jpg"), ok -> filed(u, label, ok));
        });
    }

    /** File a finished recording under what the run sheet says it was, and move on. */
    private void filed(Sonar.Update u, String label, boolean photographed) {
        busy = false;
        RunSheet.Plan p = sheet.sets.get(at);
        StringBuilder extra = new StringBuilder(String.format(Locale.US, "\"block\": \"%s\", \"run_seed\": %d, \"set_in_block\": %d, \"photo\": %b, ",
                block, seed, at + 1, photographed));
        int truth = Arrays.asList(Classifier.CLASSES).indexOf(label);
        if (truth >= 0) extra.append(String.format(Locale.US, "\"distance_cm\": %d, \"sideways_cm\": %d, ", p.distance, p.sideways));
        if (u.guess != null) {
            extra.append(guessFields(u.guess, truth));
            guesses.append(String.format(Locale.US, "%s: P(fish) %.2f   ", label, u.guess.p[0]));
        }
        file(dir, u, label, extra.toString());
        filed.add(u.file + "_" + label);
        if (label.equals(RunSheet.BACKGROUND)) scene = u.scene;
        trial++;
        if (trial == p.trials.size()) {
            trial = 0;
            at++;
            if (!block.equals(PRACTICE)) progress.edit().putInt(block + ".at", at).apply();
            log.setText(block.equals(SEALED) ? "" : guesses.toString());
        }
        prompt();
    }

    /** What the guessing made of a recording, as part of a JSON object. @param truth the class it really was, or -1 if nothing was there */
    private static String guessFields(Judge.Result g, int truth) {
        int pick = g.p[0] >= g.p[1] ? 0 : 1;
        return String.format(Locale.US, "\"guess\": \"%s\", \"p_fish\": %.4f, %s\"echo_level_db\": %.1f, \"slides\": [%.2f, %.2f], ",
                Classifier.CLASSES[pick], g.p[0], truth < 0 ? "" : "\"guess_right\": " + (pick == truth) + ", ", g.level, g.slide[0], g.slide[1]);
    }

    /** Give the recording its final name, now that what was in front is known, and write its settings beside it. */
    private static void file(File dir, Sonar.Update u, String label, String extra) {
        String name = u.file + "_" + label;
        if (!new File(dir, u.file + ".wav").renameTo(new File(dir, name + ".wav"))) Log.e(Sonar.TAG, "could not rename " + u.file);
        try (PrintWriter pw = new PrintWriter(new File(dir, name + ".json"))) {
            pw.println("{\"label\": \"" + label + "\", \"ok\": true, " + extra + u.settings + "}");
        } catch (Exception e) {
            Log.e(Sonar.TAG, "could not write settings for " + name, e);
        }
        Log.i(Sonar.TAG, "filed " + name + (extra.isEmpty() ? "" : "  " + extra));
    }

    /** Give up the set under way: the recordings so far are flagged and it starts again from its empty scene. */
    private void restart() {
        if (sheet == null || sonar != null || busy || trial == 0) return;
        stopCounting();
        log.setText("Set restarted; " + flag() + " recordings flagged.");
        trial = 0;
        prompt();
    }

    /** Mark the recordings of this set, or of the last one, as having gone wrong. They are kept. */
    private int flag() {
        int done = 0;
        for (String name : filed) {
            File f = new File(dir, name + ".json");
            try {
                String json = new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8);
                if (!json.contains("\"flagged\"")) {
                    Files.write(f.toPath(), json.replaceFirst("\\{", "{\"flagged\": true, ").getBytes(StandardCharsets.UTF_8));
                    done++;
                }
            } catch (IOException e) {
                Log.e(Sonar.TAG, "could not flag " + name, e);
            }
        }
        return done;
    }

    /** A Bluetooth shutter sends a volume or enter key, and adb can send any of these. None may change the volume. */
    private static boolean remote(int key) {
        return key == KeyEvent.KEYCODE_VOLUME_UP || key == KeyEvent.KEYCODE_VOLUME_DOWN || key == KeyEvent.KEYCODE_ENTER
                || key == KeyEvent.KEYCODE_CAMERA || key == KeyEvent.KEYCODE_MEDIA_PLAY_PAUSE || key == KeyEvent.KEYCODE_HEADSETHOOK;
    }

    @Override
    public boolean onKeyDown(int key, KeyEvent event) {
        Log.i(Sonar.TAG, "key " + KeyEvent.keyCodeToString(key));
        if (sheet == null) log.setText(KeyEvent.keyCodeToString(key) + (remote(key) ? ": the app will start on this key." : ": the app does not use this key."));
        if (!remote(key)) return super.onKeyDown(key, event);
        if (event.getRepeatCount() == 0) start();
        return true;
    }

    @Override
    public boolean onKeyUp(int key, KeyEvent event) {
        return remote(key) || super.onKeyUp(key, event);
    }

    @Override
    protected void onPause() {
        super.onPause();
        hold();
        Sonar s = sonar;
        if (s != null) s.running = false;   // leaving the app abandons the recording under way; it reports back and is repeated
    }
}
