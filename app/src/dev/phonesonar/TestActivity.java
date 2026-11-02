package dev.phonesonar;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.media.AudioManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.Gravity;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.InputStream;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The app's first screen. Says what is in front, how far away, and how sure, while it pings. Nothing is saved.
 *
 * Start takes an empty scene. After that nobody should touch the phone, so each test then starts itself after a
 * countdown: 3 seconds to get clear, as when collecting, then 10 seconds of pings. The echo shows from the first
 * pings; the first guess comes 2.8 s in, and a new one each 0.4 s after that, from all the pings so far. The result
 * stays on the screen until the next test starts, so tests run one after another until Stop.
 *
 * An empty scene is only good for a couple of minutes: in the blocks it had drifted to about the level of an empty
 * trial's echo after three. So no new test starts once it is older than STALE.
 *
 * For checking the screen over adb the launch intent may carry demo (true): made-up readings, with no sound.
 */
public class TestActivity extends Activity implements Sonar.Listener, Sonar.Watcher {
    private static final int COUNTDOWN = 20;       // seconds to change what is on the mat and step back
    private static final int FIRST = 28;           // pings before the first guess
    private static final long STALE = 120_000;     // ms
    private static final double UNSURE = 0.6;      // below this it says it is not sure
    private static final double NEAR = 9, FAR = 21;   // cm: outside these it is past the distances it was tested at

    private final Handler ui = new Handler(Looper.getMainLooper());
    private final ExecutorService worker = Executors.newSingleThreadExecutor();   // the sums take too long for the sonar's thread
    private final AtomicBoolean reading = new AtomicBoolean();                    // one is under way: skip this turn, the next has more pings anyway
    private Net net;
    private double[] line;
    private Sonar sonar;
    private volatile Follow follow; // what makes the readings of the test under way or just ended
    private float[][][] scene;      // the empty scene, once there is one
    private long sceneAt;           // when it was taken
    private int tests;
    private boolean on;             // a run of tests is under way
    private Runnable tick;          // the next second of the countdown, if one is running
    private volatile float[][][] lastRe, lastIm;   // what the sonar last handed over, for the final reading
    private volatile int[][] lastPairs;
    private volatile int lastCount;
    private Button go, collect;
    private TextView now, answer, sure, where, detail;
    private LiveView live;

    private TextView big(float size, int colour) {
        TextView t = new TextView(this);
        t.setTextSize(size);
        t.setTextColor(colour);
        t.setGravity(Gravity.CENTER);
        return t;
    }

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        StringBuilder ready = new StringBuilder();
        net = Net.load(this, "follow.pte", "follow_selftest.bin", ready);
        try (InputStream in = getAssets().open("follow.txt")) {
            line = Follow.line(in);
        } catch (Exception e) {
            Log.e(Sonar.TAG, "no follow.txt", e);
            net = null;
        }

        int pad = (int) (16 * getResources().getDisplayMetrics().density);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad / 2, pad, pad / 2);
        root.setBackgroundColor(0xff111111);

        LinearLayout top = new LinearLayout(this);
        top.setGravity(Gravity.CENTER_VERTICAL);
        now = big(16, 0xffdddddd);
        now.setGravity(Gravity.START | Gravity.CENTER_VERTICAL);
        go = new Button(this);
        go.setText("Start");
        go.setOnClickListener(v -> {
            if (on) halt("Stopped. Clear the mat, then press Start.");
            else start();
        });
        collect = new Button(this);
        collect.setText("Collect");
        collect.setOnClickListener(v -> startActivity(new Intent(this, BlockActivity.class)));
        top.addView(now, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        top.addView(go);
        top.addView(collect);

        LinearLayout said = new LinearLayout(this);
        said.setOrientation(LinearLayout.VERTICAL);
        said.setGravity(Gravity.CENTER);
        answer = big(46, 0xffffffff);
        answer.setTypeface(Typeface.DEFAULT_BOLD);
        sure = big(30, 0xffffffff);
        where = big(30, 0xffffffff);
        detail = big(13, 0xff999999);
        detail.setTypeface(Typeface.MONOSPACE);
        said.addView(answer);
        said.addView(sure);
        said.addView(where);
        said.addView(detail);

        LinearLayout body = new LinearLayout(this);
        body.addView(said, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 0.9f));
        if (line != null) {
            live = new LiveView(this, line);
            body.addView(live, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1.1f));
        }

        root.addView(top);
        root.addView(body, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);
        now.setText(net == null ? ready + " Test mode can't run." : "Clear the mat, then press Start.");
        go.setEnabled(net != null);

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 1);
        }
        if (getIntent().getBooleanExtra("demo", false)) demo(4);
    }

    private int volume() {
        return ((AudioManager) getSystemService(AUDIO_SERVICE)).getStreamVolume(AudioManager.STREAM_MUSIC);
    }

    private void wipe() {
        answer.setText("");
        sure.setText("");
        where.setText("");
        detail.setText("");
        live.clear();
    }

    private void start() {
        if (sonar != null) return;
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            now.setText("Microphone permission is needed.");
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 1);
            return;
        }
        int v = volume();
        if (v != 10 && v != 11) {
            now.setText("Media volume is " + v + ": set it to 10 or 11, then press Start.");
            return;
        }
        stopCounting();
        scene = null;
        follow = null;
        tests = 0;
        on = true;
        wipe();
        go.setText("Stop");
        collect.setEnabled(false);
        sonar = new Sonar("test", 1, "the empty mat", Sonar.AMP, v, null, null, null, this);
        sonar.start();
    }

    /** The next test, unless the empty scene has got too old. */
    private void next() {
        long age = SystemClock.elapsedRealtime() - sceneAt;
        if (age > STALE) {
            halt("The empty scene has got old. Clear the mat, then press Start for a new one.");
        } else {
            count(COUNTDOWN);
        }
    }

    /** Count down on the screen, then test what is there. */
    private void count(int seconds) {
        if (seconds == 0) {
            test();
            return;
        }
        now.setText(String.format(Locale.US, "%s, then step back. Don't touch the phone. %d s",
                tests == 0 ? "Put an object on the mat, 10 to 20 cm out, or leave it empty" : "Change what's on the mat", seconds));
        tick = () -> count(seconds - 1);
        ui.postDelayed(tick, 1000);
    }

    private void stopCounting() {
        if (tick != null) ui.removeCallbacks(tick);
        tick = null;
    }

    private void test() {
        tick = null;
        lastCount = 0;
        follow = new Follow(scene, net, line);
        wipe();
        answer.setText("Listening");
        sonar = new Sonar("test", 2, "the mat", Sonar.AMP, volume(), scene, null, null, this);
        sonar.watcher = this;
        sonar.start();
    }

    /** Stop whatever is going on and wait for Start. */
    private void halt(String why) {
        stopCounting();
        Sonar s = sonar;
        if (s != null) s.running = false;
        scene = null;
        on = false;
        go.setText("Start");
        go.setEnabled(net != null);
        collect.setEnabled(true);
        now.setText(why);
    }

    @Override
    public void onUpdate(Sonar.Update u) {
        ui.post(() -> {
            if (sonar == null) return;
            if (!u.finished) {
                if (on) now.setText(u.text.replace('\n', ' '));
                return;
            }
            boolean abandoned = !sonar.running;
            sonar = null;
            if (abandoned) return;
            if (!u.ok) {
                halt(u.text + ". Clear the mat, then press Start.");
            } else if (scene == null) {
                scene = u.scene;
                sceneAt = SystemClock.elapsedRealtime();
                next();
            } else {
                tests++;
                last();
                next();
            }
        });
    }

    /** Called on the sonar's thread with each ping kept. */
    @Override
    public void kept(float[][][] re, float[][][] im, int count, int[][] pairs) {
        lastRe = re;
        lastIm = im;
        lastPairs = pairs;
        lastCount = count;
        Follow f = follow;
        if (f == null || count % Follow.BLOCK != 0 || !reading.compareAndSet(false, true)) return;
        worker.execute(() -> {
            Follow.Reading r = f.read(re, im, count, pairs);
            reading.set(false);
            ui.post(() -> {
                if (f == follow) show(r, count, false);
            });
        });
    }

    /** All the test's pings, however the last reading fell. It goes on the queue after any reading under way. */
    private void last() {
        Follow f = follow;
        float[][][] re = lastRe, im = lastIm;
        int[][] pairs = lastPairs;
        int count = lastCount;
        if (f == null || count < FIRST) return;
        worker.execute(() -> {
            Follow.Reading r = f.read(re, im, count, pairs);
            ui.post(() -> {
                if (f == follow) show(r, count, true);
            });
        });
    }

    private void show(Follow.Reading r, int count, boolean last) {
        double seconds = count / 10.0;
        boolean guess = count >= FIRST;
        Log.i(Sonar.TAG, String.format(Locale.US, "test mode reading at %.1f s%s: found %b, %.1f cm, %.1f dB, P(fish) %.3f", seconds, last ? ", final" : "",
                r.found, r.cm, r.level, r.found ? r.p[0] : Float.NaN));
        live.add(r, seconds, guess);
        detail.setText(String.format(Locale.US, "%.1f s of %d%s   echo %.0f dB", seconds, Sonar.LENGTH, last ? ", final" : "", r.level));
        if (!guess) {
            answer.setText("Listening");
            sure.setText(String.format(Locale.US, "first guess in %.0f s", Math.ceil((FIRST - count) / 10.0)));
            where.setText("");
        } else if (!r.found) {
            answer.setText("Nothing there");
            sure.setText("");
            where.setText("");
        } else {
            int c = r.p[0] >= r.p[1] ? 0 : 1;
            answer.setText(Classifier.CLASSES[c].toUpperCase(Locale.US));
            sure.setText(String.format(Locale.US, "%.0f%%%s", 100 * r.p[c], r.p[c] < UNSURE ? ", not sure" : ""));
            where.setText(String.format(Locale.US, "about %.0f cm%s", r.cm, r.cm < NEAR || r.cm > FAR ? ", outside what was tested" : ""));
        }
    }

    /** Made-up readings of a bobblehead at 19 cm, one each 0.4 s, for looking at the screen. */
    private void demo(int count) {
        Follow.Reading r = new Follow.Reading();
        r.echo = new float[Follow.TO - Follow.FROM];
        double top = (21 - line[1]) / line[0] * Follow.MS + Follow.PRE - Follow.FROM;
        for (int k = 0; k < r.echo.length; k++) {
            double x = (k - top) / 7.0;
            r.echo[k] = (float) (-62 + 18 * Math.exp(-x * x) + 1.5 * Math.sin(k * 0.9 + count));
        }
        r.found = true;
        r.cm = 19;
        r.level = -44;
        float fish = 0.34f - 0.21f * count / 100 + 0.03f * (float) Math.sin(count * 0.3);
        r.p = new float[]{fish, 1 - fish};
        show(r, count, count == 100);
        if (count < 100) ui.postDelayed(() -> demo(count + 4), 400);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (on) halt("Stopped. Clear the mat, then press Start.");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        worker.shutdown();
    }
}
