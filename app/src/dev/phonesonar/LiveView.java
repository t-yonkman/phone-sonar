package dev.phonesonar;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.DashPathEffect;
import android.graphics.Paint;
import android.graphics.Path;
import android.view.View;

/**
 * What test mode is hearing, as it comes. Above, the echo laid out along the mat, with a line where the distance
 * was read and the level an echo has to pass. Below, how the guess has moved over the seconds of the test.
 */
final class LiveView extends View {
    private static final float NEAR = 8, FAR = 22;         // cm shown along the mat
    private static final float QUIET = -68, LOUD = -28;    // dB shown
    private static final int SECONDS = 10;
    private static final int AXIS = 0xff3a3a3a, TEXT = 0xff999999, CURVE = 0xffe8e8e8, MARK = 0xffffffff;
    private static final int FISH = 0xff5b9be8, BOBBLEHEAD = 0xffeb6834;

    private final double[] line;      // as Follow.line reads it
    private final float dp, sp;
    private final Paint axis = new Paint(Paint.ANTI_ALIAS_FLAG), text = new Paint(Paint.ANTI_ALIAS_FLAG), curve = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Path path = new Path();
    private final float[] when = new float[SECONDS * 3], fish = new float[SECONDS * 3];
    private int guesses;
    private float[] echo;
    private double mark = Double.NaN;

    LiveView(Context context, double[] line) {
        super(context);
        this.line = line;
        dp = context.getResources().getDisplayMetrics().density;
        sp = context.getResources().getDisplayMetrics().scaledDensity;
        axis.setStrokeWidth(dp);
        axis.setColor(AXIS);
        text.setColor(TEXT);
        text.setTextSize(12 * sp);
        curve.setStyle(Paint.Style.STROKE);
        curve.setStrokeWidth(2 * dp);
        curve.setStrokeJoin(Paint.Join.ROUND);
    }

    void clear() {
        guesses = 0;
        echo = null;
        mark = Double.NaN;
        invalidate();
    }

    /** @param guess whether this reading's name counts: before the first guess only the echo is shown */
    void add(Follow.Reading r, double seconds, boolean guess) {
        echo = r.echo;
        mark = r.found ? r.cm : Double.NaN;
        if (guess && r.found && guesses < when.length) {
            when[guesses] = (float) seconds;
            fish[guesses++] = r.p[0];
        }
        invalidate();
    }

    private float cm(int k) {
        return (float) (line[0] * (Follow.FROM + k - Follow.PRE) / Follow.MS + line[1]);
    }

    @Override
    protected void onDraw(Canvas c) {
        float left = 8 * dp, right = getWidth() - 8 * dp, gap = 34 * dp, row = 20 * sp;
        float top = row, bottom = getHeight() * 0.6f - row;
        drawEcho(c, left, top, right, bottom);
        drawGuess(c, left, bottom + gap + row * 0.4f, right, getHeight() - row);
    }

    private void drawEcho(Canvas c, float l, float t, float r, float b) {
        text.setTextAlign(Paint.Align.LEFT);
        c.drawText("Echo along the mat", l, t - 6 * dp, text);
        text.setTextAlign(Paint.Align.RIGHT);
        c.drawText("below the dashes: too quiet to be an object", r, t - 6 * dp, text);
        float perCm = (r - l) / (FAR - NEAR), perDb = (b - t) / (LOUD - QUIET);
        text.setTextAlign(Paint.Align.CENTER);
        for (int x = 10; x <= 20; x += 2) {
            float at = l + (x - NEAR) * perCm;
            c.drawLine(at, t, at, b, axis);
            c.drawText(x + (x == 20 ? " cm" : ""), at, b + 16 * sp, text);
        }
        c.drawLine(l, b, r, b, axis);
        float level = b - ((float) line[2] - QUIET) * perDb;
        axis.setPathEffect(new DashPathEffect(new float[]{6 * dp, 4 * dp}, 0));
        c.drawLine(l, level, r, level, axis);
        axis.setPathEffect(null);
        if (echo == null) return;
        path.reset();
        boolean going = false;
        for (int k = 0; k < echo.length; k++) {
            float x = cm(k);
            if (x < NEAR || x > FAR) continue;
            float px = l + (x - NEAR) * perCm, py = b - (Math.max(QUIET, Math.min(LOUD, echo[k])) - QUIET) * perDb;
            if (going) path.lineTo(px, py);
            else path.moveTo(px, py);
            going = true;
        }
        curve.setColor(CURVE);
        c.drawPath(path, curve);
        if (!Double.isNaN(mark) && mark >= NEAR && mark <= FAR) {
            float at = l + ((float) mark - NEAR) * perCm;
            curve.setColor(MARK);
            c.drawLine(at, t, at, b, curve);
        }
    }

    private void drawGuess(Canvas c, float l, float t, float r, float b) {
        float perSecond = (r - l) / SECONDS;
        text.setTextAlign(Paint.Align.LEFT);
        c.drawText("fish", l, t - 6 * dp, text);
        c.drawLine(l, t, r, t, axis);
        c.drawLine(l, b, r, b, axis);
        axis.setPathEffect(new DashPathEffect(new float[]{6 * dp, 4 * dp}, 0));
        c.drawLine(l, (t + b) / 2, r, (t + b) / 2, axis);
        axis.setPathEffect(null);
        text.setTextAlign(Paint.Align.CENTER);
        for (int x = 2; x < SECONDS; x += 2) c.drawText(String.valueOf(x), l + x * perSecond, b + 16 * sp, text);
        text.setTextAlign(Paint.Align.RIGHT);
        c.drawText(SECONDS + " s", r, b + 16 * sp, text);
        text.setTextAlign(Paint.Align.LEFT);
        c.drawText("bobblehead", l, b + 16 * sp, text);
        if (guesses == 0) return;
        path.reset();
        for (int i = 0; i < guesses; i++) {
            float px = l + when[i] * perSecond, py = b - fish[i] * (b - t);
            if (i == 0) path.moveTo(px, py);
            else path.lineTo(px, py);
        }
        boolean isFish = fish[guesses - 1] >= 0.5f;
        curve.setColor(isFish ? FISH : BOBBLEHEAD);
        c.drawPath(path, curve);
        curve.setStyle(Paint.Style.FILL);
        c.drawCircle(l + when[guesses - 1] * perSecond, b - fish[guesses - 1] * (b - t), 5 * dp, curve);
        curve.setStyle(Paint.Style.STROKE);
    }
}
