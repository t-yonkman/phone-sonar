package dev.phonesonar;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.view.View;

/**
 * What the latest ping from each speaker sounded like at each mic: loudness against time after its
 * direct sound, one line per speaker and mic. Nothing is worked out from it here; it is for seeing at a
 * glance that every pair hears its pings and that they hear them differently.
 */
final class ScopeView extends View {
    private static final int[] COLOURS = {0xff3987e5, 0xffd95926, 0xff199e70, 0xffc98500};   // one per line, in the order given
    private static final float MS0 = -0.5f, MS1 = 4f, DB0 = -80, DB1 = 0;                    // the ranges drawn
    private final Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Path line = new Path();
    private Sonar.Update u = new Sonar.Update();

    ScopeView(Context c) {
        super(c);
    }

    void set(Sonar.Update u) {
        if (u.profile.length == 0 && this.u.profile.length > 0) return;   // keep the last picture when a message has none
        this.u = u;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas g) {
        float dp = getResources().getDisplayMetrics().density;
        float x0 = 34 * dp, x1 = getWidth() - 6 * dp, y0 = 8 * dp, y1 = getHeight() - 22 * dp;
        p.setTextSize(11 * dp);
        p.setStrokeWidth(dp);
        for (int db = (int) DB1; db >= DB0; db -= 20) {   // loudness grid
            float y = y0 + (DB1 - db) / (DB1 - DB0) * (y1 - y0);
            p.setStyle(Paint.Style.STROKE);
            p.setColor(0xff2c2c2a);
            g.drawLine(x0, y, x1, y, p);
            p.setStyle(Paint.Style.FILL);
            p.setColor(0xff898781);
            g.drawText(String.valueOf(db), 4 * dp, y + 4 * dp, p);
        }
        for (int ms = 0; ms <= MS1; ms++) {               // time grid
            float x = x0 + (ms - MS0) / (MS1 - MS0) * (x1 - x0);
            p.setStyle(Paint.Style.STROKE);
            p.setColor(0xff2c2c2a);
            g.drawLine(x, y0, x, y1, p);
            p.setStyle(Paint.Style.FILL);
            p.setColor(0xff898781);
            g.drawText(ms + " ms", x - 10 * dp, y1 + 15 * dp, p);
        }
        g.drawText("dB below the direct sound, after each ping", x0 + 6 * dp, y0 + 12 * dp, p);

        g.save();
        g.clipRect(x0, y0, x1, y1);
        for (int c = 0; c < u.profile.length; c++) {
            float[] v = u.profile[c];
            line.rewind();
            for (int k = 0; k < v.length; k++) {
                float ms = (k - Sonar.PRE) * 1000f / Sonar.FS;
                if (ms < MS0 || ms > MS1) continue;
                float x = x0 + (ms - MS0) / (MS1 - MS0) * (x1 - x0);
                float y = y0 + (DB1 - Math.max(DB0, v[k])) / (DB1 - DB0) * (y1 - y0);
                if (line.isEmpty()) line.moveTo(x, y);
                else line.lineTo(x, y);
            }
            p.setStyle(Paint.Style.STROKE);
            p.setStrokeWidth(2 * dp);
            p.setColor(COLOURS[c % COLOURS.length]);
            g.drawPath(line, p);
        }
        g.restore();

        // which line is which speaker and mic
        p.setStrokeWidth(2 * dp);
        for (int c = 0; c < u.profile.length; c++) {
            float y = y0 + (30 + 16 * c) * dp, x = x1 - 190 * dp;
            p.setStyle(Paint.Style.STROKE);
            p.setColor(COLOURS[c % COLOURS.length]);
            g.drawLine(x, y, x + 18 * dp, y, p);
            p.setStyle(Paint.Style.FILL);
            p.setColor(0xffc3c2b7);
            g.drawText(u.names[c], x + 24 * dp, y + 4 * dp, p);
        }
    }
}
