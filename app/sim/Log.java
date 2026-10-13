package android.util;

/** Desktop stand-in so the app's Sonar class can run outside Android. */
public final class Log {
    public static boolean echo = false;

    public static int i(String tag, String msg) {
        if (echo && (msg.startsWith("acquired") || msg.startsWith("started"))) System.out.println("      " + msg);
        return 0;
    }

    public static int e(String tag, String msg, Throwable t) {
        System.out.println("ERROR " + msg + " " + t);
        return 0;
    }
}
