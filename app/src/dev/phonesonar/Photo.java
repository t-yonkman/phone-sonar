package dev.phonesonar;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.util.Log;
import android.util.Size;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * Takes one photo with the phone's widest rear camera. The back of the phone faces the object, so a photo
 * taken as each recording ends shows what stood where, and settles any later doubt about a label. It is
 * taken after the recording, never during it, and a failed photo never holds up a run.
 */
final class Photo {
    interface Done {
        void done(boolean ok);
    }

    private static final int MIN_WIDTH = 1600;
    private static final long SETTLE_MS = 1200, GIVE_UP_MS = 6000;   // for focus and exposure to settle; for the whole photo

    private final Context context;
    private final CameraManager cameras;
    private final String id;
    private final Handler ui = new Handler(Looper.getMainLooper());

    Photo(Context context) {
        this.context = context;
        cameras = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        id = widest();
        Log.i(Sonar.TAG, "photos from camera " + id);
    }

    /** The rear camera with the shortest focal length: an object 10 cm out is well off to one side of the lens. */
    private String widest() {
        String best = null;
        float shortest = Float.MAX_VALUE;
        try {
            for (String c : cameras.getCameraIdList()) {
                CameraCharacteristics about = cameras.getCameraCharacteristics(c);
                Integer facing = about.get(CameraCharacteristics.LENS_FACING);
                float[] lengths = about.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
                if (facing == null || facing != CameraCharacteristics.LENS_FACING_BACK || lengths == null) continue;
                for (float length : lengths) {
                    if (length < shortest) {
                        shortest = length;
                        best = c;
                    }
                }
            }
        } catch (CameraAccessException e) {
            Log.e(Sonar.TAG, "could not list the cameras", e);
        }
        return best;
    }

    /** Take a photo and save it as a JPEG. done is called once, on the main thread, however it went. */
    void take(File to, Done done) {
        if (id == null || context.checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            done.done(false);
            return;
        }
        new Shot(to, done).open();
    }

    /** One photo from opening the camera to closing it, all on its own thread. */
    private final class Shot {
        private final File to;
        private final Done done;
        private final HandlerThread thread = new HandlerThread("photo");
        private final Handler here;
        private CameraDevice camera;
        private CameraCaptureSession session;
        private ImageReader still, frames;
        private boolean over;

        Shot(File to, Done done) {
            this.to = to;
            this.done = done;
            thread.start();
            here = new Handler(thread.getLooper());
        }

        void open() {
            here.postDelayed(() -> end(false), GIVE_UP_MS);
            try {
                cameras.openCamera(id, new CameraDevice.StateCallback() {
                    @Override
                    public void onOpened(CameraDevice c) {
                        camera = c;
                        configure();
                    }

                    @Override
                    public void onDisconnected(CameraDevice c) {
                        end(false);
                    }

                    @Override
                    public void onError(CameraDevice c, int error) {
                        Log.e(Sonar.TAG, "camera error " + error);
                        end(false);
                    }
                }, here);
            } catch (CameraAccessException | SecurityException e) {
                Log.e(Sonar.TAG, "could not open the camera", e);
                end(false);
            }
        }

        /** The smallest JPEG size at least MIN_WIDTH wide, or else the largest there is. */
        private Size size() throws CameraAccessException {
            StreamConfigurationMap map = cameras.getCameraCharacteristics(id).get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Size best = null, largest = null;
            for (Size s : map.getOutputSizes(ImageFormat.JPEG)) {
                if (largest == null || s.getWidth() > largest.getWidth()) largest = s;
                if (s.getWidth() >= MIN_WIDTH && (best == null || s.getWidth() < best.getWidth())) best = s;
            }
            return best != null ? best : largest;
        }

        private void configure() {
            try {
                Size s = size();
                still = ImageReader.newInstance(s.getWidth(), s.getHeight(), ImageFormat.JPEG, 1);
                still.setOnImageAvailableListener(r -> save(), here);
                // Frames nobody looks at, so that focus and exposure can settle before the photo
                frames = ImageReader.newInstance(640, 480, ImageFormat.YUV_420_888, 2);
                frames.setOnImageAvailableListener(r -> {
                    Image i = r.acquireLatestImage();
                    if (i != null) i.close();
                }, here);
                camera.createCaptureSession(Arrays.asList(frames.getSurface(), still.getSurface()), new CameraCaptureSession.StateCallback() {
                    @Override
                    public void onConfigured(CameraCaptureSession s) {
                        session = s;
                        settle();
                    }

                    @Override
                    public void onConfigureFailed(CameraCaptureSession s) {
                        end(false);
                    }
                }, here);
            } catch (CameraAccessException | IllegalStateException e) {
                Log.e(Sonar.TAG, "could not set up the camera", e);
                end(false);
            }
        }

        private void settle() {
            try {
                CaptureRequest.Builder b = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                b.addTarget(frames.getSurface());
                b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                session.setRepeatingRequest(b.build(), null, here);
                here.postDelayed(this::shoot, SETTLE_MS);
            } catch (CameraAccessException | IllegalStateException e) {
                Log.e(Sonar.TAG, "could not start the camera", e);
                end(false);
            }
        }

        private void shoot() {
            if (over) return;
            try {
                CaptureRequest.Builder b = camera.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
                b.addTarget(still.getSurface());
                b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                b.set(CaptureRequest.JPEG_QUALITY, (byte) 85);
                session.capture(b.build(), null, here);
            } catch (CameraAccessException | IllegalStateException e) {
                Log.e(Sonar.TAG, "could not take the photo", e);
                end(false);
            }
        }

        private void save() {
            try (Image image = still.acquireLatestImage(); FileOutputStream out = new FileOutputStream(to)) {
                ByteBuffer b = image.getPlanes()[0].getBuffer();
                byte[] jpeg = new byte[b.remaining()];
                b.get(jpeg);
                out.write(jpeg);
                end(true);
            } catch (IOException | RuntimeException e) {
                Log.e(Sonar.TAG, "could not save the photo", e);
                end(false);
            }
        }

        private void end(boolean ok) {
            if (over) return;
            over = true;
            here.removeCallbacksAndMessages(null);
            try {
                if (session != null) session.close();
                if (camera != null) camera.close();
                if (still != null) still.close();
                if (frames != null) frames.close();
            } catch (RuntimeException e) {
                Log.e(Sonar.TAG, "could not close the camera", e);
            }
            thread.quitSafely();
            ui.post(() -> done.done(ok));
        }
    }
}
