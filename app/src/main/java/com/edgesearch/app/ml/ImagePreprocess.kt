package com.edgesearch.app.ml

import android.graphics.Bitmap
import java.nio.FloatBuffer

// CLIP preprocessing constants (must match exactly)
private val MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
private val STD  = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)
private const val CROP = 224

object ImagePreprocess {
    // Reusable buffer: 1 * 3 * 224 * 224 floats
    private val buf = FloatBuffer.allocate(1 * 3 * CROP * CROP)

    /**
     * Converts a Bitmap to a NCHW float32 tensor [1,3,224,224] matching CLIP preprocessing:
     * resize shorter side to 224, center-crop 224x224, /255, per-channel mean/std normalize.
     * Returns the internal reusable FloatBuffer (rewound); copy if needed across threads.
     */
    fun process(src: Bitmap): FloatBuffer {
        // Step 1: resize so shorter side = 224
        val (w, h) = src.width to src.height
        val (newW, newH) = if (w < h) CROP to (h * CROP / w) else (w * CROP / h) to CROP
        val scaled = Bitmap.createScaledBitmap(src, newW, newH, true)

        // Step 2: center crop 224×224
        val left = (newW - CROP) / 2
        val top  = (newH - CROP) / 2
        val pixels = IntArray(CROP * CROP)
        scaled.getPixels(pixels, 0, CROP, left, top, CROP, CROP)
        if (scaled !== src) scaled.recycle()

        // Step 3-4: [0,1] + per-channel normalize → NCHW layout
        buf.rewind()
        val r = FloatArray(CROP * CROP)
        val g = FloatArray(CROP * CROP)
        val b = FloatArray(CROP * CROP)
        for (i in pixels.indices) {
            val px = pixels[i]
            r[i] = ((px shr 16 and 0xFF) / 255f - MEAN[0]) / STD[0]
            g[i] = ((px shr  8 and 0xFF) / 255f - MEAN[1]) / STD[1]
            b[i] = ((px        and 0xFF) / 255f - MEAN[2]) / STD[2]
        }
        buf.put(r); buf.put(g); buf.put(b)
        buf.rewind()
        return buf
    }
}
