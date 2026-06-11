package com.edgesearch.app.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.util.Log
import kotlin.math.sqrt

private const val TAG = "ClipEngine"
private const val EMBED_DIM = 512

/**
 * Owns both ORT sessions (vision int8, text fp16) and the tokenizer session.
 * Created once at app start; never re-created per call.
 *
 * Session startup validates that expected tensor names exist in each model;
 * mismatches throw immediately with a clear message rather than producing
 * silent garbage embeddings.
 */
class ClipEngine(
    private val env: OrtEnvironment,
    visionModelPath: String,
    textModelPath: String,
    tokenizerModelPath: String
) : AutoCloseable {

    private val visionSession: OrtSession
    private val textSession: OrtSession
    private val tokenizer: Tokenizer

    // Detected at startup — text graph from CLIP*ModelWithProjection may not need attention_mask
    private val textHasAttentionMask: Boolean

    init {
        // Vision session — NNAPI EP with CPU fallback
        val vOpts = OrtSession.SessionOptions().apply {
            try {
                addNnapi()
                Log.i(TAG, "Vision session: NNAPI EP enabled")
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI unavailable, using CPU: ${e.message}")
            }
        }
        visionSession = env.createSession(visionModelPath, vOpts)
        assertNames(
            "vision",
            visionSession.inputNames,
            visionSession.outputInfo.keys,
            required_in  = setOf("pixel_values"),
            required_out = setOf("image_embeds")
        )

        // Text session — CPU/XNNPACK (text model is tiny, runs once per query)
        textSession = env.createSession(textModelPath)
        textHasAttentionMask = "attention_mask" in textSession.inputNames
        assertNames(
            "text",
            textSession.inputNames,
            textSession.outputInfo.keys,
            required_in  = setOf("input_ids"),
            required_out = setOf("text_embeds")
        )
        Log.i(TAG, "Text session inputs: ${textSession.inputNames}, attention_mask present: $textHasAttentionMask")

        tokenizer = Tokenizer(env, tokenizerModelPath)
    }

    private fun assertNames(
        label: String,
        actualIn: Set<String>,
        actualOut: Set<String>,
        required_in: Set<String>,
        required_out: Set<String>
    ) {
        Log.i(TAG, "$label inputs : $actualIn")
        Log.i(TAG, "$label outputs: $actualOut")
        val missingIn  = required_in  - actualIn
        val missingOut = required_out - actualOut
        check(missingIn.isEmpty() && missingOut.isEmpty()) {
            "[$label] model mismatch — missing inputs=$missingIn, missing outputs=$missingOut. " +
            "Re-run tools/export_split.py and replace the asset files."
        }
    }

    /** Returns L2-normalized 512-d embedding for a Bitmap (CLIP §4 preprocessing applied). */
    fun encodeImage(bitmap: Bitmap): FloatArray {
        val pixelsBuf = ImagePreprocess.process(bitmap)
        val pixelTensor = OnnxTensor.createTensor(
            env, pixelsBuf, longArrayOf(1, 3, 224, 224)
        )
        val results = visionSession.run(
            mapOf("pixel_values" to pixelTensor),
            setOf("image_embeds")
        )
        pixelTensor.close()
        val embeds = (results.get("image_embeds").get() as OnnxTensor).value as Array<FloatArray>
        results.close()
        return l2normalize(embeds[0])
    }

    /** Returns L2-normalized 512-d embedding for a search query string. */
    fun encodeText(query: String): FloatArray {
        val tok = tokenizer.tokenize(env, query.lowercase().trim())
        val idsTensor   = OnnxTensor.createTensor(env, tok.inputIds,   longArrayOf(1, 77))
        val maskTensor  = OnnxTensor.createTensor(env, tok.attentionMask, longArrayOf(1, 77))

        val inputs = buildMap {
            put("input_ids", idsTensor)
            if (textHasAttentionMask) put("attention_mask", maskTensor)
        }
        val results = textSession.run(inputs, setOf("text_embeds"))
        idsTensor.close(); maskTensor.close()

        val embeds = (results.get("text_embeds").get() as OnnxTensor).value as Array<FloatArray>
        results.close()
        return l2normalize(embeds[0])
    }

    private fun l2normalize(v: FloatArray): FloatArray {
        var norm = 0f
        for (x in v) norm += x * x
        norm = sqrt(norm)
        if (norm < 1e-8f) return v
        return FloatArray(v.size) { v[it] / norm }
    }

    override fun close() {
        tokenizer.close()
        textSession.close()
        visionSession.close()
        env.close()
    }
}
