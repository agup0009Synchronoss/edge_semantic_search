package com.edgesearch.app.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.extensions.OrtxPackage
import java.nio.LongBuffer

private const val SEQ_LEN = 77
private const val BOS = 49406L
private const val EOT = 49407L

/**
 * Wraps custom_op_cliptok.onnx via onnxruntime-extensions.
 * Produces padded/truncated input_ids[1,77] and attention_mask[1,77] (int64).
 */
class Tokenizer(env: OrtEnvironment, tokenizerModelPath: String) : AutoCloseable {

    private val session: OrtSession
    private val textInputName: String

    init {
        val opts = OrtSession.SessionOptions()
        opts.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        session = env.createSession(tokenizerModelPath, opts)
        // The tokenizer model takes a single string input (name varies by export)
        textInputName = session.inputNames.iterator().next()
    }

    data class TokenizerOutput(
        val inputIds: LongBuffer,      // [1, SEQ_LEN]
        val attentionMask: LongBuffer  // [1, SEQ_LEN]
    )

    fun tokenize(env: OrtEnvironment, text: String): TokenizerOutput {
        // Run the tokenizer ONNX to get raw ids (variable length)
        val inputTensor = OnnxTensor.createTensor(env, arrayOf(text))
        val results = session.run(mapOf(textInputName to inputTensor))
        inputTensor.close()

        // Extract ids as LongArray; shape may be [1, N] or [N]
        val rawIds: LongArray = when (val v = (results[0] as OnnxTensor).value) {
            is LongArray -> v
            is Array<*>  -> (v[0] as LongArray)
            else         -> LongArray(0)
        }
        results.close()

        // Build padded output: [BOS, ids..., EOT, 0, 0, …] length SEQ_LEN
        val ids  = LongArray(SEQ_LEN)
        val mask = LongArray(SEQ_LEN)
        ids[0] = BOS
        var pos = 1
        for (id in rawIds) {
            if (pos >= SEQ_LEN - 1) break  // leave room for EOT
            ids[pos++] = id
        }
        ids[pos] = EOT
        // attention_mask: 1 for real tokens, 0 for padding
        for (i in 0..pos) mask[i] = 1L

        return TokenizerOutput(LongBuffer.wrap(ids), LongBuffer.wrap(mask))
    }

    override fun close() = session.close()
}
