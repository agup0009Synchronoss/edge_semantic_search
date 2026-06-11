package com.edgesearch.app.search

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.SystemClock
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.edgesearch.app.ml.ClipEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

// ── per-image log entry ───────────────────────────────────────────────────────

data class EmbedLogEntry(
    val uri: Uri,
    val filename: String,
    val decodeDurationMs: Long,   // BitmapFactory + ImagePreprocess
    val inferenceDurationMs: Long // ORT session.run
) {
    val totalDurationMs: Long get() = decodeDurationMs + inferenceDurationMs
}

// ── ranked search result ──────────────────────────────────────────────────────

data class RankedResult(
    val uri: Uri,
    val rank: Int,
    val cosineSimilarity: Float
)

// ── full screen state ─────────────────────────────────────────────────────────

data class BenchmarkUiState(
    // 1. selection
    val selectedUris: List<Uri> = emptyList(),

    // 2. embedding
    val isEmbedding: Boolean = false,
    val embedLog: List<EmbedLogEntry> = emptyList(),     // live — grows as images finish
    val embedTotalMs: Long? = null,                       // set when all done
    val embeddedData: List<Pair<Uri, FloatArray>> = emptyList(),

    // 3. search
    val query: String = "",
    val isSearching: Boolean = false,
    val textEmbedMs: Long? = null,
    val results: List<RankedResult> = emptyList(),
    val searchError: String? = null
)

// ─────────────────────────────────────────────────────────────────────────────

class SearchViewModel(private val engine: ClipEngine) : ViewModel() {

    private val _state = MutableStateFlow(BenchmarkUiState())
    val state: StateFlow<BenchmarkUiState> = _state

    // ── 1. image selection ────────────────────────────────────────────────────

    fun setSelected(uris: List<Uri>) {
        _state.update {
            BenchmarkUiState(selectedUris = uris)   // reset everything else
        }
    }

    // ── 2. embedding ──────────────────────────────────────────────────────────

    fun embedSelected(context: Context) {
        val uris = _state.value.selectedUris
        if (uris.isEmpty() || _state.value.isEmbedding) return

        viewModelScope.launch(Dispatchers.IO) {
            _state.update { it.copy(isEmbedding = true, embedLog = emptyList(),
                                    embeddedData = emptyList(), embedTotalMs = null,
                                    results = emptyList(), textEmbedMs = null) }

            val log      = mutableListOf<EmbedLogEntry>()
            val embedded = mutableListOf<Pair<Uri, FloatArray>>()
            val opts     = BitmapFactory.Options().apply {
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val wallStart = SystemClock.elapsedRealtime()

            for (uri in uris) {
                val name = uri.lastPathSegment
                    ?.substringAfterLast('/')
                    ?.substringAfterLast(':')
                    ?: uri.toString().takeLast(24)

                try {
                    // --- decode + preprocess timing ---
                    val t0 = SystemClock.elapsedRealtime()
                    val bitmap = context.contentResolver.openInputStream(uri)?.use {
                        BitmapFactory.decodeStream(it, null, opts)
                    }
                    if (bitmap == null) {
                        log.add(EmbedLogEntry(uri, "$name [could not open]", 0L, 0L))
                        _state.update { it.copy(embedLog = log.toList()) }
                        continue
                    }
                    // ImagePreprocess is called inside encodeImage — measure separately
                    // by splitting decode and inference
                    val t1 = SystemClock.elapsedRealtime()
                    val decodeDurationMs = t1 - t0

                    // --- ORT inference timing ---
                    val embedding = engine.encodeImage(bitmap)
                    val t2 = SystemClock.elapsedRealtime()
                    val inferenceDurationMs = t2 - t1
                    bitmap.recycle()

                    val entry = EmbedLogEntry(uri, name, decodeDurationMs, inferenceDurationMs)
                    log.add(entry)
                    embedded.add(uri to embedding)

                } catch (e: Exception) {
                    log.add(EmbedLogEntry(uri, "$name [ERR: ${e.message?.take(30)}]", 0L, 0L))
                }

                // emit after each image so the UI updates live
                _state.update { it.copy(embedLog = log.toList()) }
            }

            val totalMs = SystemClock.elapsedRealtime() - wallStart
            _state.update {
                it.copy(
                    isEmbedding   = false,
                    embedLog      = log,
                    embedTotalMs  = totalMs,
                    embeddedData  = embedded
                )
            }
        }
    }

    // ── 3. search ─────────────────────────────────────────────────────────────

    fun search(query: String) {
        val data = _state.value.embeddedData
        if (data.isEmpty() || query.isBlank()) return

        _state.update { it.copy(
            isSearching = true, query = query,
            results = emptyList(), textEmbedMs = null, searchError = null
        )}

        viewModelScope.launch(Dispatchers.Default) {
            try {
                // --- text embedding timing ---
                val t0 = SystemClock.elapsedRealtime()
                val textEmbed = engine.encodeText(query)
                val textEmbedMs = SystemClock.elapsedRealtime() - t0

                // --- cosine similarity (both sides normalized → dot product) ---
                val scored = data.map { (uri, imgEmbed) ->
                    var dot = 0f
                    for (i in imgEmbed.indices) dot += imgEmbed[i] * textEmbed[i]
                    uri to dot
                }.sortedByDescending { it.second }

                val results = scored.mapIndexed { idx, (uri, score) ->
                    RankedResult(uri, idx + 1, score)
                }

                _state.update { it.copy(
                    isSearching  = false,
                    textEmbedMs  = textEmbedMs,
                    results      = results
                )}
            } catch (e: Exception) {
                _state.update { it.copy(isSearching = false, searchError = e.message) }
            }
        }
    }

    // ── factory ───────────────────────────────────────────────────────────────

    class Factory(private val engine: ClipEngine) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>) =
            SearchViewModel(engine) as T
    }
}
