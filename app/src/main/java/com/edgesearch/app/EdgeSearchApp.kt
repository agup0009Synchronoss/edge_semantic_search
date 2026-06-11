package com.edgesearch.app

import android.app.Application
import android.content.Context
import android.util.Log
import com.edgesearch.app.data.AppDatabase
import com.edgesearch.app.index.IndexRepository
import com.edgesearch.app.index.IndexWorker
import com.edgesearch.app.ml.ClipEngine
import ai.onnxruntime.OrtEnvironment
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.io.File

private const val TAG = "EdgeSearchApp"
private const val PREF_ASSETS_COPIED = "assets_copied_v7"

class EdgeSearchApp : Application() {

    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val _engineReady = MutableStateFlow(false)
    val engineReady: StateFlow<Boolean> = _engineReady

    lateinit var clipEngine: ClipEngine
    lateinit var db: AppDatabase
    lateinit var indexRepository: IndexRepository

    override fun onCreate() {
        super.onCreate()
        db = AppDatabase.getInstance(this)
        appScope.launch {
            try {
                copyAssetsOnce()
                val env = OrtEnvironment.getEnvironment()
                clipEngine = ClipEngine(
                    env,
                    assetPath("vision_model_fp32.onnx"),
                    assetPath("text_model_int8.onnx"),
                    assetPath("custom_op_cliptok.onnx")
                )
                indexRepository = IndexRepository(this@EdgeSearchApp, clipEngine, db)
                IndexWorker.schedulePeriodicIndex(this@EdgeSearchApp)
                _engineReady.value = true
                Log.i(TAG, "ClipEngine ready")
            } catch (e: Exception) {
                Log.e(TAG, "ClipEngine init failed", e)
            }
        }
    }

    private fun copyAssetsOnce() {
        val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        if (prefs.getBoolean(PREF_ASSETS_COPIED, false)) return

        // Always overwrite — pref key version change means assets changed
        listOf("vision_model_fp32.onnx", "text_model_int8.onnx", "custom_op_cliptok.onnx")
            .forEach { name ->
                val dest = File(filesDir, name)
                assets.open(name).use { input ->
                    dest.outputStream().use { input.copyTo(it) }
                }
                Log.i(TAG, "Copied asset: $name (${dest.length()} bytes)")
            }
        prefs.edit().putBoolean(PREF_ASSETS_COPIED, true).apply()
    }

    fun assetPath(name: String) = File(filesDir, name).absolutePath
}
