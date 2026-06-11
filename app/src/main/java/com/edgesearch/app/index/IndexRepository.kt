package com.edgesearch.app.index

import android.content.ContentUris
import android.content.Context
import android.graphics.BitmapFactory
import android.provider.MediaStore
import android.util.Log
import com.edgesearch.app.data.AppDatabase
import com.edgesearch.app.data.PhotoEmbedding
import com.edgesearch.app.ml.ClipEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "IndexRepository"
private const val BATCH_SIZE = 32

class IndexRepository(
    private val context: Context,
    private val engine: ClipEngine,
    private val db: AppDatabase
) {
    /**
     * Enumerates MediaStore images, skips already-indexed ids, encodes new ones.
     * [onProgress] receives (indexed so far, total new images to index).
     */
    suspend fun indexNewPhotos(onProgress: suspend (Int, Int) -> Unit) = withContext(Dispatchers.IO) {
        val existing = db.photoDao().existingIds().toHashSet()

        // Query MediaStore for all images
        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DATE_ADDED
        )
        val newRows = mutableListOf<Pair<Long, Long>>() // (mediaId, dateAdded)
        context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection, null, null,
            "${MediaStore.Images.Media.DATE_ADDED} ASC"
        )?.use { cursor ->
            val idCol   = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val dateCol = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_ADDED)
            while (cursor.moveToNext()) {
                val id = cursor.getLong(idCol)
                if (id !in existing) newRows.add(id to cursor.getLong(dateCol))
            }
        }

        Log.i(TAG, "Indexing ${newRows.size} new images (${existing.size} already indexed)")
        val total = newRows.size
        var indexed = 0
        val batch = mutableListOf<PhotoEmbedding>()

        val opts = BitmapFactory.Options().apply {
            inSampleSize = 2   // decode at half resolution; resize to 224 anyway
            inPreferredConfig = android.graphics.Bitmap.Config.ARGB_8888
        }

        for ((mediaId, dateAdded) in newRows) {
            val uri = ContentUris.withAppendedId(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, mediaId
            )
            try {
                val bitmap = context.contentResolver.openInputStream(uri)?.use {
                    BitmapFactory.decodeStream(it, null, opts)
                } ?: continue

                val embeds = engine.encodeImage(bitmap)
                bitmap.recycle()

                batch.add(PhotoEmbedding(mediaId, dateAdded, floatsToBytes(embeds)))
                indexed++

                if (batch.size >= BATCH_SIZE) {
                    db.photoDao().insertAll(batch)
                    batch.clear()
                    onProgress(indexed, total)
                }
            } catch (e: Exception) {
                Log.w(TAG, "Skipping $mediaId: ${e.message}")
            }
        }
        if (batch.isNotEmpty()) {
            db.photoDao().insertAll(batch)
            onProgress(indexed, total)
        }
        Log.i(TAG, "Indexing complete. Total indexed: ${db.photoDao().count()}")
    }

    private fun floatsToBytes(floats: FloatArray): ByteArray {
        val buf = ByteBuffer.allocate(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (f in floats) buf.putFloat(f)
        return buf.array()
    }
}
