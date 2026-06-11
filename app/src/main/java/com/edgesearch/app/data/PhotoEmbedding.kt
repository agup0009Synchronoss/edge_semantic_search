package com.edgesearch.app.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "photo_embeddings")
data class PhotoEmbedding(
    @PrimaryKey val mediaId: Long,
    val dateAdded: Long,
    // 512 float32 values stored as raw bytes (2048 bytes per row)
    val embedding: ByteArray
) {
    override fun equals(other: Any?) = other is PhotoEmbedding && mediaId == other.mediaId
    override fun hashCode() = mediaId.hashCode()
}
