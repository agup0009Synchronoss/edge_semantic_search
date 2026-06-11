package com.edgesearch.app.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface PhotoDao {
    @Query("SELECT mediaId FROM photo_embeddings")
    suspend fun existingIds(): List<Long>

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertAll(rows: List<PhotoEmbedding>)

    @Query("SELECT * FROM photo_embeddings")
    suspend fun loadAll(): List<PhotoEmbedding>

    @Query("SELECT COUNT(*) FROM photo_embeddings")
    suspend fun count(): Int
}
