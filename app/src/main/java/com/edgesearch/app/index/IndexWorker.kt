package com.edgesearch.app.index

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.annotation.SuppressLint
import androidx.core.app.NotificationCompat
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.ForegroundInfo
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.edgesearch.app.EdgeSearchApp
import java.util.concurrent.TimeUnit

private const val CHANNEL_ID     = "photo_indexing"
private const val NOTIF_ID       = 1001
private const val WORK_NAME      = "photo_index_periodic"
private const val WORK_EXPEDITED = "photo_index_now"

class IndexWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {

    override suspend fun doWork(): Result {
        val app = applicationContext as EdgeSearchApp
        setForeground(buildForegroundInfo("Building photo index…"))

        return try {
            app.indexRepository.indexNewPhotos { done, total ->
                // Update via setForeground — no POST_NOTIFICATIONS needed for foreground services
                setForeground(buildForegroundInfo("Indexed $done / $total"))
            }
            Result.success()
        } catch (e: Exception) {
            Result.retry()
        }
    }

    private fun buildForegroundInfo(text: String): ForegroundInfo {
        createChannel()
        return ForegroundInfo(NOTIF_ID, buildNotification(text).build())
    }

    private fun buildNotification(text: String) =
        NotificationCompat.Builder(applicationContext, CHANNEL_ID)
            .setContentTitle("Edge Search")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_gallery)
            .setOngoing(true)
            .setSilent(true)

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE)
                as NotificationManager
            if (nm.getNotificationChannel(CHANNEL_ID) == null) {
                nm.createNotificationChannel(
                    NotificationChannel(
                        CHANNEL_ID, "Photo indexing",
                        NotificationManager.IMPORTANCE_LOW
                    )
                )
            }
        }
    }

    companion object {
        /**
         * Schedule periodic background indexing (charging + idle, ~24h interval).
         * Note: both constraints together may not be met on some devices —
         * that's intentional; the "Index now" debug button bypasses them.
         */
        @SuppressLint("IdleBatteryChargingConstraints")
        fun schedulePeriodicIndex(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiresCharging(true)
                .setRequiresDeviceIdle(true)
                .build()

            val request = PeriodicWorkRequestBuilder<IndexWorker>(1, TimeUnit.DAYS)
                .setConstraints(constraints)
                .build()

            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                WORK_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                request
            )
        }

        /** Enqueue an immediate, expedited one-time index run (no idle/charging constraint). */
        fun scheduleImmediateIndex(context: Context) {
            val request = OneTimeWorkRequestBuilder<IndexWorker>()
                .setExpedited(androidx.work.OutOfQuotaPolicy.RUN_AS_NON_EXPEDITED_WORK_REQUEST)
                .build()

            WorkManager.getInstance(context).enqueueUniqueWork(
                WORK_EXPEDITED,
                ExistingWorkPolicy.REPLACE,
                request
            )
        }
    }
}
