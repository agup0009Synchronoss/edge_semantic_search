package com.edgesearch.app.search

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage

@Composable
fun SearchScreen(viewModel: SearchViewModel) {
    val state   by viewModel.state.collectAsState()
    val context = LocalContext.current
    var queryText by remember { mutableStateOf("") }

    val pickLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetMultipleContents()
    ) { uris -> if (uris.isNotEmpty()) viewModel.setSelected(uris) }

    // Root: LazyColumn so the embed log + results can both scroll together
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = PaddingValues(vertical = 12.dp)
    ) {

        // ── 1. Pick + Embed buttons ───────────────────────────────────────────
        item {
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                OutlinedButton(
                    onClick = { pickLauncher.launch("image/*") },
                    modifier = Modifier.weight(1f)
                ) {
                    Text(
                        if (state.selectedUris.isEmpty()) "Pick images"
                        else "${state.selectedUris.size} selected"
                    )
                }
                Button(
                    onClick   = { viewModel.embedSelected(context) },
                    enabled   = state.selectedUris.isNotEmpty() && !state.isEmbedding,
                    modifier  = Modifier.weight(1f)
                ) {
                    if (state.isEmbedding) {
                        CircularProgressIndicator(
                            modifier    = Modifier.size(14.dp),
                            strokeWidth = 2.dp,
                            color       = MaterialTheme.colorScheme.onPrimary
                        )
                        Spacer(Modifier.width(6.dp))
                        Text("Embedding…")
                    } else {
                        Text("Embed ${state.selectedUris.size}")
                    }
                }
            }
        }

        // ── 2. Live embedding log ─────────────────────────────────────────────
        if (state.embedLog.isNotEmpty() || state.isEmbedding) {
            item {
                EmbedLogCard(
                    log       = state.embedLog,
                    totalMs   = state.embedTotalMs,
                    embedding = state.isEmbedding
                )
            }
        }

        // ── 3. Text search bar (only available once embeddings exist) ─────────
        if (state.embeddedData.isNotEmpty()) {
            item {
                Column {
                    Row(
                        verticalAlignment         = Alignment.CenterVertically,
                        horizontalArrangement     = Arrangement.spacedBy(8.dp),
                        modifier                  = Modifier.fillMaxWidth()
                    ) {
                        OutlinedTextField(
                            value         = queryText,
                            onValueChange = { queryText = it },
                            placeholder   = { Text("Search query…") },
                            modifier      = Modifier.weight(1f),
                            singleLine    = true
                        )
                        Button(
                            onClick  = { viewModel.search(queryText) },
                            enabled  = queryText.isNotBlank() && !state.isSearching
                        ) {
                            if (state.isSearching)
                                CircularProgressIndicator(
                                    Modifier.size(14.dp), strokeWidth = 2.dp,
                                    color = MaterialTheme.colorScheme.onPrimary
                                )
                            else Text("Search")
                        }
                    }

                    // text embedding latency chip
                    state.textEmbedMs?.let { ms ->
                        Spacer(Modifier.height(4.dp))
                        Surface(
                            shape = RoundedCornerShape(4.dp),
                            color = MaterialTheme.colorScheme.secondaryContainer
                        ) {
                            Text(
                                text     = "Text embedding: ${ms} ms",
                                style    = MaterialTheme.typography.labelSmall,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                color    = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                        }
                    }

                    state.searchError?.let { err ->
                        Text(err, color = MaterialTheme.colorScheme.error,
                             style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }

        // ── 4. Ranked results grid ────────────────────────────────────────────
        if (state.results.isNotEmpty()) {
            item {
                Text(
                    "Results — \"${state.query}\"  |  ${state.results.size} images ranked",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.secondary
                )
            }

            // Build rows of 3 for the grid (simpler than nested LazyVerticalGrid)
            val rows = state.results.chunked(3)
            items(rows) { row ->
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    row.forEach { result ->
                        ResultCard(result = result, modifier = Modifier.weight(1f))
                    }
                    // pad incomplete last row
                    repeat(3 - row.size) {
                        Spacer(Modifier.weight(1f))
                    }
                }
            }
        }
    }
}

// ── Embed log card ─────────────────────────────────────────────────────────────

@Composable
private fun EmbedLogCard(
    log: List<EmbedLogEntry>,
    totalMs: Long?,
    embedding: Boolean
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(10.dp)) {

            Row(verticalAlignment = Alignment.CenterVertically) {
                if (embedding) {
                    CircularProgressIndicator(Modifier.size(12.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(6.dp))
                }
                Text(
                    "Embedding log  (${log.size} done)",
                    style = MaterialTheme.typography.labelMedium
                )
            }

            Spacer(Modifier.height(6.dp))

            // Header row
            Row(modifier = Modifier.fillMaxWidth()) {
                Text("Image", Modifier.weight(1f),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.outline)
                Text("Decode", Modifier.width(58.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.outline)
                Text("Infer", Modifier.width(58.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.outline)
                Text("Total", Modifier.width(52.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.outline)
            }

            HorizontalDivider(Modifier.padding(vertical = 3.dp))

            log.forEach { entry ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text     = entry.filename,
                        modifier = Modifier.weight(1f),
                        style    = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    TimingCell(entry.decodeDurationMs,   width = 58.dp)
                    TimingCell(entry.inferenceDurationMs, width = 58.dp,
                        color = MaterialTheme.colorScheme.primary)
                    TimingCell(entry.totalDurationMs,    width = 52.dp,
                        color = MaterialTheme.colorScheme.tertiary)
                }
            }

            totalMs?.let { total ->
                HorizontalDivider(Modifier.padding(vertical = 4.dp))
                val avg = if (log.isNotEmpty()) total / log.size else 0L
                Row(modifier = Modifier.fillMaxWidth()) {
                    Text(
                        "Wall total: ${total} ms   avg/img: ${avg} ms",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.secondary,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
    }
}

@Composable
private fun TimingCell(
    ms: Long,
    width: androidx.compose.ui.unit.Dp,
    color: Color = MaterialTheme.colorScheme.onSurface
) {
    Text(
        text      = "${ms} ms",
        modifier  = Modifier.width(width),
        style     = MaterialTheme.typography.bodySmall,
        fontFamily = FontFamily.Monospace,
        color     = color,
        maxLines  = 1
    )
}

// ── Single result card ─────────────────────────────────────────────────────────

@Composable
private fun ResultCard(result: RankedResult, modifier: Modifier = Modifier) {
    Column(
        modifier            = modifier,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box {
            AsyncImage(
                model            = result.uri,
                contentDescription = "Rank ${result.rank}",
                contentScale     = ContentScale.Crop,
                modifier         = Modifier
                    .aspectRatio(1f)
                    .fillMaxWidth()
            )
            // Rank badge — top-left
            Surface(
                color  = MaterialTheme.colorScheme.primary.copy(alpha = 0.85f),
                shape  = RoundedCornerShape(bottomEnd = 6.dp),
                modifier = Modifier.align(Alignment.TopStart)
            ) {
                Text(
                    text     = "#${result.rank}",
                    style    = MaterialTheme.typography.labelSmall,
                    color    = Color.White,
                    modifier = Modifier.padding(horizontal = 5.dp, vertical = 2.dp)
                )
            }
        }
        // Cosine score
        Text(
            text  = "cos: ${"%.3f".format(result.cosineSimilarity)}",
            style = MaterialTheme.typography.labelSmall,
            color = scoreColor(result.cosineSimilarity),
            modifier = Modifier.padding(top = 2.dp)
        )
    }
}

@Composable
private fun scoreColor(score: Float): Color {
    val colors = MaterialTheme.colorScheme
    return when {
        score >= 0.28f -> colors.primary
        score >= 0.20f -> colors.secondary
        else           -> colors.outline
    }
}
