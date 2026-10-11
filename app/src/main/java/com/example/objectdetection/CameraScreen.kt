package com.example.objectdetection

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.compose.material3.Button
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.min

/** ONNX models bundled in assets (without the `.onnx` extension). */
private val MODELS = listOf("yolo11n", "yolo11s", "yolo11s_480")

@Composable
fun CameraScreen(contentPadding: PaddingValues = PaddingValues()) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var result by remember { mutableStateOf(DetectionResult(emptyList(), 0)) }
    // Aspect ratio (w/h) of the upright analysis frame, used to align the overlay with the preview.
    var frameAspect by remember { mutableStateOf(3f / 4f) }

    val previewView = remember {
        PreviewView(context).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }
    }
    // The analyzer thread reloads the detector when the requested model or provider changes.
    val requestedModel = remember { AtomicReference(MODELS.first()) }
    val requestedProvider = remember { AtomicReference(Provider.NNAPI_FP16) }
    var modelLabel by remember { mutableStateOf(MODELS.first()) }
    var providerLabel by remember { mutableStateOf(Provider.NNAPI_FP16) }
    // Provider actually in use (NNAPI may fall back to XNNPACK).
    var activeProvider by remember { mutableStateOf(Provider.NNAPI_FP16) }

    DisposableEffect(lifecycleOwner) {
        val executor = Executors.newSingleThreadExecutor()
        var detector: YoloDetector? = null
        var loadedModel = ""
        var loadedProvider: Provider? = null
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy(AspectRatio.RATIO_4_3, AspectRatioStrategy.FALLBACK_RULE_AUTO))
                .build()
            val preview = Preview.Builder().setResolutionSelector(selector).build()
                .also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(executor) { image ->
                image.use {
                    val raw = it.toBitmap()
                    val upright = if (it.imageInfo.rotationDegrees == 0) raw else {
                        val m = Matrix().apply { postRotate(it.imageInfo.rotationDegrees.toFloat()) }
                        Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
                    }
                    val wanted = requestedModel.get()
                    val wantedProvider = requestedProvider.get()
                    if (wanted != loadedModel || wantedProvider != loadedProvider) {
                        detector?.close()
                        detector = YoloDetector(context, "$wanted.onnx", wantedProvider)
                        loadedModel = wanted
                        loadedProvider = wantedProvider
                        activeProvider = detector!!.provider
                    }
                    val r = detector!!.detect(upright)
                    Log.d("Inference", "$loadedModel ${activeProvider.name} ${r.inferenceMs} ms")
                    frameAspect = upright.width.toFloat() / upright.height
                    result = r
                }
            }
            provider.unbindAll()
            provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            ProcessCameraProvider.getInstance(context).get().unbindAll()
            executor.shutdown()
            executor.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS)
            detector?.close()
        }
    }

    Box(modifier = Modifier.fillMaxSize().padding(contentPadding)) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
        DetectionOverlay(result.detections, frameAspect, Modifier.fillMaxSize())
        Text(
            text = "$modelLabel · ${activeProvider.label} · ${result.detections.size} objects · ${result.inferenceMs} ms",
            color = Color.White,
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.align(Alignment.TopStart).padding(8.dp),
        )
        Row(
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp),
        ) {
            Button(onClick = {
                val next = MODELS[(MODELS.indexOf(modelLabel) + 1) % MODELS.size]
                modelLabel = next
                requestedModel.set(next)
            }) { Text("Switch model") }
            Button(onClick = {
                val all = Provider.entries
                val next = all[(all.indexOf(providerLabel) + 1) % all.size]
                providerLabel = next
                requestedProvider.set(next)
            }) { Text("Switch provider") }
        }
    }
}

@Composable
private fun DetectionOverlay(detections: List<Detection>, frameAspect: Float, modifier: Modifier = Modifier) {
    val textPaint = remember {
        android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 42f
            isAntiAlias = true
        }
    }
    val bgPaint = remember { android.graphics.Paint().apply { style = android.graphics.Paint.Style.FILL } }

    Canvas(modifier = modifier) {
        // The preview is letterboxed (FIT_CENTER); compute the rectangle the frame occupies.
        val fit = min(size.width / frameAspect, size.height)
        val frameH = fit
        val frameW = fit * frameAspect
        val originX = (size.width - frameW) / 2f
        val originY = (size.height - frameH) / 2f

        for (d in detections) {
            val color = Color.hsv((d.classId * 47 % 360).toFloat(), 0.85f, 1f)
            val left = originX + d.left * frameW
            val top = originY + d.top * frameH
            drawRect(
                color = color,
                topLeft = Offset(left, top),
                size = Size((d.right - d.left) * frameW, (d.bottom - d.top) * frameH),
                style = Stroke(width = 4f),
            )
            val text = "${d.label} ${(d.score * 100).toInt()}%"
            val textW = textPaint.measureText(text)
            bgPaint.color = color.copy(alpha = 0.8f).toArgb()
            drawContext.canvas.nativeCanvas.apply {
                drawRect(left, top - 52f, left + textW + 16f, top, bgPaint)
                drawText(text, left + 8f, top - 12f, textPaint)
            }
        }
    }
}

