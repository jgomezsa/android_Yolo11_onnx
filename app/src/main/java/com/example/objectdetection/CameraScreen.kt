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
import androidx.compose.foundation.layout.Box
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
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.min

/** Model used by default (asset name without extension: `yolo11s.onnx` / `yolo11s.tflite`). */
const val DEFAULT_MODEL = "yolo11s"

/** Cheaper model to switch to automatically when the default one runs below [MIN_FPS]. */
private val FALLBACK_MODEL = mapOf("yolo11s" to "yolo11s_480")
private const val MIN_FPS = 3.0

/**
 * Loads [model] with [provider]. Fallback chain when the preferred provider cannot run the model:
 * LiteRT GPU (needs `$model.tflite`) -> ONNX Runtime NNAPI fp16 -> XNNPACK on the CPU.
 */
private fun createDetector(context: android.content.Context, model: String, provider: Provider): ObjectDetector {
    if (provider == Provider.LITERT_GPU) {
        try {
            return LiteRtDetector(context, "$model.tflite")
        } catch (e: Throwable) {
            Log.w("Inference", "LiteRT GPU unavailable for $model, falling back to NNAPI", e)
        }
        return YoloDetector(context, "$model.onnx", Provider.NNAPI_FP16)
    }
    return YoloDetector(context, "$model.onnx", provider)
}

/**
 * [initialModel] / [initialProvider] are developer overrides (launch extras `model` / `provider`):
 * `adb shell am start -n com.example.objectdetection/.MainActivity --es model yolo11n --es provider XNNPACK`.
 */
@Composable
fun CameraScreen(
    contentPadding: PaddingValues = PaddingValues(),
    initialModel: String = DEFAULT_MODEL,
    initialProvider: Provider = Provider.LITERT_GPU,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var result by remember { mutableStateOf(DetectionResult(emptyList(), 0)) }
    // Aspect ratio (w/h) of the upright analysis frame, used to align the overlay with the preview.
    var frameAspect by remember { mutableStateOf(3f / 4f) }

    val previewView = remember {
        PreviewView(context).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }
    }
    // The analyzer thread reloads the detector when the requested model changes.
    val requestedModel = remember { AtomicReference(initialModel) }
    var modelLabel by remember { mutableStateOf(initialModel) }
    var autoFallback by remember { mutableStateOf(false) }
    var loading by remember { mutableStateOf(true) }
    // Provider actually in use (NNAPI falls back to XNNPACK if the model cannot be loaded with it).
    var activeProvider by remember { mutableStateOf(initialProvider) }

    DisposableEffect(lifecycleOwner) {
        val executor = Executors.newSingleThreadExecutor()
        val fpsMonitor = FpsMonitor(MIN_FPS)
        var detector: ObjectDetector? = null
        var loadedModel = ""
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
                    val frameStart = System.nanoTime()
                    val raw = it.toBitmap()
                    val upright = if (it.imageInfo.rotationDegrees == 0) raw else {
                        val m = Matrix().apply { postRotate(it.imageInfo.rotationDegrees.toFloat()) }
                        Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
                    }
                    val wanted = requestedModel.get()
                    if (wanted != loadedModel) {
                        // Accelerators compile the model on creation, which can take many seconds.
                        loading = true
                        val loadStart = System.nanoTime()
                        detector?.close()
                        detector = createDetector(context, wanted, initialProvider)
                        loadedModel = wanted
                        activeProvider = detector!!.provider
                        fpsMonitor.reset()
                        loading = false
                        Log.d("Inference", "loaded $wanted ${activeProvider.name} in ${(System.nanoTime() - loadStart) / 1_000_000} ms")
                    }
                    val r = detector!!.detect(upright)
                    val frameMs = (System.nanoTime() - frameStart) / 1_000_000
                    Log.d(
                        "Inference",
                        "$loadedModel ${activeProvider.name} ${r.inferenceMs} ms " +
                            "(pre ${r.preprocessMs}, run ${r.runMs}, post ${r.postMs}) frame $frameMs ms",
                    )
                    frameAspect = upright.width.toFloat() / upright.height
                    result = r
                    // Too slow on the default model: drop to the cheaper one (once, only downwards).
                    val fallback = FALLBACK_MODEL[loadedModel]
                    if (fallback != null && fpsMonitor.record(frameMs)) {
                        Log.w("Inference", "$loadedModel below $MIN_FPS FPS, switching to $fallback")
                        requestedModel.set(fallback)
                        modelLabel = fallback
                        autoFallback = true
                    }
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
            text = if (loading) "Loading $modelLabel…" else
                "$modelLabel${if (autoFallback) " (auto: <${MIN_FPS.toInt()} FPS)" else ""} · ${activeProvider.label} · " +
                    "${result.detections.size} objects · ${result.inferenceMs} ms",
            color = Color.White,
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.align(Alignment.TopStart).padding(8.dp),
        )
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

