package com.example.objectdetection

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
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
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.min

/** Model used by default (asset name without extension: `yolo11s.onnx` / `yolo11s.tflite`). */
const val DEFAULT_MODEL = "yolo11s"

/** Cheaper model to switch to automatically when the default one runs below [MIN_FPS]. */
private val FALLBACK_MODEL = mapOf("yolo11s" to "yolo11s_480")
private const val MIN_FPS = 3.0

/** Camera analysis size (landscape 4:3). It is also what is shown on screen and recorded, so it sets the picture sharpness. */
private const val ANALYSIS_WIDTH = 1280
private const val ANALYSIS_HEIGHT = 960

/** Number of recent frames shown in the processing-time plot. */
private const val FRAME_HISTORY = 50

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
 * `adb shell am start -n com.example.objectdetection/.MainActivity --es model yolo11s_480 --es provider XNNPACK`.
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
    // The frame on screen together with the detections computed on that very frame, so boxes never lag behind the picture.
    var shown by remember { mutableStateOf<ShownFrame?>(null) }
    // The analyzer thread reloads the detector when the requested model changes.
    val requestedModel = remember { AtomicReference(initialModel) }
    var loading by remember { mutableStateOf(true) }
    var usingGpu by remember { mutableStateOf(false) }
    var frameTimes by remember { mutableStateOf(emptyList<Long>()) }
    // When false the camera image keeps updating but frames are not analysed (no detection, no boxes).
    var processing by remember { mutableStateOf(true) }
    val processingFlag = remember { AtomicBoolean(true) }
    // Video recording: the UI sets the flag, the analyzer thread starts/stops the recorder and reports the elapsed time.
    var recording by remember { mutableStateOf(false) }
    var recordedMs by remember { mutableStateOf(0L) }
    val recordFlag = remember { AtomicBoolean(false) }

    DisposableEffect(lifecycleOwner) {
        val executor = Executors.newSingleThreadExecutor()
        val fpsMonitor = FpsMonitor(MIN_FPS)
        val history = FrameTimeHistory(FRAME_HISTORY)
        var detector: ObjectDetector? = null
        var loadedModel = ""
        // Provider actually in use (a requested one may have fallen back); only logged, the user never sees it.
        var activeProvider = initialProvider
        var recorder: VideoRecorder? = null
        val mainExecutor = ContextCompat.getMainExecutor(context)
        fun toast(message: String) = mainExecutor.execute { Toast.makeText(context, message, Toast.LENGTH_LONG).show() }
        fun stopRecording(announce: Boolean = true) {
            val rec = recorder ?: return
            recorder = null
            recordFlag.set(false)
            recording = false
            val saved = rec.finish()
            Log.d("Inference", "recording finished, saved=$saved")
            if (announce) toast(if (saved) "Video saved to Movies/ObjectDetection" else "Nothing was recorded")
        }
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy(AspectRatio.RATIO_4_3, AspectRatioStrategy.FALLBACK_RULE_AUTO))
                .setResolutionStrategy(ResolutionStrategy(Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                .build()
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
                    if (!processingFlag.get()) {
                        // Paused: keep the picture live, without boxes.
                        stopRecording()
                        shown = ShownFrame(upright, emptyList())
                        return@use
                    }
                    val wanted = requestedModel.get()
                    if (wanted != loadedModel) {
                        // Accelerators compile the model on creation, which can take many seconds; show the picture meanwhile.
                        shown = ShownFrame(upright, emptyList())
                        loading = true
                        val loadStart = System.nanoTime()
                        detector?.close()
                        detector = createDetector(context, wanted, initialProvider)
                        loadedModel = wanted
                        activeProvider = detector!!.provider
                        usingGpu = activeProvider.isGpu
                        fpsMonitor.reset()
                        history.clear()
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
                    shown = ShownFrame(upright, r.detections)
                    result = r
                    history.add(r.inferenceMs)
                    frameTimes = history.snapshot()
                    if (recordFlag.get()) {
                        try {
                            val rec = recorder ?: VideoRecorder(
                                context,
                                RecordingFormat.evenSize(upright.width),
                                RecordingFormat.evenSize(upright.height),
                            ).also { recorder = it }
                            rec.addFrame(upright, r.detections)
                            recordedMs = rec.elapsedMs
                        } catch (e: Exception) {
                            Log.e("Inference", "Recording failed", e)
                            runCatching { stopRecording(announce = false) }
                            recordFlag.set(false)
                            recording = false
                            toast("Recording failed")
                        }
                    } else {
                        stopRecording()
                    }
                    // Too slow on the default model: drop to the cheaper one (once, only downwards).
                    val fallback = FALLBACK_MODEL[loadedModel]
                    if (fallback != null && fpsMonitor.record(frameMs)) {
                        Log.w("Inference", "$loadedModel below $MIN_FPS FPS, switching to $fallback")
                        requestedModel.set(fallback)
                    }
                }
            }
            provider.unbindAll()
            provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            ProcessCameraProvider.getInstance(context).get().unbindAll()
            executor.shutdown()
            executor.awaitTermination(2, java.util.concurrent.TimeUnit.SECONDS)
            runCatching { stopRecording() }
            detector?.close()
        }
    }

    Box(modifier = Modifier.fillMaxSize().padding(contentPadding).background(Color.Black)) {
        shown?.let { FrameView(it, Modifier.fillMaxSize()) }
        Row(
            modifier = Modifier.align(Alignment.TopStart).padding(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            if (!loading && frameTimes.size > 1) {
                FrameTimePlot(frameTimes)
                Spacer(Modifier.width(8.dp))
            }
            Text(
                text = when {
                    loading -> "Loading…"
                    !processing -> "Paused"
                    else -> "${result.inferenceMs} ms"
                },
                color = Color.White,
                style = MaterialTheme.typography.titleMedium,
            )
        }
        if (!loading) {
            GpuBadge(usingGpu, Modifier.align(Alignment.TopEnd).padding(8.dp))
            Row(
                modifier = Modifier.align(Alignment.BottomStart).padding(16.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                ProcessingButton(processing, onClick = {
                    processing = !processing
                    processingFlag.set(processing)
                    // Pausing also ends a recording (the analyzer thread saves it on the next frame).
                    if (!processing) {
                        recordFlag.set(false)
                        recording = false
                    }
                })
                RecordButton(
                    recording = recording,
                    elapsedMs = recordedMs,
                    enabled = processing,
                    onClick = {
                        recording = !recording
                        if (recording) recordedMs = 0
                        recordFlag.set(recording)
                    },
                )
            }
        }
    }
}

/**
 * Tiny line plot of the last frame times: gray axes, two labelled ticks on the vertical axis (top = auto-scaled maximum,
 * and half of it), none on the horizontal axis. The line grows from the left until [FRAME_HISTORY] frames are shown.
 */
@Composable
private fun FrameTimePlot(times: List<Long>, modifier: Modifier = Modifier) {
    val density = LocalDensity.current
    val labelPaint = remember(density) {
        android.graphics.Paint().apply {
            color = android.graphics.Color.LTGRAY
            textSize = with(density) { 10.sp.toPx() }
            textAlign = android.graphics.Paint.Align.RIGHT
            isAntiAlias = true
        }
    }
    val axis = Color(0xFF9CA3AF)
    Canvas(
        modifier = modifier
            .size(width = 130.dp, height = 56.dp)
            .background(Color(0x66000000), RoundedCornerShape(6.dp)),
    ) {
        val top = 8.dp.toPx()
        val bottom = size.height - 6.dp.toPx()
        val left = 30.dp.toPx()
        val right = size.width - 6.dp.toPx()
        val max = FrameTimeHistory.niceMax(times.max())
        fun y(ms: Float) = bottom - (bottom - top) * ms / max

        drawLine(axis, Offset(left, top), Offset(left, bottom), strokeWidth = 1.dp.toPx())
        drawLine(axis, Offset(left, bottom), Offset(right, bottom), strokeWidth = 1.dp.toPx())
        val tick = 3.dp.toPx()
        val textOffset = labelPaint.textSize / 3f
        for (value in longArrayOf(max, max / 2)) {
            val ty = y(value.toFloat())
            drawLine(axis, Offset(left - tick, ty), Offset(left, ty), strokeWidth = 1.dp.toPx())
            drawContext.canvas.nativeCanvas.drawText("$value", left - tick - 2.dp.toPx(), ty + textOffset, labelPaint)
        }

        val path = Path()
        times.forEachIndexed { i, ms ->
            val x = left + (right - left) * i / (FRAME_HISTORY - 1)
            val py = y(ms.toFloat())
            if (i == 0) path.moveTo(x, py) else path.lineTo(x, py)
        }
        drawPath(path, Color.White, style = Stroke(width = 1.5.dp.toPx()))
    }
}

/** Round-cornered button that pauses or resumes the detection: shows "Stop" while processing and "Start" while paused. */
@Composable
private fun ProcessingButton(processing: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .background(Color(0x99000000), RoundedCornerShape(24.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            painter = painterResource(if (processing) R.drawable.ic_stop else R.drawable.ic_play),
            contentDescription = null,
            tint = Color.White,
            modifier = Modifier.size(24.dp),
        )
        Spacer(Modifier.width(8.dp))
        Text(if (processing) "Stop" else "Start", color = Color.White, style = MaterialTheme.typography.labelLarge)
    }
}

/** Starts or stops recording the video with the boxes: a red dot and "Record", or a red square and the elapsed time while recording. */
@Composable
private fun RecordButton(recording: Boolean, elapsedMs: Long, enabled: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    val red = Color(0xFFF87171)
    Row(
        modifier = modifier
            .alpha(if (enabled) 1f else 0.4f)
            .background(Color(0x99000000), RoundedCornerShape(24.dp))
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            painter = painterResource(if (recording) R.drawable.ic_stop else R.drawable.ic_record),
            contentDescription = if (recording) "Stop recording" else "Start recording",
            tint = red,
            modifier = Modifier.size(24.dp),
        )
        Spacer(Modifier.width(8.dp))
        Text(if (recording) RecordingFormat.elapsed(elapsedMs) else "Record", color = Color.White, style = MaterialTheme.typography.labelLarge)
    }
}

/** A small chip icon with a label: teal "GPU" when the model runs on the GPU, amber "No GPU" when it fell back. */
@Composable
private fun GpuBadge(usingGpu: Boolean, modifier: Modifier = Modifier) {
    val color = if (usingGpu) Color(0xFF2DD4BF) else Color(0xFFFBBF24)
    Row(
        modifier = modifier
            .background(Color(0x99000000), RoundedCornerShape(16.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            painter = painterResource(R.drawable.ic_chip),
            contentDescription = if (usingGpu) "Running on the GPU" else "Not running on the GPU",
            tint = color,
            modifier = Modifier.size(20.dp),
        )
        Spacer(Modifier.width(6.dp))
        Text(if (usingGpu) "GPU" else "No GPU", color = color, style = MaterialTheme.typography.labelLarge)
    }
}

/** An analysed camera frame (upright) and the detections computed on it. */
private class ShownFrame(val bitmap: Bitmap, val detections: List<Detection>)

/** Draws the frame letterboxed in the available space, with its boxes on top. */
@Composable
private fun FrameView(frame: ShownFrame, modifier: Modifier = Modifier) {
    val renderer = remember { DetectionRenderer() }
    val paint = remember { android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG) }

    Canvas(modifier = modifier) {
        val aspect = frame.bitmap.width.toFloat() / frame.bitmap.height
        val frameH = min(size.width / aspect, size.height)
        val frameW = frameH * aspect
        val originX = (size.width - frameW) / 2f
        val originY = (size.height - frameH) / 2f

        val canvas = drawContext.canvas.nativeCanvas
        canvas.drawBitmap(frame.bitmap, null, android.graphics.RectF(originX, originY, originX + frameW, originY + frameH), paint)
        renderer.draw(canvas, frame.detections, originX, originY, frameW, frameH)
    }
}

