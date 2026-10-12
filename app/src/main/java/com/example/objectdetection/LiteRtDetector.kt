package com.example.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Runs a YOLO11 TFLite model (NHWC float input `[1, size, size, 3]`, output `[1, 84, anchors]`) with LiteRT
 * on the GPU. The compiled GPU program is cached in the app's cache directory, so only the first start of a model
 * pays the compilation cost (about 4 s for yolo11s on a Poco X7 Pro, ~0.3 s afterwards).
 *
 * The model must have static tensor shapes: the GPU delegate rejects dynamic ones
 * (see "Models" in the README for how to convert).
 */
class LiteRtDetector(context: Context, modelAsset: String, private val family: ModelFamily = ModelFamily.YOLO11) : ObjectDetector {
    override val provider = Provider.LITERT_GPU

    private val env = Environment.create(context)
    private val model: CompiledModel
    private val inputBuffers: List<TensorBuffer>
    private val outputBuffers: List<TensorBuffer>

    /** Square input side, derived from the size of the model's input tensor. */
    private val inputSize: Int
    private val renderer: LetterboxRenderer
    private val input: FloatArray

    init {
        // The cache key includes the model size, so replacing a model file never picks up a stale compiled program.
        val modelBytes = context.assets.open(modelAsset).use { it.available() }
        val gpu = CompiledModel.GpuOptions(
            precision = CompiledModel.GpuOptions.Precision.FP16,
            serializationDir = context.cacheDir.absolutePath,
            modelCacheKey = "${modelAsset.removeSuffix(".tflite")}-$modelBytes",
            serializeProgramCache = true,
        )
        val options = CompiledModel.Options(Accelerator.GPU).apply { gpuOptions = gpu }
        model = CompiledModel.create(context.assets, modelAsset, options, env)
        inputBuffers = model.createInputBuffers()
        outputBuffers = model.createOutputBuffers()

        inputSize = sqrt(inputBuffers[0].readFloat().size / 3.0).roundToInt()
        renderer = LetterboxRenderer(inputSize)
        input = FloatArray(inputSize * inputSize * 3)
        Log.d(TAG, "$modelAsset ready on the GPU (input ${inputSize}x$inputSize)")
    }

    override fun detect(bitmap: Bitmap): DetectionResult {
        val t0 = System.nanoTime()
        val letterbox = Letterbox(bitmap.width, bitmap.height, inputSize)
        val pixels = renderer.render(bitmap, letterbox)
        // HWC float (TFLite models from onnx2tf are NHWC). YOLO11: RGB in 0..1. YOLOX: BGR in 0..255, as it was trained.
        if (family == ModelFamily.YOLOX) {
            for (i in pixels.indices) {
                val p = pixels[i]
                input[3 * i] = (p and 0xFF).toFloat()
                input[3 * i + 1] = ((p shr 8) and 0xFF).toFloat()
                input[3 * i + 2] = ((p shr 16) and 0xFF).toFloat()
            }
        } else {
            for (i in pixels.indices) {
                val p = pixels[i]
                input[3 * i] = ((p shr 16) and 0xFF) / 255f
                input[3 * i + 1] = ((p shr 8) and 0xFF) / 255f
                input[3 * i + 2] = (p and 0xFF) / 255f
            }
        }
        inputBuffers[0].writeFloat(input)
        val t1 = System.nanoTime()

        model.run(inputBuffers, outputBuffers)
        val raw = outputBuffers[0].readFloat()
        val t2 = System.nanoTime()

        val detections = when (family) {
            ModelFamily.YOLOX -> YoloxPostProcessor.decode(raw, inputSize, letterbox)
            ModelFamily.YOLO11 -> YoloPostProcessor.decode(raw, raw.size / (4 + YoloPostProcessor.NUM_CLASSES), letterbox)
        }
        val t3 = System.nanoTime()
        return DetectionResult(
            detections,
            inferenceMs = (t3 - t0) / 1_000_000,
            preprocessMs = (t1 - t0) / 1_000_000,
            runMs = (t2 - t1) / 1_000_000,
            postMs = (t3 - t2) / 1_000_000,
        )
    }

    override fun close() {
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        model.close()
        env.close()
    }

    private companion object {
        const val TAG = "LiteRtDetector"
    }
}
