package com.example.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.nio.FloatBuffer

data class DetectionResult(val detections: List<Detection>, val inferenceMs: Long)

/** Runs YOLO11n (ONNX, 640x640 input) with ONNX Runtime on the CPU. */
class YoloDetector(context: Context, modelAsset: String = "yolo11n.onnx") : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    private val inputPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val inputBuffer = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)
    private val letterboxBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    private val letterboxCanvas = Canvas(letterboxBitmap)
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val bytes = context.assets.open(modelAsset).use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) }
        session = env.createSession(bytes, options)
        inputName = session.inputNames.first()
    }

    /** [bitmap] must already be upright (rotation applied). Boxes are normalized to it. */
    fun detect(bitmap: Bitmap): DetectionResult {
        val start = System.nanoTime()
        val letterbox = Letterbox(bitmap.width, bitmap.height, INPUT_SIZE)
        preprocess(bitmap, letterbox)

        val shape = longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        OnnxTensor.createTensor(env, inputBuffer, shape).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val tensor = result[0] as OnnxTensor
                val numAnchors = tensor.info.shape[2].toInt()
                val raw = FloatArray(tensor.floatBuffer.remaining()).also { tensor.floatBuffer.get(it) }
                val detections = YoloPostProcessor.decode(raw, numAnchors, letterbox)
                return DetectionResult(detections, (System.nanoTime() - start) / 1_000_000)
            }
        }
    }

    private fun preprocess(bitmap: Bitmap, lb: Letterbox) {
        letterboxCanvas.drawColor(Color.rgb(114, 114, 114))
        val w = lb.srcWidth * lb.scale
        val h = lb.srcHeight * lb.scale
        val dst = android.graphics.RectF(lb.padX, lb.padY, lb.padX + w, lb.padY + h)
        letterboxCanvas.drawBitmap(bitmap, null, dst, bitmapPaint)
        letterboxBitmap.getPixels(inputPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // HWC uint8 ARGB -> CHW float RGB in 0..1.
        val plane = INPUT_SIZE * INPUT_SIZE
        val out = inputBuffer.array()
        for (i in 0 until plane) {
            val p = inputPixels[i]
            out[i] = ((p shr 16) and 0xFF) / 255f
            out[plane + i] = ((p shr 8) and 0xFF) / 255f
            out[2 * plane + i] = (p and 0xFF) / 255f
        }
        inputBuffer.rewind()
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val INPUT_SIZE = 640
    }
}
