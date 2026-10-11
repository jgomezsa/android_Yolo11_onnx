package com.example.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import java.nio.FloatBuffer

data class DetectionResult(val detections: List<Detection>, val inferenceMs: Long)

/** Runs a YOLO11 ONNX model (640x640 input) with ONNX Runtime on the CPU, using the XNNPACK provider. */
class YoloDetector(context: Context, modelAsset: String = "yolo11n.onnx") : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = createSession(context.assets.open(modelAsset).use { it.readBytes() })
    private val inputName: String = session.inputNames.first()

    /** Square input side, read from the model (`[1, 3, size, size]`), so any exported size works. */
    private val inputSize: Int = session.inputInfo.getValue(inputName).info.let { (it as TensorInfo).shape[2].toInt() }

    private val inputPixels = IntArray(inputSize * inputSize)
    private val inputBuffer = FloatBuffer.allocate(3 * inputSize * inputSize)
    private val letterboxBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    private val letterboxCanvas = Canvas(letterboxBitmap)
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    private fun createSession(bytes: ByteArray): OrtSession = try {
        // XNNPACK kernels use their own thread pool; ORT's intra-op pool stays at 1 to avoid oversubscription.
        val xnnpack = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            addXnnpack(mapOf("intra_op_num_threads" to THREADS.toString()))
        }
        env.createSession(bytes, xnnpack)
    } catch (e: Exception) {
        Log.w(TAG, "XNNPACK unavailable, falling back to the default CPU provider", e)
        env.createSession(bytes, OrtSession.SessionOptions().apply { setIntraOpNumThreads(THREADS) })
    }

    /** [bitmap] must already be upright (rotation applied). Boxes are normalized to it. */
    fun detect(bitmap: Bitmap): DetectionResult {
        val start = System.nanoTime()
        val letterbox = Letterbox(bitmap.width, bitmap.height, inputSize)
        preprocess(bitmap, letterbox)

        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
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
        letterboxBitmap.getPixels(inputPixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // HWC uint8 ARGB -> CHW float RGB in 0..1.
        val plane = inputSize * inputSize
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
        private const val THREADS = 4
        private const val TAG = "YoloDetector"
    }
}
