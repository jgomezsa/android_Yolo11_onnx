package com.example.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.FloatBuffer
import java.util.EnumSet

/** Runs a YOLO11 ONNX model with ONNX Runtime; falls back to XNNPACK, then plain CPU, if [requested] can't load. */
class YoloDetector(
    context: Context,
    modelAsset: String = "yolo11n.onnx",
    private val requested: Provider = Provider.NNAPI_FP16,
) : ObjectDetector {
    private val env = OrtEnvironment.getEnvironment()

    /** The provider actually in use (differs from [requested] after a fallback). */
    override var provider: Provider = requested
        private set
    private val session: OrtSession = createSession(context.assets.open(modelAsset).use { it.readBytes() })
    private val inputName: String = session.inputNames.first()

    /** Square input side, read from the model (`[1, 3, size, size]`), so any exported size works. */
    private val inputSize: Int = session.inputInfo.getValue(inputName).info.let { (it as TensorInfo).shape[2].toInt() }

    private val inputBuffer = FloatBuffer.allocate(3 * inputSize * inputSize)
    private val renderer = LetterboxRenderer(inputSize)

    private fun createSession(bytes: ByteArray): OrtSession {
        try {
            return env.createSession(bytes, options(requested))
        } catch (e: Exception) {
            Log.w(TAG, "$requested unavailable, falling back to XNNPACK", e)
        }
        provider = Provider.XNNPACK
        try {
            return env.createSession(bytes, options(Provider.XNNPACK))
        } catch (e: Exception) {
            Log.w(TAG, "XNNPACK unavailable, falling back to the default CPU provider", e)
        }
        return env.createSession(bytes, OrtSession.SessionOptions().apply { setIntraOpNumThreads(THREADS) })
    }

    private fun options(p: Provider) = OrtSession.SessionOptions().apply {
        when (p) {
            Provider.XNNPACK -> {
                // XNNPACK kernels use their own thread pool; ORT's intra-op pool stays at 1 to avoid oversubscription.
                setIntraOpNumThreads(1)
                addXnnpack(mapOf("intra_op_num_threads" to THREADS.toString()))
            }
            Provider.NNAPI -> {
                setIntraOpNumThreads(THREADS) // threads for the operators NNAPI does not take
                addNnapi()
            }
            Provider.NNAPI_FP16 -> {
                setIntraOpNumThreads(THREADS)
                addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            }
            Provider.LITERT_GPU -> error("not an ONNX Runtime provider")
        }
    }

    override fun detect(bitmap: Bitmap): DetectionResult {
        val t0 = System.nanoTime()
        val letterbox = Letterbox(bitmap.width, bitmap.height, inputSize)
        preprocess(bitmap, letterbox)
        val t1 = System.nanoTime()

        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        OnnxTensor.createTensor(env, inputBuffer, shape).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val t2 = System.nanoTime()
                val tensor = result[0] as OnnxTensor // [1, 84, anchors] with boxes already decoded
                val raw = FloatArray(tensor.floatBuffer.remaining()).also { tensor.floatBuffer.get(it) }
                val detections = YoloPostProcessor.decode(raw, tensor.info.shape[2].toInt(), letterbox)
                val t3 = System.nanoTime()
                return DetectionResult(
                    detections,
                    inferenceMs = (t3 - t0) / 1_000_000,
                    preprocessMs = (t1 - t0) / 1_000_000,
                    runMs = (t2 - t1) / 1_000_000,
                    postMs = (t3 - t2) / 1_000_000,
                )
            }
        }
    }

    private fun preprocess(bitmap: Bitmap, lb: Letterbox) {
        val pixels = renderer.render(bitmap, lb)

        // HWC uint8 ARGB -> CHW float RGB in 0..1.
        val plane = inputSize * inputSize
        val out = inputBuffer.array()
        for (i in 0 until plane) {
            val p = pixels[i]
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
