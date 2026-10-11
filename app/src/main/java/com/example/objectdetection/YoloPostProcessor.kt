package com.example.objectdetection

import kotlin.math.max
import kotlin.math.min

/** A detection with a box in normalized (0..1) coordinates of the source image. */
data class Detection(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val classId: Int,
    val score: Float,
) {
    val label: String get() = CocoLabels.name(classId)
}

/** Maps between a source image and the square, gray-padded model input (YOLO "letterbox"). */
class Letterbox(val srcWidth: Int, val srcHeight: Int, val size: Int) {
    val scale: Float = min(size.toFloat() / srcWidth, size.toFloat() / srcHeight)
    val padX: Float = (size - srcWidth * scale) / 2f
    val padY: Float = (size - srcHeight * scale) / 2f
}

/** Decodes the raw output of YOLOv8/YOLO11 (`[1, 4 + numClasses, numAnchors]`) into detections. */
object YoloPostProcessor {
    const val NUM_CLASSES = 80

    fun decode(
        output: FloatArray,
        numAnchors: Int,
        letterbox: Letterbox,
        confThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f,
        numClasses: Int = NUM_CLASSES,
    ): List<Detection> {
        require(output.size >= (4 + numClasses) * numAnchors) { "Unexpected output size ${output.size}" }
        val candidates = ArrayList<Detection>()
        for (i in 0 until numAnchors) {
            var best = -1
            var bestScore = confThreshold
            for (c in 0 until numClasses) {
                val s = output[(4 + c) * numAnchors + i]
                if (s > bestScore) {
                    bestScore = s
                    best = c
                }
            }
            if (best < 0) continue
            val cx = output[i]
            val cy = output[numAnchors + i]
            val w = output[2 * numAnchors + i]
            val h = output[3 * numAnchors + i]
            // Model space -> source image space (undo padding and scale), then normalize.
            val l = ((cx - w / 2 - letterbox.padX) / letterbox.scale / letterbox.srcWidth).coerceIn(0f, 1f)
            val r = ((cx + w / 2 - letterbox.padX) / letterbox.scale / letterbox.srcWidth).coerceIn(0f, 1f)
            val t = ((cy - h / 2 - letterbox.padY) / letterbox.scale / letterbox.srcHeight).coerceIn(0f, 1f)
            val b = ((cy + h / 2 - letterbox.padY) / letterbox.scale / letterbox.srcHeight).coerceIn(0f, 1f)
            if (r > l && b > t) candidates.add(Detection(l, t, r, b, best, bestScore))
        }
        return nms(candidates, iouThreshold)
    }

    /** Class-aware non-maximum suppression. */
    fun nms(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.score }
        val kept = ArrayList<Detection>()
        for (d in sorted) {
            if (kept.none { it.classId == d.classId && iou(it, d) > iouThreshold }) kept.add(d)
        }
        return kept
    }

    fun iou(a: Detection, b: Detection): Float {
        val iw = min(a.right, b.right) - max(a.left, b.left)
        val ih = min(a.bottom, b.bottom) - max(a.top, b.top)
        if (iw <= 0f || ih <= 0f) return 0f
        val inter = iw * ih
        val union = area(a) + area(b) - inter
        return if (union <= 0f) 0f else inter / union
    }

    private fun area(d: Detection) = (d.right - d.left) * (d.bottom - d.top)
}
