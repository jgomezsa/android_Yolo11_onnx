package com.example.objectdetection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YoloPostProcessorTest {
    private val anchors = 4

    /** Builds a `[4 + 80, anchors]` channel-major output with the given boxes (cx, cy, w, h, class, score). */
    private fun output(vararg boxes: FloatArray): FloatArray {
        val out = FloatArray((4 + YoloPostProcessor.NUM_CLASSES) * anchors)
        boxes.forEachIndexed { i, b ->
            for (k in 0..3) out[k * anchors + i] = b[k]
            out[(4 + b[4].toInt()) * anchors + i] = b[5]
        }
        return out
    }

    @Test
    fun letterbox_wideImage_padsVertically() {
        val lb = Letterbox(1280, 640, 640)
        assertEquals(0.5f, lb.scale, 1e-6f)
        assertEquals(0f, lb.padX, 1e-6f)
        assertEquals(160f, lb.padY, 1e-6f)
    }

    @Test
    fun decode_mapsBoxBackToNormalizedSourceCoordinates() {
        val lb = Letterbox(1280, 640, 640) // scale 0.5, padY 160
        // Box covering the right half of the source: model cx=480, cy=320, w=320, h=320.
        val dets = YoloPostProcessor.decode(output(floatArrayOf(480f, 320f, 320f, 320f, 2f, 0.9f)), anchors, lb)
        assertEquals(1, dets.size)
        val d = dets[0]
        assertEquals(2, d.classId)
        assertEquals("car", d.label)
        assertEquals(0.5f, d.left, 1e-5f)
        assertEquals(1f, d.right, 1e-5f)
        assertEquals(0f, d.top, 1e-5f)
        assertEquals(1f, d.bottom, 1e-5f)
    }

    @Test
    fun decode_dropsLowConfidence() {
        val lb = Letterbox(640, 640, 640)
        val dets = YoloPostProcessor.decode(output(floatArrayOf(100f, 100f, 50f, 50f, 0f, 0.1f)), anchors, lb)
        assertTrue(dets.isEmpty())
    }

    @Test
    fun nms_suppressesOverlappingSameClassOnly() {
        val lb = Letterbox(640, 640, 640)
        val out = output(
            floatArrayOf(100f, 100f, 100f, 100f, 0f, 0.9f),
            floatArrayOf(102f, 102f, 100f, 100f, 0f, 0.8f), // same class, overlaps -> removed
            floatArrayOf(102f, 102f, 100f, 100f, 1f, 0.7f), // other class -> kept
            floatArrayOf(500f, 500f, 50f, 50f, 0f, 0.6f), // far away -> kept
        )
        val dets = YoloPostProcessor.decode(out, anchors, lb)
        assertEquals(3, dets.size)
        assertEquals(listOf(0.9f, 0.7f, 0.6f), dets.map { it.score })
    }

    @Test
    fun iou_ofIdenticalBoxesIsOne_andDisjointIsZero() {
        val a = Detection(0f, 0f, 0.5f, 0.5f, 0, 1f)
        val b = Detection(0.6f, 0.6f, 1f, 1f, 0, 1f)
        assertEquals(1f, YoloPostProcessor.iou(a, a), 1e-6f)
        assertEquals(0f, YoloPostProcessor.iou(a, b), 1e-6f)
    }
}
