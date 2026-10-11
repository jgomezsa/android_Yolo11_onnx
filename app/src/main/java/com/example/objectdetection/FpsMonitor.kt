package com.example.objectdetection

/**
 * Tracks per-frame processing time and reports when the average frame rate over a window of frames
 * falls below [minFps]. The first [warmupFrames] frames are ignored (the NNAPI driver is slow at first).
 */
class FpsMonitor(
    private val minFps: Double = 3.0,
    private val warmupFrames: Int = 3,
    private val windowFrames: Int = 10,
) {
    private var seen = 0
    private var count = 0
    private var totalMs = 0L

    /** Records one frame that took [frameMs]; returns true when a full window averaged below [minFps]. */
    fun record(frameMs: Long): Boolean {
        seen++
        if (seen <= warmupFrames) return false
        totalMs += frameMs
        count++
        if (count < windowFrames) return false
        val fps = if (totalMs <= 0L) Double.MAX_VALUE else 1000.0 * count / totalMs
        count = 0
        totalMs = 0L
        return fps < minFps
    }

    fun reset() {
        seen = 0
        count = 0
        totalMs = 0L
    }
}
