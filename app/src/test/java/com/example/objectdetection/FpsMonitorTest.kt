package com.example.objectdetection

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FpsMonitorTest {
    private fun FpsMonitor.feed(frames: Int, ms: Long): List<Boolean> = List(frames) { record(ms) }

    @Test
    fun slowFrames_triggerAfterWarmupAndFullWindow() {
        val m = FpsMonitor(minFps = 3.0, warmupFrames = 3, windowFrames = 10)
        val results = m.feed(13, 400) // 2.5 FPS
        assertFalse(results.take(12).any { it })
        assertTrue(results[12])
    }

    @Test
    fun fastFrames_neverTrigger() {
        val m = FpsMonitor(minFps = 3.0, warmupFrames = 3, windowFrames = 10)
        assertFalse(m.feed(50, 250).any { it }) // 4 FPS
    }

    @Test
    fun slowWarmupFrames_areIgnored() {
        val m = FpsMonitor(minFps = 3.0, warmupFrames = 3, windowFrames = 10)
        assertFalse(m.feed(3, 5000).any { it })
        assertFalse(m.feed(10, 200).any { it })
    }

    @Test
    fun exactlyAtThreshold_doesNotTrigger() {
        val m = FpsMonitor(minFps = 3.0, warmupFrames = 0, windowFrames = 3)
        assertFalse(m.feed(3, 334 - 1).any { it }) // 333 ms/frame = 3.003 FPS
    }

    @Test
    fun reset_startsWarmupAgain() {
        val m = FpsMonitor(minFps = 3.0, warmupFrames = 2, windowFrames = 2)
        assertTrue(m.feed(4, 1000).last())
        m.reset()
        assertFalse(m.feed(2, 1000).any { it }) // warmup again
    }
}
