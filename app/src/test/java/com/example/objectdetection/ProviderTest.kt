package com.example.objectdetection

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ProviderTest {
    @Test
    fun onlyLiteRtGpu_isReportedAsGpu() {
        assertTrue(Provider.LITERT_GPU.isGpu)
        assertFalse(Provider.NNAPI.isGpu)
        assertFalse(Provider.NNAPI_FP16.isGpu)
        assertFalse(Provider.XNNPACK.isGpu)
    }
}
