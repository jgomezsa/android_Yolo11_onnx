package com.example.objectdetection

/** Model families the app can decode; the family is taken from the asset name (`yolox_s` is YOLOX, everything else YOLO11-style). */
enum class ModelFamily {
    /** Output `[1, 84, anchors]`, RGB 0..1 input. */
    YOLO11,

    /** Output `[1, 8400, 85]` (raw, with objectness), BGR 0..255 input. Apache-2.0 licensed. */
    YOLOX;

    companion object {
        fun of(model: String): ModelFamily = if (model.startsWith("yolox")) YOLOX else YOLO11
    }
}

/**
 * Name shown to the user for a model asset: `yolo11s` -> "YOLO11s", `yolo11s_480` -> "YOLO11s 480", `yolox_s` -> "YOLOX-s".
 * Other names are shown as they are.
 */
fun modelDisplayName(model: String): String = when {
    model.startsWith("yolox") -> "YOLOX" + model.removePrefix("yolox").replace('_', '-')
    model.startsWith("yolo") -> "YOLO" + model.removePrefix("yolo").replace('_', ' ')
    else -> model
}
