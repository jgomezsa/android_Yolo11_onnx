package com.example.objectdetection

/** Name shown to the user for a model asset: `yolo11s` -> "YOLO11s", `yolo11s_480` -> "YOLO11s 480". Other names are shown as they are. */
fun modelDisplayName(model: String): String =
    if (model.startsWith("yolo")) "YOLO" + model.removePrefix("yolo").replace('_', ' ') else model
