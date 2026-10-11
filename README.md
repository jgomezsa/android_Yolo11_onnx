# Android YOLO11 Object Detection (GPU, LiteRT + ONNX Runtime)

Real-time object detection on an Android phone. The camera feed goes through a
[YOLO11](https://docs.ultralytics.com/models/yolo11/) model, run on-device on the **phone's GPU** with
[LiteRT](https://ai.google.dev/edge/litert) (the successor of TensorFlow Lite), and the detected boxes, labels and
confidences are drawn over the live preview.

- Default model: **YOLO11s** (small, accurate) at ~12 FPS on a Poco X7 Pro, with a **start-up under 4 s** (0.2 s once the GPU program is cached)
- If the model runs below 3 FPS (e.g. on a slower phone), the app switches itself to a cheaper **YOLO11s at 480x480**
- If the GPU cannot run the model, it falls back to ONNX Runtime with NNAPI (fp16), and then to the CPU (XNNPACK)
- 80 COCO classes (person, car, bottle, laptop, ...)
- Kotlin + Jetpack Compose + CameraX, no network access needed at runtime
- Tested on a Poco X7 Pro (Dimensity 8400-Ultra, Android 15 / HyperOS)

## What you need

| Requirement | Notes |
|---|---|
| Android phone | Android 8.0+ (API 26), with a camera and a USB cable |
| JDK 17 | Set `JAVA_HOME`. Tested with Microsoft OpenJDK 17 |
| Android SDK | `platform-tools`, `platforms;android-35`, `build-tools;35.0.0` (command-line tools are enough; Android Studio is optional) |
| Git | To clone the repo |
| Python + [uv](https://docs.astral.sh/uv/) | **Only** if you want to re-export or change the models |

Gradle 8.9 is downloaded automatically by the wrapper (`gradlew`). The project was developed on Windows 11
with PowerShell; on macOS/Linux use `./gradlew` instead of `.\gradlew.bat`.

### Install the Android SDK (command line only)

```powershell
# After installing the Android command-line tools and putting cmdline-tools\latest\bin on PATH:
sdkmanager "platform-tools" "platforms;android-35" "build-tools;35.0.0"
sdkmanager --licenses
```

Create `local.properties` in the project root pointing to your SDK (it is git-ignored). Android Studio does this for you:

```properties
sdk.dir=C\:\\Users\\<you>\\AppData\\Local\\Android\\Sdk
```

## Prepare the phone

1. **Settings > About phone**, tap **Build number** 7 times to enable Developer options.
2. **Developer options**: enable **USB debugging**.
   - On **Xiaomi / Poco / Redmi (HyperOS, MIUI)** also enable **Install via USB** and
     **USB debugging (Security settings)**, otherwise installs fail with `INSTALL_FAILED_USER_RESTRICTED`.
     Keep the phone unlocked and accept the install prompt when it appears.
3. Connect the phone over USB and accept the RSA fingerprint prompt.
4. Check it shows up as `device`:

```powershell
adb devices
```

## Build and run

```powershell
git clone https://github.com/jgomezsa/android_Yolo11_onnx.git
cd android_Yolo11_onnx

.\gradlew.bat test                 # unit tests (host JVM, no phone needed)
.\gradlew.bat assembleDebug        # -> app\build\outputs\apk\debug\app-debug.apk
.\gradlew.bat installDebug         # build + install on the connected phone
adb shell am start -n com.example.objectdetection/.MainActivity
```

You can also open the folder in Android Studio and press Run.

## Using the app

1. Launch **objectdetection** on the phone and allow the **Camera** permission.
2. Point the camera at objects. Each detection is drawn as a coloured box with `label confidence%`.
3. The top-left text shows `model · provider · number of objects · inference time (ms)`.
4. There is nothing to configure. The app runs `yolo11s` on the GPU. If it averages below 3 FPS over 10 frames it switches to `yolo11s_480`
   and shows `(auto: <3 FPS)`. If the GPU cannot run the model, the top line shows the provider actually in use (`NNAPI fp16` or `XNNPACK`).
5. The first start of a model compiles it for the GPU (about 2-4 s, "Loading ..." is shown). The compiled program is cached in the app's
   cache directory, so later starts take ~0.2 s. Clearing the app's cache brings the one-time compile back.

Developer overrides (no UI): choose the model and execution provider from the command line, e.g. to benchmark:

```powershell
adb shell am start -n com.example.objectdetection/.MainActivity --es model yolo11n --es provider XNNPACK
# provider: LITERT_GPU (default) | NNAPI_FP16 | NNAPI | XNNPACK
# model: a file name in app/src/main/assets without the extension (.tflite for LITERT_GPU, .onnx for the others)
adb logcat -s Inference:D     # per-frame time, split into preprocess / run / post, and the model load time
```

Detections below 25% confidence are hidden, and overlapping boxes of the same class are merged (NMS, IoU 0.45).
Both thresholds are defaults of `YoloPostProcessor.decode`.

## How it works

```
CameraX ImageAnalysis (RGBA frame)
  -> rotate upright -> letterbox to 640x640 (gray 114 padding) -> float RGB tensor
  -> LiteRT on the GPU (OpenCL, fp16)  ->  output [1,84,8400]  (cx, cy, w, h + 80 class scores per anchor)
  -> confidence filter + class-aware NMS -> boxes mapped back to the camera frame (normalized 0..1)
  -> Compose Canvas overlay on top of the PreviewView
```

| File | Role |
|---|---|
| `MainActivity.kt` | Edge-to-edge Compose host, camera permission request, developer launch extras |
| `CameraScreen.kt` | CameraX preview + analysis, detector fallback chain, automatic switch to the cheaper model, box overlay |
| `ObjectDetector.kt` | `ObjectDetector` interface, `Provider` enum, shared letterbox rendering |
| `LiteRtDetector.kt` | LiteRT GPU backend (`.tflite` models) with the compiled-program cache |
| `YoloDetector.kt` | ONNX Runtime backend (`.onnx` models): NNAPI fp16 / NNAPI / XNNPACK |
| `YoloPostProcessor.kt` | Letterbox math, output decoding, NMS (pure Kotlin, unit tested) |
| `FpsMonitor.kt` | Detects "average below 3 FPS" to trigger the cheaper model (pure Kotlin, unit tested) |
| `CocoLabels.kt` | The 80 COCO class names |
| `app/src/main/assets/` | The bundled models (`.tflite` for the GPU path, `.onnx` for the fallbacks) |
| `tools/` | Model conversion/quantization scripts (see below) |

## Models

The models in `app/src/main/assets/` come from the pretrained Ultralytics weights (COCO). `yolo11s.tflite` / `yolo11s_480.tflite` run on the
GPU; the `.onnx` files (`yolo11s`, `yolo11s_480`, `yolo11n`, `yolo11s_int8`) are for the ONNX Runtime fallbacks and for benchmarking.

**1. Export to ONNX** (opset 12):

```powershell
uv run --python 3.12 --with ultralytics --with onnx --with onnxslim --with onnxruntime `
  yolo export model=yolo11s.pt format=onnx imgsz=640 opset=12 simplify=True
```

**2. Convert the ONNX file to TFLite** for the GPU path. Ultralytics' own `format=litert` export only works on Linux/macOS, but `onnx2tf` runs on Windows:

```powershell
uv run --python 3.12 --with onnx2tf --with tensorflow --with tf_keras --with onnx --with onnxruntime --with onnxsim `
  --with onnx_graphsurgeon --with sng4onnx --with ai_edge_litert --with psutil --with protobuf `
  onnx2tf -i yolo11s.onnx -o out -tb tf_converter -b 1
# use out/yolo11s_float32.tflite, saved in app/src/main/assets/ as yolo11s.tflite
```

Two details matter: use `-tb tf_converter -b 1` (the default `flatbuffer_direct` backend leaves hundreds of dynamic-shape tensors and the GPU
delegate then fails with "Failed to compile model"), and use the **float32** file (the GPU computes in fp16 by itself; the float16 file has fp16 inputs).
The input size is read from the model, so other sizes work too (export ONNX with e.g. `imgsz=480`).

To add or replace a model, put the files in `app/src/main/assets/` and use its name (without extension) as `DEFAULT_MODEL` or in `FALLBACK_MODEL`
in `CameraScreen.kt`. A custom-trained model with a different number of classes needs the class names in `CocoLabels.kt`
and `YoloPostProcessor.NUM_CLASSES` updated.

### INT8 quantization (ONNX Runtime experiment)

`yolo11s_int8.onnx` was made with static post-training quantization (QDQ format, per-channel weights, 99.99th-percentile calibration on
32 COCO val2017 images). Only the Conv layers are quantized; the final classification/box convolutions and the DFL decode stay in float.
The model is 10 MB instead of 36 MB.

```powershell
# calibration images: any folder of ~40 .jpg files (the last 8 are held out for the accuracy check)
uv run --python 3.12 --with onnx --with onnxruntime --with pillow --with numpy python -I tools/quantize_int8.py `
  app/src/main/assets/yolo11s.onnx yolo11s_int8.onnx 640 percentile path/to/images
uv run --python 3.12 --with onnx --with onnxruntime --with pillow --with numpy python -I tools/eval_quantized.py `
  app/src/main/assets/yolo11s.onnx yolo11s_int8.onnx 640 path/to/images
```

On 8 held-out images the INT8 model reproduced 95% of the fp32 detections (with no extra false ones); calibrating with min/max instead of
percentile gave 93%. The script converts the model to opset 17 first, because per-channel quantization needs opset 13+.

## Performance

Measured on a Poco X7 Pro (Dimensity 8400-Ultra). Times are milliseconds per frame (preprocessing + model + decoding), averaged over
15-240 frames. "Load" is the time to create the model before the first frame:

| Backend | Model | Load | ms / frame | ~FPS |
|---|---|---|---|---|
| **LiteRT GPU (default)** | `yolo11s` 640 | **1.9-3.9 s** first start, **0.2 s** cached | **74-97** | **10-13** |
| LiteRT GPU | `yolo11s_480` | 1.9 s first start | 53 | 18 |
| ONNX Runtime, NNAPI fp16 | `yolo11s` 640 | 6.1 s | 223-245 | 4 |
| ONNX Runtime, NNAPI fp16 | `yolo11s_int8` | **19-29 s** | 89-129 | 8-11 |
| ONNX Runtime, XNNPACK (CPU) | `yolo11s` 640 | 0.3 s | 398 | 2.5 |
| ONNX Runtime, XNNPACK (CPU) | `yolo11s_480` | 0.3 s | 219 | 4.5 |
| ONNX Runtime, XNNPACK (CPU) | `yolo11n` 640 | 0.3 s | 157 | 6 |

What the measurements showed:

- **The GPU through LiteRT is the best option here**: full-precision `yolo11s` (no quantization) runs ~3x faster than on NNAPI and ~5x faster than
  on the CPU, and starts in seconds. NNAPI/ONNX Runtime has no real GPU backend: NNAPI sends the model to MediaTek's AI accelerator (APU), where the
  compile step is long (6 s for fp32, 19-29 s for INT8) and is repeated on every start (ONNX Runtime cannot cache it).
- **The CPU is not the bottleneck**: with XNNPACK the process uses ~3 cores for the model; with NNAPI or the GPU the whole app (camera, preprocessing,
  decoding, UI) uses under 1 core.
- INT8 only helps on NNAPI (fp16 flag); on the CPU it is no faster than fp32. The compile time makes it unusable as the default.
- Cutting the YOLO head (DFL decoding) out of the graph to reduce NNAPI partitions made no difference: the 11 partitions of the fp32 model come
  from other operators (Split, attention, Resize).
- MediaTek's NeuroPilot NPU accelerator for LiteRT currently lists Dimensity 7300/8300/9000/9200/9300/9400, not the 8400, and needs ahead-of-time
  compilation and Play packaging, so it was not tried.
- Time scales with the number of input pixels, so a smaller input is a cheap extra speed-up. The cost is accuracy on small objects.
- Numbers vary with scene, phone temperature and background load; treat them as ballparks. Every frame logs its time: `adb logcat -s Inference:D`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `INSTALL_FAILED_USER_RESTRICTED` | Enable **Install via USB** in Developer options (Xiaomi/Poco/Redmi) and accept the prompt on the phone |
| `adb devices` shows `unauthorized` or nothing | Re-plug the cable, accept the RSA prompt, set USB mode to File transfer |
| Black screen from `adb shell screencap` | The phone is locked or the screen is off; unlock it |
| `SDK location not found` | Create `local.properties` with `sdk.dir=...` |
| Gradle `Couldn't delete ...\R.jar` (Windows) | Something locks build files, usually the VS Code Kotlin language server or a stray Gradle daemon. Run `.\gradlew.bat --stop`, close/kill `java.exe` processes, delete `app\build`, rebuild |
| App crashes on start | `adb logcat -s AndroidRuntime:E` |
| Top line says `NNAPI fp16` or `XNNPACK` instead of `LiteRT GPU` | The GPU path failed; see the reason with `adb logcat -s Inference:W`. A TFLite model with dynamic-shape tensors gives "Failed to compile model" (re-convert as described under Models) |
| Build error `Module was compiled with an incompatible version of Kotlin` (LiteRT) | LiteRT 2.3 is built with a newer Kotlin than 2.0.21; `app/build.gradle.kts` sets `-Xskip-metadata-version-check` for it |
| Emulator is slow or fails to start | `emulator -accel-check`; the emulator camera only shows a virtual scene, a real phone is recommended |

## License

The app code is provided as-is for experimentation. The bundled models are derived from
[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), licensed under **AGPL-3.0**; commercial use
requires an Ultralytics license or a different model.
