# Technical documentation

Everything a developer needs to build, understand, measure and change the app. The [README](README.md) is the user-facing introduction.

## 1. Overview

An Android app (Kotlin, Jetpack Compose) that runs YOLO11 object detection on live camera frames, entirely on the device, and draws boxes with labels over the
camera preview. The default path runs `yolo11s` on the phone's **GPU through LiteRT**; ONNX Runtime (NNAPI, then XNNPACK on the CPU) is kept as a fallback.

- Package `com.example.objectdetection`, single module `:app`, entry point `MainActivity.kt`
- Remote: https://github.com/jgomezsa/android_Yolo11_onnx
- Tested on a Poco X7 Pro (Dimensity 8400-Ultra, Mali GPU, MediaTek APU, Android 15 / HyperOS)

## 2. Stack and toolchain

| Item | Version / note |
|---|---|
| Kotlin | 2.0.21 (`-Xskip-metadata-version-check` is set because LiteRT 2.3.0 is built with a newer Kotlin) |
| Android Gradle Plugin / Gradle | 8.7.3 / 8.9 (wrapper, committed) |
| JDK | 17 (Microsoft OpenJDK), `JAVA_HOME` set as a user env var |
| compileSdk / targetSdk / minSdk | 35 / 35 / 26 (adaptive icons need no PNG fallbacks) |
| UI | Jetpack Compose, Material 3, BOM 2024.10.01 |
| Camera | CameraX 1.4.0 (`camera-core`, `-camera2`, `-lifecycle`, `-view`) |
| GPU inference | LiteRT `com.google.ai.edge.litert:litert` and `litert-gpu` 2.3.0 |
| Fallback inference | ONNX Runtime Android 1.20.0 (`onnxruntime-android`) |
| Unit tests | JUnit 4 in `app/src/test/` (host JVM, no device) |

Versions live in the Gradle Kotlin DSL files. Android Studio-only files are not used (`.idea/` is ignored).

## 3. Development environment (Windows 11)

- Android SDK at `%LOCALAPPDATA%\Android\Sdk` (`ANDROID_HOME`), installed with the command-line tools only: platform-tools, `platforms;android-35`,
  `build-tools;35.0.0`, emulator, `system-images;android-35;google_apis;x86_64`.
- User PATH includes `platform-tools`, `cmdline-tools\latest\bin`, `emulator` (open a new terminal after changes).
- `local.properties` (git-ignored) holds `sdk.dir`; regenerate it if the SDK moves.
- AVD `Pixel_API35` (Pixel 7, API 35, x86_64). The emulator needs CPU virtualization (BIOS + Windows Hypervisor Platform);
  `emulator -accel-check` should report "WHPX is installed and usable". The emulator camera only shows a virtual scene; a real phone is needed for real tests.
- VS Code extensions: `fwcd.kotlin` (language server) and `mathiasfrohlich.Kotlin` (syntax).
- Python tooling (model export, conversion, icon) runs through `uv` (`uv run --python 3.12 --with <packages> ...`); no global Python.
- Shell is PowerShell; use `.\gradlew.bat` (not `./gradlew`).

### Running on a physical phone

1. Settings > About phone > tap "Build number" 7 times to enable Developer options.
2. Developer options > "USB debugging" (or "Wireless debugging" and `adb pair`). On Xiaomi/Poco/Redmi also enable "Install via USB" and
   "USB debugging (Security settings)", keep the phone unlocked and accept the install prompt (otherwise `INSTALL_FAILED_USER_RESTRICTED`).
3. Connect via USB, accept the RSA prompt, check `adb devices` shows `device`.
4. `.\gradlew.bat installDebug`, then launch from the phone or `adb shell am start -n com.example.objectdetection/.MainActivity`.

## 4. Commands

```powershell
.\gradlew.bat test assembleDebug         # verify after any change (unit tests + debug APK)
.\gradlew.bat installDebug               # build + install on the connected device/emulator
adb shell am start -n com.example.objectdetection/.MainActivity
adb devices                              # list phones/emulators
adb logcat -s AndroidRuntime:E           # crash logs
adb logcat -s Inference:D                # per-frame timing, model load time, fallbacks
emulator -avd Pixel_API35                # start the emulator
sdkmanager --list_installed              # SDK packages
```

Add `--no-daemon` to Gradle if memory is tight. Screenshot: `adb shell screencap -p /sdcard/s.png` then `adb pull /sdcard/s.png`
(PowerShell `>` redirection corrupts binary). The screenshot is black when the phone is locked or its screen is off.

Developer overrides (no UI), to benchmark a model/provider combination:

```powershell
adb shell am start -n com.example.objectdetection/.MainActivity --es model yolo11s_480 --es provider XNNPACK
# provider: LITERT_GPU (default) | NNAPI_FP16 | NNAPI | XNNPACK ; model: asset name without extension (.tflite for LITERT_GPU, .onnx for the others)
adb shell run-as com.example.objectdetection sh -c "rm -rf cache/*"     # clear the GPU program cache (cold-start test)
```

## 5. Architecture

### 5.1 Frame pipeline

```
CameraX ImageAnalysis (RGBA_8888, STRATEGY_KEEP_ONLY_LATEST, 4:3)
  -> Bitmap, rotated upright
  -> LetterboxRenderer: scale to fit a square (gray 114 padding) -> ARGB pixels
  -> detector-specific tensor (NHWC float for LiteRT, NCHW float for ONNX Runtime), RGB 0..1
  -> model -> [1, 84, N]  (cx, cy, w, h in model pixels + 80 class scores per anchor)
  -> YoloPostProcessor.decode: best class per anchor, score > 0.25, class-aware NMS (IoU 0.45), map back to normalized source coordinates
  -> Compose Canvas overlay above the PreviewView (FIT_CENTER; the overlay computes the same fitted rectangle)
```

Analysis and preview both use a 4:3 `ResolutionSelector`, so normalized coordinates of the analysis frame map directly onto the fitted preview rectangle.
Everything in the analyzer runs on one single-thread executor; results are published to Compose state.

### 5.2 Components (`app/src/main/java/com/example/objectdetection/`)

| File | Role |
|---|---|
| `MainActivity.kt` | Edge-to-edge host (`enableEdgeToEdge()`, `Scaffold` with `innerPadding` on the root), CAMERA permission gate, reads the `model` / `provider` launch extras |
| `CameraScreen.kt` | CameraX setup, analyzer loop, `createDetector` fallback chain, automatic switch to the lighter model, overlay drawing, top-left frame time and `FrameTimePlot`, top-right `GpuBadge` |
| `FrameTimeHistory.kt` | Ring buffer of the last 50 frame times plus `niceMax` (auto-scaled top tick of the plot; pure Kotlin, unit tested) |
| `ObjectDetector.kt` | `ObjectDetector` interface, `Provider` enum (`LITERT_GPU`, `NNAPI_FP16`, `NNAPI`, `XNNPACK`; `isGpu`), `DetectionResult` (with phase timings), `LetterboxRenderer` |
| `LiteRtDetector.kt` | LiteRT `CompiledModel` on the GPU (fp16 precision, program cache); input size derived from the model's input tensor |
| `YoloDetector.kt` | ONNX Runtime session with NNAPI fp16 / NNAPI / XNNPACK options; input size read from the model |
| `YoloPostProcessor.kt` | `Letterbox` math, `decode`, `nms`, `iou` (pure Kotlin, unit tested) |
| `FpsMonitor.kt` | Average-FPS-below-threshold detector (3 FPS, 3 warm-up frames, 10-frame window; pure Kotlin, unit tested) |
| `CocoLabels.kt` | The 80 COCO class names |

### 5.3 Backends and fallbacks

`createDetector(context, model, provider)` in `CameraScreen.kt`:

1. `LITERT_GPU` (default): `LiteRtDetector("<model>.tflite")`. Needs a static-shape TFLite model.
2. If creating it throws: `YoloDetector("<model>.onnx", NNAPI_FP16)`.
3. If NNAPI fails inside `YoloDetector`: XNNPACK, then the default CPU provider.

The chosen provider is kept in `activeProvider` (logged) and in the `usingGpu` state (shown by the badge). The model is only ever named in logcat.

**GPU program cache.** `CompiledModel.GpuOptions(precision = FP16, serializationDir = cacheDir, modelCacheKey = "<name>-<asset size>", serializeProgramCache = true)`.
The first start of a model compiles OpenCL programs (about 2-4 s for yolo11s); later starts load them from `cacheDir` (about 0.2 s). The key includes the asset size so a
replaced model never reuses a stale cache. The OS may clear the cache directory, which just brings the one-time compile back.

**Automatic fallback model.** The analyzer measures the whole per-frame time (including bitmap conversion). If `FpsMonitor` reports an average below 3 FPS while
`yolo11s` is loaded, the analyzer reloads `yolo11s_480` (`FALLBACK_MODEL`). It only goes down, never back up. The monitor is reset whenever a model is (re)loaded
and ignores warm-up frames.

### 5.4 User interface rules

- Top-left: the frame processing time in ms (`result.inferenceMs`, i.e. preprocess + model + decode), or "Loading…" while the detector is being created.
- Under the time: `FrameTimePlot`, a small line plot of the last 50 frame times (`FRAME_HISTORY`). Gray axes, two labelled ticks on the vertical axis (auto-scaled
  maximum from `FrameTimeHistory.niceMax`, and half of it), no horizontal ticks. The history is cleared when a model is (re)loaded.
- Top-right: `GpuBadge`, shown after loading: a chip icon (`res/drawable/ic_chip.xml`, tinted) with "GPU" (teal) when the model runs on the GPU or "No GPU" (amber) otherwise.
- The user is never shown which model or provider is used, and there are no model/provider controls.
- Edge-to-edge with `Scaffold`; `innerPadding` is applied to the root composable.

### 5.5 Logging

Tag `Inference`: `loaded <model> <PROVIDER> in N ms` once per load, and per frame `<model> <PROVIDER> T ms (pre a, run b, post c) frame F ms`.
Warnings for provider fallbacks and for the automatic switch to the lighter model.

## 6. Models

Bundled in `app/src/main/assets/` (about 145 MB in total; each file is under GitHub's 100 MB limit):

| File | Used for |
|---|---|
| `yolo11s.tflite`, `yolo11s_480.tflite` | LiteRT GPU path (default and lighter model) |
| `yolo11s.onnx`, `yolo11s_480.onnx` | ONNX Runtime fallbacks (NNAPI / XNNPACK) |

Models come from the pretrained Ultralytics COCO weights. Model contract: input RGB 0..1 letterboxed with gray 114; output `[1, 84, N]` = cx, cy, w, h (model pixels) +
80 class scores (already sigmoid-ed); `N` = 8400 at 640, 4725 at 480.

### 6.1 Export to ONNX (opset 12)

```powershell
uv run --python 3.12 --with ultralytics --with onnx --with onnxslim --with onnxruntime `
  yolo export model=yolo11s.pt format=onnx imgsz=640 opset=12 simplify=True      # imgsz=480 for the lighter model
```

### 6.2 ONNX to TFLite for the GPU

Ultralytics' `format=litert` only works on Linux/macOS; `onnx2tf` works on Windows:

```powershell
uv run --python 3.12 --with onnx2tf --with tensorflow --with tf_keras --with onnx --with onnxruntime --with onnxsim `
  --with onnx_graphsurgeon --with sng4onnx --with ai_edge_litert --with psutil --with protobuf `
  onnx2tf -i yolo11s.onnx -o out -tb tf_converter -b 1
# use out/yolo11s_float32.tflite as app/src/main/assets/yolo11s.tflite
```

- **`-tb tf_converter -b 1` is mandatory.** The default `flatbuffer_direct` backend leaves 331 dynamic-shape tensors and the GPU delegate fails with
  "Failed to compile model" ("a delegate that only supports static-sized tensors ... dynamic-sized tensor").
- Use the **float32** file: the GPU computes in fp16 by itself, and the float16 file has fp16 inputs/outputs.
- The TFLite input is NHWC `[1, S, S, 3]`; the output keeps `[1, 84, N]`.
- Check a converted model on the host against the ONNX one with `ai_edge_litert.interpreter.Interpreter` (the max absolute difference was 0.008 on a test image).

### 6.3 Adding or replacing a model

Put the files in `assets/` and use the name (without extension) as `DEFAULT_MODEL` / in `FALLBACK_MODEL` (`CameraScreen.kt`). A model with a different number of classes needs
`CocoLabels.kt` and `YoloPostProcessor.NUM_CLASSES` updated. Input size is read from the model, so other sizes (multiples of 32) work.

### 6.4 Tried and discarded

- `yolo11n` and `yolo11s_384`: removed (not needed; 384 was judged not worth the accuracy loss).
- **INT8 (QDQ, Conv-only, percentile calibration on 32 COCO val images, opset 17)**: reproduced 95% of the fp32 detections on host and ran at 89-129 ms on NNAPI, but
  NNAPI compiles it for 19-29 s on every start. The scripts are in git history (`tools/quantize_int8.py`, `tools/eval_quantized.py`, commit 1821ae6).
  Per-channel quantization needs opset 13+, so the script converted the model to opset 17 first.
- **Cutting the DFL decode out of the graph** (to reduce NNAPI partitions): no gain, the 11 partitions come from Split/attention/Resize ops. Code removed.
- **MediaTek NeuroPilot via LiteRT**: the supported SoC list is Dimensity 7300/8300/9000/9200/9300/9400 (not the 8400) and it needs ahead-of-time compilation and Play packaging.

## 7. Measurements (Poco X7 Pro)

Milliseconds per frame = preprocess + model + decode, from the `Inference` log; "load" is the time to create the model.

| Backend | Model | Load | ms / frame |
|---|---|---|---|
| **LiteRT GPU (default)** | yolo11s 640 | 1.9-3.9 s cold, 0.2 s cached | 63-97 (~10-15 FPS) |
| LiteRT GPU | yolo11s_480 | 1.9 s cold | 53 |
| ORT NNAPI fp16 | yolo11s 640 | 6.1 s | 223-245 |
| ORT NNAPI fp16 | yolo11s INT8 | 19-29 s | 89-129 |
| ORT NNAPI fp32 | yolo11s | ~6 s | 358 |
| ORT XNNPACK (CPU) | yolo11s / yolo11s_480 / yolo11n | 0.3 s | 398 / 219 / 157 |

- On the GPU path the frame breaks down as preprocess ~8 ms, model ~60-70 ms, decode ~3 ms.
- ONNX Runtime for Android has no GPU backend. NNAPI sent the model to MediaTek's APU (the logs show `apusys`), not to the GPU or CPU; its compile step is paid on every start
  because ORT cannot cache it.
- CPU use: XNNPACK burns about 3 cores for the model; with NNAPI or the GPU the whole app (camera, preprocessing, decoding, UI) uses under 1 core
  (measured from `/proc/<pid>/stat` utime+stime over 10 s).
- XNNPACK vs the default ORT CPU provider made no difference.
- Requirement from the product owner: model load time must be under 5 s on every launch after the first. The first (cold) load may take longer if later loads
  are fast thanks to a cache. The GPU path caches its compiled programs (cold 2-4 s, cached 0.2 s); NNAPI via ORT cannot cache, so it pays its 6-29 s on every start.
- Numbers vary with scene, temperature and background load; the phone was often charging and warm.

## 8. Icons

### 8.1 Launcher icon

Adaptive icon (API 26+, no PNGs): `ic_launcher_background.xml` (dark gradient), `ic_launcher_foreground.xml` (two detection boxes, teal and amber, each with a pointed
label tag containing a text bar; a person inside the teal box and a ball inside the amber one) and `ic_launcher_monochrome.xml` (themed icons, Android 13+).
`mipmap-anydpi-v26/ic_launcher.xml` and `ic_launcher_round.xml` and the manifest entries are hand-written.

The three drawables are generated from one geometry description by `tools/make_icon.py`, which also renders a preview with circle / rounded-square masks and reports how far the
artwork reaches from the centre (it must stay within 33 dp: the adaptive-icon safe zone is a 66 dp circle):

```powershell
uv run --python 3.12 --with pillow python tools/make_icon.py app/src/main/res preview.png
```

Design notes: a plain tab above a box reads as a folder, so the tags are pointed (price-tag shape); objects inside the boxes make it read as "detections".

### 8.2 GPU badge icon

`res/drawable/ic_chip.xml` is a hand-drawn processor chip (24 dp viewport, white, tinted in Compose).

## 9. Testing

`.\gradlew.bat test` runs the JVM unit tests: `YoloPostProcessorTest` (letterbox, decode, NMS, IoU), `FpsMonitorTest` (threshold, warm-up, reset), `FrameTimeHistoryTest` (window, `niceMax`), `ProviderTest` (`isGpu`).
Detection quality and speed are checked on the phone with `adb logcat` and screenshots. Add a unit test for any new logic.

## 10. Licensing

- Ultralytics YOLO11 code and the pretrained weights are **AGPL-3.0**, or a paid Ultralytics Enterprise License. AGPL does not forbid commercial use, but anything that
  incorporates the weights or code must be released under AGPL with its source offered to users. A closed-source Play Store app needs the Enterprise License or a different model.
- Permissively licensed alternatives to evaluate (verify the exact weights licence first): RT-DETR, YOLOX, PP-YOLOE, D-FINE (Apache-2.0), EfficientDet-Lite and SSD-MobileNet
  (Apache-2.0, TFLite/MediaPipe), RF-DETR (Apache-2.0 for most sizes). Avoid YOLOv5/8/10/11 (AGPL), YOLOv7/v9 (GPL-3.0), YOLO-NAS (non-commercial weights).
- Other components: CameraX, LiteRT, Compose (Apache-2.0); ONNX Runtime (MIT). COCO annotations: CC BY 4.0.
- Not legal advice; check the current terms before publishing.

## 11. Known issues and gotchas

- **Gradle `Couldn't delete ...\R.jar` (Windows).** The VS Code Kotlin language server and stray Gradle daemons lock build files. Kill all `java.exe` processes and delete
  `app\build\intermediates\compile_and_runtime_not_namespaced_r_class_jar`, then rebuild; the language server respawns, so do both in one command.
- **IDE diagnostics are unreliable.** The language server does not know about `-Xskip-metadata-version-check` ("incompatible version of Kotlin" on LiteRT classes) and is often stale right
  after an edit. Gradle is the truth.
- **LiteRT 2.3 and Kotlin.** It is built with a newer Kotlin than 2.0.21; compiling works only with the skip-metadata flag in `app/build.gradle.kts`. Upgrading the Kotlin toolchain
  would remove the flag.
- **Dynamic-shape TFLite models** make the GPU path fail and silently fall back to NNAPI; the reason is in `adb logcat -s Inference:W`.
- **First start.** The cold GPU compile takes a few seconds ("Loading…" is shown); a cleared cache brings it back.
- Release signing is not configured; only debug builds exist.
