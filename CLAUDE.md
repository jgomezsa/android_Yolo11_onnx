# Android Object Detection (Kotlin) — project guide

Goal: try an on-device object detection model on an Android phone (camera -> model -> boxes/labels).
Created on 2026-10-12 by copying `D:\Projects\android_helloworld` (the template, with the same toolchain/env
described below) and renaming the package to `com.example.objectdetection`. The template's Hello World
screen (`Greeting.kt`, `HelloScreen`) is still a placeholder to replace.

## Stack
- Kotlin 2.0.21, Android Gradle Plugin 8.7.3, Gradle 8.9 (wrapper, committed)
- JDK 17 (Microsoft OpenJDK) — `JAVA_HOME` is set as a user env var
- compileSdk/targetSdk 35, minSdk 26; UI is Jetpack Compose (Material 3, BOM 2024.10.01, Kotlin Compose compiler plugin 2.0.21)
- Unit tests: JUnit 4 in `app/src/test/` (host JVM, no device needed)
- Single module `:app`, package `com.example.objectdetection`, entry point `MainActivity.kt`

## Environment (Windows 11, set up on 2026-10-11)
- SDK at `%LOCALAPPDATA%\Android\Sdk` (`ANDROID_HOME`), installed via command-line tools only:
  platform-tools, platforms;android-35, build-tools;35.0.0, emulator, system-images;android-35;google_apis;x86_64
- User PATH includes `platform-tools`, `cmdline-tools\latest\bin`, `emulator` (open a NEW terminal after changes)
- `local.properties` (git-ignored) holds `sdk.dir`; regenerate it if the SDK moves
- AVD: `Pixel_API35` (Pixel 7, API 35, x86_64)
- VS Code extensions: `fwcd.kotlin` (language server), `mathiasfrohlich.Kotlin` (syntax)
- Shell is PowerShell; use `.\gradlew.bat` (not `./gradlew`)

## Commands
```powershell
.\gradlew.bat assembleDebug              # build -> app\build\outputs\apk\debug\app-debug.apk
.\gradlew.bat installDebug               # build + install on the connected device/emulator
.\gradlew.bat test                       # unit tests (reports in app\build\reports\tests)
adb shell am start -n com.example.objectdetection/.MainActivity   # launch the app
adb devices                              # list phones/emulators
adb logcat -s AndroidRuntime:E           # crash logs
emulator -avd Pixel_API35                # start the emulator (needs hardware acceleration)
sdkmanager --list_installed              # SDK packages; sdkmanager "<package>" to add
```
Add `--no-daemon` to Gradle if memory is tight. Prefer `installDebug` followed by `am start` as the run loop.

## Running on a physical phone
1. Settings > About phone > tap "Build number" 7 times to enable Developer options.
2. Developer options > enable "USB debugging" (or "Wireless debugging" and `adb pair`).
3. Connect via USB, accept the RSA fingerprint prompt, confirm `adb devices` shows `device`.
4. `.\gradlew.bat installDebug`, then launch from the phone or via `adb shell am start`.

## Emulator prerequisite
The x86_64 emulator needs CPU virtualization. Initially it was off; it is now enabled
(BIOS virtualization + Windows Hypervisor Platform) and `emulator -accel-check` reports
"WHPX is installed and usable". If the emulator ever fails to start, re-run that check first.

## Conventions for future work
- Keep versions in the Gradle Kotlin DSL files; don't introduce Android Studio-only files (`.idea/` is ignored).
- After any code change, verify with `.\gradlew.bat test assembleDebug` before saying it works.
- Add a unit test in `app/src/test/` for any new logic.
- UI is edge-to-edge: `enableEdgeToEdge()` in `onCreate`, content inside a `Scaffold` and the `innerPadding` applied to the root composable (keeps content clear of status/navigation bars, and system bar icons stay visible).
- To screenshot the emulator: `adb shell screencap -p /sdcard/s.png` then `adb pull /sdcard/s.png` (don't use PowerShell `>` redirection, it corrupts binary).
- Put new Kotlin code under `app/src/main/java/com/example/objectdetection/`.
- Release signing is not configured; only debug builds exist so far.

## Status
- [x] Camera (CameraX 1.4.0 + CAMERA permission), inference on frames, box/label overlay; verified on the Poco X7 Pro (HyperOS) on 2026-10-12.
      If `installDebug` gives INSTALL_FAILED_USER_RESTRICTED: Developer options > "Install via USB", accept the prompt on the phone.
- [x] Backend: **LiteRT 2.3.0 on the GPU** (`LiteRtDetector.kt`, `yolo11s.tflite`, fp16, compiled-program cache in cacheDir). Default model `yolo11s`;
      `FpsMonitor` (unit tested) switches to `yolo11s_480` below 3 FPS. Fallback chain (`createDetector` in `CameraScreen.kt`): LiteRT GPU -> ORT NNAPI fp16 -> XNNPACK.
- [x] UI (user's requirements): no model/provider buttons; the top-left text shows only the frame processing time (or "Loading…"); the user is never told
      which model/provider is used (it is only in logcat). Load time must be < 5 s (met: 2-4 s cold, 0.2 s cached).
- [x] Pruned models: assets keep `yolo11s.tflite`, `yolo11s_480.tflite` (GPU) and `yolo11s.onnx`, `yolo11s_480.onnx` (ORT fallbacks). Removed (still in git history):
      `yolo11n.onnx`, `yolo11s_int8.onnx`, `yolo11s_384`, the DFL-cut models.
- [x] Launcher icon: adaptive icon (two detection boxes with pointed label tags, a person and a ball inside) generated by `tools/make_icon.py`
      (`uv run --python 3.12 --with pillow python tools/make_icon.py app/src/main/res preview.png` rewrites the 3 drawables and writes a preview with circle/squircle masks;
      keep all shapes within 33 dp of the centre = safe zone). The mipmap-anydpi-v26 XMLs and the manifest entries are hand-written.
- [ ] Ideas: pipeline preprocessing (~8 ms of ~75) with inference; INT8/fp16 TFLite for the GPU.
- [ ] Before publishing: licensing (see "Licensing" below), release signing, privacy policy (camera), proper app name/package id.

## README policy
README.md is for the users of the repo (requirements, setup, build/run, how the app works internally, license). Measurements, benchmark/dev commands and
conversion commands live HERE, not in the README.

## Measurements (Poco X7 Pro, Dimensity 8400-Ultra; logcat tag `Inference`, ms/frame = preprocess + model + decode)
| Backend | Model | Load | ms/frame |
|---|---|---|---|
| LiteRT GPU (default) | yolo11s 640 | 1.9-3.9 s cold, 0.2 s cached | 74-97 (~10-13 FPS) |
| LiteRT GPU | yolo11s_480 | 1.9 s cold | 53 |
| ORT NNAPI fp16 | yolo11s 640 | 6.1 s | 223-245 |
| ORT NNAPI fp16 | yolo11s INT8 (QDQ, Conv only, percentile calibration) | 19-29 s | 89-129 |
| ORT NNAPI fp32 | yolo11s | ~6 s | 358 |
| ORT XNNPACK (CPU) | yolo11s / yolo11s_480 / yolo11n | 0.3 s | 398 / 219 / 157 |
- NNAPI runs on MediaTek's APU (logs show `apusys`), not on the CPU or GPU; ORT for Android has no GPU backend. XNNPACK burns ~3 cores, NNAPI/GPU < 1 core
  (measure with /proc/<pid>/stat utime+stime over 10 s).
- NNAPI compile time is paid on every start (ORT cannot cache it): 6 s fp32, 19-29 s INT8 -> unusable under the 5 s rule. INT8 reproduced 95% of fp32 detections on
  host (script in git history: `tools/quantize_int8.py`, commit 1821ae6). XNNPACK vs default CPU: no difference.
- Cutting the head (DFL decode) out of the graph to reduce NNAPI partitions: no gain (11 partitions come from Split/attention/Resize). Code removed.
- MediaTek NeuroPilot via LiteRT: supported SoC list is 7300/8300/9000/9200/9300/9400 (not 8400) and needs AOT compile + Play packaging; not tried.
- Per-frame breakdown on the GPU path: preprocess ~8 ms, model ~60-70 ms, decode ~3 ms.

## Models and conversion commands
- ONNX export (opset 12): `uv run --python 3.12 --with ultralytics --with onnx --with onnxslim --with onnxruntime yolo export model=yolo11s.pt format=onnx imgsz=640 opset=12 simplify=True`
  (`imgsz=480` for the 480 model; input `[1,3,S,S]` RGB 0..1 letterboxed with gray 114, output `[1,84,N]` = cx,cy,w,h + 80 COCO scores, decoded in `YoloPostProcessor`).
- ONNX -> TFLite for the GPU (Ultralytics `format=litert` only works on Linux/macOS; `onnx2tf` works on Windows):
  `uv run --python 3.12 --with onnx2tf --with tensorflow --with tf_keras --with onnx --with onnxruntime --with onnxsim --with onnx_graphsurgeon --with sng4onnx --with ai_edge_litert --with psutil --with protobuf onnx2tf -i yolo11s.onnx -o out -tb tf_converter -b 1`
  then use `out/yolo11s_float32.tflite` as `yolo11s.tflite`. MUST use `-tb tf_converter -b 1`: the default `flatbuffer_direct` backend leaves 331 dynamic-shape tensors
  and the GPU delegate fails with "Failed to compile model". Use float32 (the GPU computes in fp16 itself; the float16 file has fp16 inputs). TFLite input is NHWC.
  Check a converted model on the host against the ONNX one (max abs diff was 0.008) with `ai_edge_litert.interpreter.Interpreter`.
- Dev override (benchmarking, no UI): `adb shell am start -n com.example.objectdetection/.MainActivity --es model yolo11s_480 --es provider XNNPACK`
  (provider: LITERT_GPU | NNAPI_FP16 | NNAPI | XNNPACK; model = asset name without extension). Per-frame logs: `adb logcat -s Inference:D` (includes "loaded <model> in N ms").
- Cold-start test: `adb shell run-as com.example.objectdetection sh -c "rm -rf cache/*"` clears the GPU program cache.

## Licensing (YOLO11 / Ultralytics) — for the "can we publish this?" question
- Ultralytics YOLO11 code AND the pretrained weights are AGPL-3.0, or a paid Ultralytics Enterprise License. AGPL does not forbid commercial use, but anything that
  incorporates the weights/code must be released under AGPL with its source offered to users; a closed-source Play Store app needs the Enterprise License (or a different model).
- Permissively licensed alternatives to evaluate (verify the exact weights license first): RT-DETR (Apache-2.0), YOLOX (Apache-2.0), PP-YOLOE / D-FINE (Apache-2.0),
  EfficientDet-Lite and SSD-MobileNet (Apache-2.0, TFLite/MediaPipe), RF-DETR (Apache-2.0 for most sizes). Avoid YOLOv5/8/10/11 (AGPL), YOLOv7/v9 (GPL-3.0), YOLO-NAS (non-commercial weights).
- Other components: CameraX, LiteRT (Apache-2.0), ONNX Runtime (MIT), Compose (Apache-2.0). COCO labels/annotations: CC BY 4.0.

## Gotchas
- Gradle `Couldn't delete ...R.jar` on Windows: the VS Code Kotlin language server / stray Gradle daemons lock build files. Kill all `java.exe` and delete
  `app\build\intermediates\compile_and_runtime_not_namespaced_r_class_jar`, then rebuild (the language server respawns, so do both in one command).
- LiteRT 2.3 is built with a newer Kotlin than 2.0.21: `app/build.gradle.kts` sets `-Xskip-metadata-version-check`. IDE diagnostics about "incompatible version of Kotlin" are noise; Gradle is the truth.
- IDE diagnostics right after an edit are often stale; build to confirm.
- Adb screenshots are black when the phone is locked; the phone's home screens contain personal apps, do not browse them.
- Remote: https://github.com/jgomezsa/android_Yolo11_onnx
