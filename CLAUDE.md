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
- [x] Project copied from the Hello World template, renamed (package, app name, project name)
- [x] Camera (CameraX 1.4.0 + CAMERA permission), inference on frames, box/label overlay; verified on the Poco X7 Pro (HyperOS) on 2026-10-12.
      If `installDebug` gives INSTALL_FAILED_USER_RESTRICTED: Developer options > "Install via USB", accept the prompt on the phone.
- [x] Backend: **LiteRT 2.3.0 on the GPU** (`LiteRtDetector.kt`, `yolo11s.tflite`, fp16, compiled-program cache in cacheDir). Default model `yolo11s`;
      `FpsMonitor` (unit tested) switches to `yolo11s_480` below 3 FPS. Fallback chain (`createDetector` in `CameraScreen.kt`): LiteRT GPU -> ORT NNAPI fp16 -> XNNPACK.
      No model/provider buttons. Dev override: `adb shell am start ... --es model <name> --es provider <LITERT_GPU|NNAPI_FP16|NNAPI|XNNPACK>`.
- [x] Measured on the Poco (ms/frame): LiteRT GPU s@640 74-97 (load 1.9-3.9 s cold, 0.2 s cached), s@480 53; ORT NNAPI fp16 s 223-245 (load 6 s);
      NNAPI INT8 89-129 (load 19-29 s, unusable); XNNPACK s 398, n 157. Requirement from the user: load time must be < 5 s.
- [x] Tried and dropped: yolo11s_384, cutting the DFL head out of the graph (no fewer NNAPI partitions), NNAPI INT8 as default (compile time).
      MediaTek NeuroPilot via LiteRT: Dimensity 8400 not in the supported list (7300/8300/9000/9200/9300/9400) and needs AOT + Play packaging; not tried.
- [ ] Ideas: pipeline preprocessing (~8 ms) with inference; INT8/fp16 TFLite for the GPU; skip the 4-core CPU preprocessing loop with a GPU/RenderScript-free resize.
- Notes: LiteRT 2.3 needs `-Xskip-metadata-version-check` (built with a newer Kotlin than 2.0.21). The TFLite must have static shapes: convert with
  `onnx2tf -tb tf_converter -b 1` (see README "Models"). Bundled models total ~170 MB (each file < 100 MB); consider pruning/Git LFS.

## Model
- ONNX models in `app/src/main/assets/` (and the `.tflite` GPU models converted from them, see README), exported with:
  `uv run --python 3.12 --with ultralytics --with onnx --with onnxslim --with onnxruntime yolo export model=yolo11s.pt format=onnx imgsz=640 opset=12 simplify=True`
- Input `[1,3,640,640]` RGB 0..1 (letterboxed, gray 114); output `[1,84,8400]` = cx,cy,w,h + 80 COCO class scores. Decoding + NMS in `YoloPostProcessor`.
- Ultralytics YOLO is AGPL-3.0 (commercial use needs a license).

## Gotchas
- Gradle `Couldn't delete ...R.jar` on Windows: the VS Code Kotlin language server / stray Gradle daemons lock build files. Kill `java.exe` processes and delete the locked dir, then rebuild.
- Remote: https://github.com/jgomezsa/android_Yolo11_onnx
