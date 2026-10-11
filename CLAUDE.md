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
- [x] Model/runtime chosen: YOLO11n in ONNX format + ONNX Runtime Mobile (`onnxruntime-android`, CPU, 4 threads)
- [x] Camera access (CameraX 1.4.0 + runtime CAMERA permission, `MainActivity.kt`)
- [x] Inference on camera frames + box/label overlay (`CameraScreen.kt`, `YoloDetector.kt`, `YoloPostProcessor.kt`); unit tests pass
- [ ] Verify on the physical phone (Poco X7 Pro, HyperOS): install was blocked with INSTALL_FAILED_USER_RESTRICTED.
      Enable Developer options > "Install via USB" (and "USB debugging (Security settings)"), keep the phone unlocked, retry `installDebug`.
- [ ] Try NNAPI/XNNPACK providers, INT8 quantization, YOLO11s

## Model
- `app/src/main/assets/yolo11n.onnx` (10 MB), exported with:
  `uv run --python 3.12 --with ultralytics --with onnx --with onnxslim --with onnxruntime yolo export model=yolo11n.pt format=onnx imgsz=640 opset=12 simplify=True`
- Input `[1,3,640,640]` RGB 0..1 (letterboxed, gray 114); output `[1,84,8400]` = cx,cy,w,h + 80 COCO class scores. Decoding + NMS in `YoloPostProcessor`.
- Ultralytics YOLO is AGPL-3.0 (commercial use needs a license).

## Gotchas
- Gradle `Couldn't delete ...R.jar` on Windows: the VS Code Kotlin language server / stray Gradle daemons lock build files. Kill `java.exe` processes and delete the locked dir, then rebuild.
- Remote: https://github.com/jgomezsa/android_Yolo11_onnx
