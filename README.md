# Android YOLO11 Object Detection (ONNX Runtime)

Real-time object detection on an Android phone. The camera feed goes through a
[YOLO11](https://docs.ultralytics.com/models/yolo11/) model in ONNX format, run on-device with
[ONNX Runtime Mobile](https://onnxruntime.ai/docs/get-started/with-android.html) (CPU, XNNPACK provider),
and the detected boxes, labels and confidences are drawn over the live preview.

- Bundled models, switchable in the app: **YOLO11n** (fast) and **YOLO11s** at 640, 480 and 384 input sizes (more accurate, slower; smaller input = faster)
- 80 COCO classes (person, car, bottle, laptop, ...)
- Kotlin + Jetpack Compose + CameraX, no network access needed at runtime
- Tested on a Poco X7 Pro (Android 15 / HyperOS)

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
3. The top-left text shows `model · number of objects · inference time (ms)`.
4. Tap **Switch model** to cycle through the bundled models (`yolo11n`, `yolo11s`, `yolo11s_480`, `yolo11s_384`) and compare speed and accuracy on the same scene.

Detections below 25% confidence are hidden, and overlapping boxes of the same class are merged (NMS, IoU 0.45).
Both thresholds are defaults of `YoloPostProcessor.decode`.

## How it works

```
CameraX ImageAnalysis (RGBA frame)
  -> rotate upright -> letterbox to 640x640 (gray 114 padding) -> float RGB CHW tensor [1,3,640,640]
  -> ONNX Runtime (XNNPACK, 4 threads) -> output [1,84,8400]  (cx, cy, w, h + 80 class scores per anchor)
  -> confidence filter + class-aware NMS -> boxes mapped back to the camera frame (normalized 0..1)
  -> Compose Canvas overlay on top of the PreviewView
```

| File | Role |
|---|---|
| `MainActivity.kt` | Edge-to-edge Compose host, camera permission request |
| `CameraScreen.kt` | CameraX preview + analysis, model switch button, box overlay |
| `YoloDetector.kt` | Loads the ONNX model, preprocessing, runs inference |
| `YoloPostProcessor.kt` | Letterbox math, output decoding, NMS (pure Kotlin, unit tested) |
| `CocoLabels.kt` | The 80 COCO class names |
| `app/src/main/assets/*.onnx` | The bundled models |

## Models

The ONNX files live in `app/src/main/assets/` and are exported from the pretrained
Ultralytics weights (COCO, 640x640, opset 12):

```powershell
uv run --python 3.12 --with ultralytics --with onnx --with onnxslim --with onnxruntime `
  yolo export model=yolo11n.pt format=onnx imgsz=640 opset=12 simplify=True
```

To add or replace a model:

1. Export it as above (any YOLOv8/YOLO11 detection model with 80 classes produces the same `[1,84,N]` output) and copy the `.onnx` file into `app/src/main/assets/`.
   The input size is read from the model, so you can export smaller or larger sizes (multiples of 32), e.g. `imgsz=480`.
2. Add its name (without `.onnx`) to `MODELS` in `CameraScreen.kt`.

A custom-trained model with a different number of classes needs the class names in `CocoLabels.kt`
and `YoloPostProcessor.NUM_CLASSES` updated.

## Performance

Measured on a Poco X7 Pro (Dimensity 8400-Ultra), ONNX Runtime with the XNNPACK provider and 4 CPU threads.
Times are per frame (preprocessing + inference + decoding), averaged over 50-100 frames:

| Model (switch button) | Input | File size | Time per frame | ~FPS |
|---|---|---|---|---|
| `yolo11n` | 640 | 10 MB | 130-160 ms | 6-8 |
| `yolo11s` | 640 | 36 MB | ~390 ms | 2.5 |
| `yolo11s_480` | 480 | 36 MB | ~250 ms | 4 |
| `yolo11s_384` | 384 | 36 MB | ~165 ms | 6 |

- Switching from the default CPU provider to XNNPACK made no meaningful difference on this phone.
- Time scales with the number of input pixels, so a smaller input is the most effective speed-up. The cost is accuracy on small objects.
- Numbers vary with scene, phone temperature and background load; treat them as ballparks.
- Every frame also logs its time: `adb logcat -s Inference:D`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `INSTALL_FAILED_USER_RESTRICTED` | Enable **Install via USB** in Developer options (Xiaomi/Poco/Redmi) and accept the prompt on the phone |
| `adb devices` shows `unauthorized` or nothing | Re-plug the cable, accept the RSA prompt, set USB mode to File transfer |
| Black screen from `adb shell screencap` | The phone is locked or the screen is off; unlock it |
| `SDK location not found` | Create `local.properties` with `sdk.dir=...` |
| Gradle `Couldn't delete ...\R.jar` (Windows) | Something locks build files, usually the VS Code Kotlin language server or a stray Gradle daemon. Run `.\gradlew.bat --stop`, close/kill `java.exe` processes, delete `app\build`, rebuild |
| App crashes on start | `adb logcat -s AndroidRuntime:E` |
| Emulator is slow or fails to start | `emulator -accel-check`; the emulator camera only shows a virtual scene, a real phone is recommended |

## License

The app code is provided as-is for experimentation. The bundled models are derived from
[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), licensed under **AGPL-3.0**; commercial use
requires an Ultralytics license or a different model.
