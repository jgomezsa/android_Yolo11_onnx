# Android Object Detection (YOLO11)

Point your phone's camera at the world and the app draws a box and a label around every object it recognises,
live, using an AI model that runs entirely **on the phone**. No internet connection and no server are needed, and
no image ever leaves the device.

- Recognises the 80 everyday object classes of the COCO dataset (person, car, bottle, laptop, cat, ...)
- Runs on the phone's **GPU**, so it stays smooth on a recent phone
- Adapts to slower phones by itself
- Kotlin, Jetpack Compose and CameraX; tested on a Poco X7 Pro (Android 15)

## What you need

| Requirement | Notes |
|---|---|
| Android phone | Android 8.0+ (API 26), with a camera and a USB cable |
| JDK 17 | Set `JAVA_HOME`. Tested with Microsoft OpenJDK 17 |
| Android SDK | `platform-tools`, `platforms;android-35`, `build-tools;35.0.0` (command-line tools are enough; Android Studio is optional) |
| Git | To clone the repo |

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

1. Open **Object Detection** and allow the **Camera** permission.
2. Point the camera at objects. Each one gets a coloured box with its name and a confidence percentage.
3. The number in the top-left corner is how long the last frame took to process, in milliseconds. The lower it is, the smoother the detection. The small graph to its left shows the same time for the last 50 frames.
4. The chip icon in the top-right corner tells you whether the phone's GPU is doing the work (teal **GPU**) or the app had to fall back to another way of running (amber **No GPU**, slower).
5. The **Stop** button in the bottom-left corner pauses the detection (the camera keeps showing the live view, and no boxes are drawn). Press **Start** to resume.

There is nothing to configure: the app picks the best way to run on your phone by itself.
The first start after installing takes a few seconds (the top-left corner says "Loading..."); later starts are almost instant.

## How it works

Every camera frame goes through the same steps:

```
Camera frame -> resize and pad to a square -> AI model on the GPU -> raw predictions
             -> keep the confident ones, remove duplicates -> boxes drawn over the preview
```

1. **Camera.** CameraX delivers two streams: the live preview you see, and a stream of frames for analysis. If the analysis falls behind it
   simply skips frames, so the preview never lags.
2. **Preparation.** The model expects a square 640x640 picture. The frame is rotated upright and scaled to fit inside the square, and the
   unused space is filled with grey so nothing is stretched. The pixels are converted to numbers between 0 and 1.
3. **The model.** It is [YOLO11](https://docs.ultralytics.com/models/yolo11/) "small", a neural network trained on the COCO dataset. In one pass it looks at the
   whole picture and proposes 8,400 candidate boxes, each with a score for every one of the 80 classes. It runs through
   [LiteRT](https://ai.google.dev/edge/litert) (Google's on-device runtime, the successor of TensorFlow Lite) on the phone's GPU.
4. **Cleaning up the result** (`YoloPostProcessor`). Candidates scoring below 25% are dropped. Often several boxes cover the same object, so
   the best one is kept and the others that overlap it by more than 45% are discarded (*non-maximum suppression*, applied per class).
   The surviving boxes are mapped back from the padded square to the real camera picture.
5. **Drawing.** The boxes and labels are painted on a Compose canvas laid over the camera preview.

Some details that keep it fast and robust:

- **GPU program cache.** Before the first run the GPU needs the model translated into programs for that specific chip, which takes a few seconds.
  The result is saved in the app's cache folder, so every later start skips that step. If the model file changes, the cache key changes with it.
- **Fallbacks.** If a phone's GPU cannot run the model, the app silently falls back to the phone's AI accelerator (through ONNX Runtime and NNAPI),
  and if that fails too, to the CPU. The user never has to choose.
- **Slow phones.** The app measures how many frames per second it really achieves. If the average drops below 3 frames per second it swaps in a
  lighter version of the model that works on a smaller 480x480 picture, trading a little accuracy on small objects for speed.
- **Everything is local.** The models are bundled inside the app; nothing is downloaded or uploaded.

### Project layout

| File | Role |
|---|---|
| `MainActivity.kt` | Edge-to-edge Compose host and the camera permission request |
| `CameraScreen.kt` | CameraX preview and analysis, picks and falls back between detectors, draws the boxes |
| `ObjectDetector.kt` | Common interface for the detectors, and the code that pads a frame into the square model input |
| `LiteRtDetector.kt` | The GPU detector (LiteRT, `.tflite` models) with the program cache |
| `YoloDetector.kt` | The fallback detector (ONNX Runtime, `.onnx` models) |
| `YoloPostProcessor.kt` | Turns the model's raw output into boxes: thresholds, overlap removal, coordinate mapping (unit tested) |
| `FrameTimeHistory.kt` | Keeps the last 50 frame times for the graph (unit tested) |
| `FpsMonitor.kt` | Decides when the phone is too slow and the lighter model is needed (unit tested) |
| `CocoLabels.kt` | The 80 class names |
| `app/src/main/assets/` | The bundled models: `yolo11s` and the lighter `yolo11s_480`, each in `.tflite` (GPU) and `.onnx` (fallback) form |

More technical detail (architecture, models and how to convert them, measurements, licensing): [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md).

## Troubleshooting

| Problem | Fix |
|---|---|
| `INSTALL_FAILED_USER_RESTRICTED` | Enable **Install via USB** in Developer options (Xiaomi/Poco/Redmi) and accept the prompt on the phone |
| `adb devices` shows `unauthorized` or nothing | Re-plug the cable, accept the RSA prompt, set USB mode to File transfer |
| `SDK location not found` | Create `local.properties` with `sdk.dir=...` |
| Gradle `Couldn't delete ...\R.jar` (Windows) | Something locks build files, usually an IDE language server or a stray Gradle daemon. Run `.\gradlew.bat --stop`, close the IDE, delete `app\build`, rebuild |
| The app crashes on start | `adb logcat -s AndroidRuntime:E` |

## License and commercial use

The app code is provided as-is for experimentation. The bundled models are derived from
[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), which is licensed under the **AGPL-3.0**
(or, alternatively, a paid Ultralytics Enterprise License). AGPL does not forbid commercial use, but anything that
includes the YOLO11 code or weights must itself be released under the AGPL with its source made available to its users.
To distribute a closed-source or commercial app you need an Ultralytics Enterprise License or a model with a permissive licence
(for example RT-DETR or YOLOX, both Apache-2.0). Check the current terms before publishing; this is not legal advice.
