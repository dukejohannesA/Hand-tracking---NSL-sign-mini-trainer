 # NSL Fingerspelling Mini-Trainer

 A browser-based mini-trainer for recognising selected Namibian Sign Language (NSL) fingerspelling letters from a webcam. The application runs hand tracking and classification locally in the browser, so camera frames are not uploaded to a server by this project.

 ## Features

 - Real-time webcam hand tracking with MediaPipe Hand Landmarker.
 - Recognition of nine trained letters: `B`, `C`, `D`, `I`, `L`, `O`, `V`, `W`, and `Y`.
 - Live predicted letter, confidence score, confidence bar, and top-three alternatives.
 - Reference cards for the supported letters.
 - Separate data-collection page for recording new landmark samples.
 - Browser persistence for recorded samples using `localStorage`.
 - JSON export for retraining the classifier.

 ## Quick Start

 This is a static web project and has no `npm` or Python package dependencies for running the trainer. It does need to be served from a local web server; opening `index.html` directly with a `file://` URL can prevent module imports, model loading, or webcam access from working.

 From the project directory, run one of the following:

 ```powershell
 # Python
 python -m http.server 8000
 ```

 Or use any equivalent static server, such as the VS Code Live Server extension. Then open:

 ```text
 http://localhost:8000/index.html
 ```

 Allow camera access when the browser asks. A modern browser with WebGL support is recommended because the application requests the MediaPipe GPU delegate.

 ## Using the Trainer

 1. Start the local server and open `index.html`.
 2. Allow webcam access.
 3. Place one hand in view of the camera.
 4. Hold the sign steady and watch the prediction and confidence.
 5. A prediction is displayed as recognised only when confidence is at least `70%`.

 The displayed result is smoothed over five frames to reduce flicker. The page mirrors the webcam image and draws the detected hand landmarks over the video.

 ## Recording Training Data

 Open `record.html` from the same local server:

 ```text
 http://localhost:8000/record.html
 ```

 The recorder currently collects the same nine classes as the classifier and targets 50 samples per class.

 - Select a letter from the class list, or use the Next/Previous controls.
 - Show the selected sign in the camera view.
 - Hold `Space` to record. Samples are captured about eight times per second while the key is held.
 - Vary hand angle, position, and distance slightly while recording.
 - Use `N` and `P` to move to the next or previous letter.
 - Use the per-letter clear action to delete one class, or the reset action to delete the whole browser dataset.
 - Download the completed dataset as `nsl_landmarks_letters.json`.

 Samples are kept in browser storage under the key `nsl_dataset_letters`. They are local to the browser profile and are not automatically written into the repository.

 ## Data and Feature Format

 MediaPipe supplies 21 hand landmarks with `x`, `y`, and `z` coordinates. Each sample is transformed in both `record.js` and `main.js` using the same procedure:

 1. Translate every landmark so the wrist is the origin.
 2. Find the greatest 3D distance from the wrist.
 3. Divide all coordinates by that distance.
 4. Flatten the 21 landmarks into 63 values in `x, y, z` order.

 The exported dataset has this shape:

 ```json
 {
	 "classes": ["B", "C", "D", "I", "L", "O", "V", "W", "Y"],
	 "normalization": "wrist-origin, max-distance-scaled, flat 63 values (x,y,z per landmark)",
	 "recordedAt": "ISO-8601 timestamp",
	 "counts": {"B": 54},
	 "data": {"B": [[0, 0, 0]]}
 }
 ```

 Each real sample in `data` contains 63 numbers. The small example above is abbreviated and is not a valid classifier input by itself.

 ## Model

 The current browser model is stored in [models/letters/nsl_model_letters.json](models/letters/nsl_model_letters.json). It contains the class order, architecture, and dense-layer weights so the model can be rebuilt with TensorFlow.js in the browser.

 ```text
 Input: 63 normalised landmark values
 Dense: 64 units, ReLU
 Dense: 32 units, ReLU
 Output: 9 units, softmax
 ```

 The training notebook uses dropout during training, but dropout layers do not have weights and are omitted from the exported browser architecture. The application uses a `70%` confidence threshold and averages predictions over a five-frame window.

 The checked-in training run used 762 samples, split into 609 training samples and 153 test samples with stratification. The notebook reports:

 - K-nearest-neighbours baseline accuracy: `96.7%`
 - Neural-network train accuracy: `96.2%`
 - Neural-network test accuracy: `97.4%`

 These are evaluation results for the recorded dataset, not a guarantee of performance for new users, cameras, lighting conditions, or hand orientations.

 ## Retraining the Model

 The training workflow is documented in [nsl_model_letters_train (1).ipynb](nsl_model_letters_train%20%281%29.ipynb) and is designed for Google Colab.

 1. Open the notebook in Colab.
 2. Install or enable the required Python libraries used by the notebook: TensorFlow, TensorFlow.js converter, NumPy, scikit-learn, Matplotlib, and Seaborn.
 3. Upload the exported `nsl_landmarks_letters.json` file when prompted.
 4. Build `X` and `y` from the class-labelled landmark arrays.
 5. Train and evaluate the dense network in the notebook.
 6. Run the export cell to create `nsl_model_letters.json`.
 7. Replace [models/letters/nsl_model_letters.json](models/letters/nsl_model_letters.json) with the new export.

 If classes are added or removed, update the class list and reference images in `index.html`, the `CLASSES` constant in `record.js`, and the exported model together. The class order in the model must match the output order used by the classifier.

 ## Project Structure

 ```text
 index.html                         Recognition UI
 main.js                            Camera, hand tracking, prediction, and UI logic
 record.html                        Data-collection UI
 record.js                          Recording, persistence, and JSON export logic
 nsl_model_letters_train (1).ipynb  Colab training and export workflow
 images/letters/                    Reference images shown by the trainer
 images/dataset/                    Dataset image folders and source material
 models/letters/                    Landmark dataset, class metadata, and model weights
 ```

 ## Dependencies

 Runtime dependencies are loaded from CDNs in the HTML and JavaScript files:

 - TensorFlow.js `4.20.0`
 - MediaPipe Tasks Vision, including the Hand Landmarker WASM runtime
 - MediaPipe Hand Landmarker task asset
 - Google Fonts: Inter and Space Grotesk

 An internet connection is therefore required unless these assets are self-hosted and the source URLs are updated.

 ## Limitations and Troubleshooting

 - Only one hand is tracked at a time.
 - Only the nine listed letters are supported; the model does not recognise arbitrary signs or words.
 - Predictions are based on hand landmarks, not raw image appearance, so occlusion, poor lighting, unusual orientations, and missed landmarks can reduce accuracy.
 - Camera access normally requires `localhost` or HTTPS. Check browser permissions if the page reports that access was denied.
 - If the model fails to load, confirm that the server was started from the project root and that `models/letters/nsl_model_letters.json` is reachable.
 - If the recorder appears to lose samples, avoid private browsing or clearing site data before downloading the dataset.

 ## Attribution

 The trainer footer identifies the NSL reference material as NNAD 1991 documentation. MediaPipe and TensorFlow.js are third-party runtime dependencies loaded from their public CDN assets.
