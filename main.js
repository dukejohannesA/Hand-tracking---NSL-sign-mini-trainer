import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

// ============================================================
// CONFIG
// ============================================================
const MODEL_PATH = 'models/letters/nsl_model_letters.json';
const CONFIDENCE_THRESHOLD = 0.70;   // below this, show "unsure"
const SMOOTHING_WINDOW = 5;           // average predictions over N frames


// ============================================================
// DOM
// ============================================================
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const statusBar = document.getElementById('status-bar');
const errorBox = document.getElementById('error-box');
const predLetterEl = document.getElementById('pred-letter');
const predConfidenceEl = document.getElementById('pred-confidence');
const confFillEl = document.getElementById('conf-fill');
const top3El = document.getElementById('top3');

const ctx = canvas.getContext('2d');

// ============================================================
// STATE
// ============================================================
let handLandmarker = null;
let classifier = null;
let CLASSES = [];
let cameraReady = false;
let modelReady = false;
let lastVideoTime = -1;
let predictionHistory = [];

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17]
];

function showError(message) {
  errorBox.textContent = message;
  errorBox.style.display = 'block';
}

// ============================================================
// NORMALIZATION — must match record.js exactly
// ============================================================
function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];

  const translated = landmarks.map(lm => ({
    x: lm.x - wrist.x,
    y: lm.y - wrist.y,
    z: (lm.z || 0) - (wrist.z || 0)
  }));

  let maxDist = 0;
  for (const p of translated) {
    const d = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (d > maxDist) maxDist = d;
  }
  if (maxDist === 0) maxDist = 1;

  const flat = [];
  for (const p of translated) {
    flat.push(p.x / maxDist, p.y / maxDist, p.z / maxDist);
  }
  return flat;
}

// ============================================================
// LOAD THE TRAINED CLASSIFIER
// ============================================================
// ============================================================
// LOAD THE TRAINED CLASSIFIER
// Rebuilds the model in the browser from exported weights.
// ============================================================
async function loadClassifier() {
  try {
    statusBar.textContent = 'Loading recognition model...';

    const resp = await fetch(MODEL_PATH);
    if (!resp.ok) throw new Error(`Could not load ${MODEL_PATH}`);
    const spec = await resp.json();

    CLASSES = spec.classes;

    // Rebuild the same architecture that was trained in Colab
    const model = tf.sequential();
    spec.architecture.forEach((layer, i) => {
      model.add(tf.layers.dense({
        units: layer.units,
        activation: layer.activation,
        inputShape: i === 0 ? [spec.inputDim] : undefined
      }));
    });

    // Load the trained weights layer by layer
    spec.weights.forEach((w, i) => {
      const kernel = tf.tensor2d(w.kernel);
      const bias = tf.tensor1d(w.bias);
      model.layers[i].setWeights([kernel, bias]);
      kernel.dispose();
      bias.dispose();
    });

    classifier = model;

    // Warm up so the first real prediction isn't slow
    const warmup = tf.zeros([1, spec.inputDim]);
    classifier.predict(warmup).dispose();
    warmup.dispose();

    modelReady = true;
    console.log('Classifier loaded. Classes:', CLASSES);
    tryStart();
  } catch (err) {
    console.error('Classifier loading error:', err);
    showError(`Failed to load the recognition model: ${err.message}`);
  }
}

// ============================================================
// PREDICTION with temporal smoothing
// ============================================================
function predict(landmarks) {
  const features = normalizeLandmarks(landmarks);

  const probs = tf.tidy(() => {
    const input = tf.tensor2d([features], [1, 63]);
    return classifier.predict(input).dataSync();
  });

  // Keep a short history and average it — reduces frame-to-frame flicker
  predictionHistory.push(Array.from(probs));
  if (predictionHistory.length > SMOOTHING_WINDOW) predictionHistory.shift();

  const averaged = new Array(CLASSES.length).fill(0);
  for (const p of predictionHistory) {
    for (let i = 0; i < CLASSES.length; i++) averaged[i] += p[i];
  }
  for (let i = 0; i < CLASSES.length; i++) averaged[i] /= predictionHistory.length;

  // Rank the results
  const ranked = averaged
    .map((prob, i) => ({ letter: CLASSES[i], prob }))
    .sort((a, b) => b.prob - a.prob);

  return ranked;
}

// ============================================================
// UI UPDATE
// ============================================================
function updatePredictionUI(ranked) {
  const top = ranked[0];
  const confident = top.prob >= CONFIDENCE_THRESHOLD;

  predLetterEl.textContent = confident ? top.letter : '?';
  predConfidenceEl.textContent = `${(top.prob * 100).toFixed(0)}%`;
  confFillEl.style.width = `${top.prob * 100}%`;

  // Top 3 alternatives
  top3El.innerHTML = ranked.slice(0, 3).map(r =>
    `<div class="top3-item">${r.letter}<span class="pct">${(r.prob * 100).toFixed(0)}%</span></div>`
  ).join('');

  // Highlight the matching reference card
  document.querySelectorAll('.letter-card.active').forEach(c => c.classList.remove('active'));
  if (confident) {
    const card = document.querySelector(`.letter-card[data-letter="${top.letter}"]`);
    if (card) card.classList.add('active');
  }

  statusBar.textContent = confident
    ? `Recognised "${top.letter}"`
    : 'Hold the sign steady — not confident yet';
}

function clearPredictionUI() {
  predLetterEl.textContent = '—';
  predConfidenceEl.textContent = '0%';
  confFillEl.style.width = '0%';
  top3El.innerHTML = '';
  document.querySelectorAll('.letter-card.active').forEach(c => c.classList.remove('active'));
  predictionHistory = [];
}

// ============================================================
// MEDIAPIPE + CAMERA
// ============================================================
async function loadHandLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      numHands: 1,
      runningMode: "VIDEO"
    });
    console.log('HandLandmarker loaded');
    tryStart();
  } catch (err) {
    console.error('Hand tracking error:', err);
    showError(`Failed to load hand tracking: ${err.message}`);
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false
    });
    video.srcObject = stream;
    video.addEventListener('loadedmetadata', () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      cameraReady = true;
      console.log(`Camera started at ${video.videoWidth} x ${video.videoHeight}`);
      tryStart();
    });
  } catch (err) {
    console.error('Camera error:', err);
    if (err.name === 'NotAllowedError') {
      showError('Camera permission denied. Allow camera access and refresh the page.');
    } else if (err.name === 'NotFoundError') {
      showError('No camera found on this device.');
    } else {
      showError(`Camera error: ${err.message}`);
    }
  }
}

function tryStart() {
  if (cameraReady && handLandmarker && modelReady) {
    statusBar.textContent = 'Show your hand to the camera';
    console.log('Starting detection loop');
    detectLoop();
  }
}

function drawLandmarks(landmarks) {
  ctx.strokeStyle = '#ffcd00';
  ctx.lineWidth = 3;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * canvas.width, landmarks[a].y * canvas.height);
    ctx.lineTo(landmarks[b].x * canvas.width, landmarks[b].y * canvas.height);
    ctx.stroke();
  }
  ctx.fillStyle = '#192230';
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function detectLoop() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    const results = handLandmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];
      drawLandmarks(landmarks);

      const ranked = predict(landmarks);
      updatePredictionUI(ranked);
    } else {
      clearPredictionUI();
      statusBar.textContent = 'No hand detected — show your hand to the camera';
    }
  }

  requestAnimationFrame(detectLoop);
}

// ============================================================
// INIT
// ============================================================
startCamera();
loadHandLandmarker();
loadClassifier();