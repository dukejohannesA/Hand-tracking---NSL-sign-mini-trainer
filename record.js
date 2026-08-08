import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

// ============================================================
// CONFIG — change CLASSES to ["1","2",...] later for numbers
// ============================================================
const CLASSES = ["B", "C", "D", "I", "L", "O", "V", "W", "Y"];
const SAMPLES_GOAL = 50;
const CAPTURE_INTERVAL_MS = 120;   // ~8 samples/sec while holding SPACE
const STORAGE_KEY = "nsl_dataset_letters";

// ============================================================
// DOM
// ============================================================
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const videoContainer = document.getElementById('video-container');
const statusBar = document.getElementById('status-bar');
const errorBox = document.getElementById('error-box');
const targetLetterEl = document.getElementById('target-letter');
const targetCountEl = document.getElementById('target-count');
const targetGoalEl = document.getElementById('target-goal');
const targetProgressEl = document.getElementById('target-progress');
const classListEl = document.getElementById('class-list');
const totalInfoEl = document.getElementById('total-info');

const ctx = canvas.getContext('2d');

// ============================================================
// STATE
// ============================================================
let handLandmarker = null;
let cameraReady = false;
let lastVideoTime = -1;
let currentLandmarks = null;

let currentClassIndex = 0;
let isRecording = false;
let lastCaptureTime = 0;

// dataset: { "B": [[63 numbers], ...], "C": [...], ... }
let dataset = {};
CLASSES.forEach(c => dataset[c] = []);

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17]
];

// ============================================================
// STORAGE
// ============================================================
function saveDataset() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(dataset));
  } catch (e) {
    console.warn('Could not save to localStorage:', e);
  }
}

function loadDataset() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      CLASSES.forEach(c => {
        if (Array.isArray(parsed[c])) dataset[c] = parsed[c];
      });
      console.log('Loaded existing dataset from browser storage');
    }
  } catch (e) {
    console.warn('Could not load from localStorage:', e);
  }
}

// ============================================================
// NORMALIZATION (Option B)
// Translate so wrist is origin, scale so hand size is 1.
// Returns a flat array of 63 numbers.
// ============================================================
function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];

  // Translate: wrist becomes (0,0,0)
  const translated = landmarks.map(lm => ({
    x: lm.x - wrist.x,
    y: lm.y - wrist.y,
    z: (lm.z || 0) - (wrist.z || 0)
  }));

  // Scale: largest distance from wrist becomes 1
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
// UI
// ============================================================
function buildClassList() {
  classListEl.innerHTML = '';
  CLASSES.forEach((cls, i) => {
    const row = document.createElement('div');
    row.className = 'class-row';
    row.dataset.index = i;
    row.innerHTML = `
      <span class="cls-name">${cls}</span>
      <span class="cls-bar"><span class="cls-bar-fill"></span></span>
      <span class="cls-count">0/${SAMPLES_GOAL}</span>
    `;
    row.addEventListener('click', () => {
      currentClassIndex = i;
      updateUI();
    });
    classListEl.appendChild(row);
  });
}

function updateUI() {
  const currentClass = CLASSES[currentClassIndex];
  const count = dataset[currentClass].length;

  targetLetterEl.textContent = currentClass;
  targetCountEl.textContent = count;
  targetGoalEl.textContent = SAMPLES_GOAL;
  targetProgressEl.style.width = `${Math.min(100, (count / SAMPLES_GOAL) * 100)}%`;

  let total = 0;
  CLASSES.forEach((cls, i) => {
    const c = dataset[cls].length;
    total += c;
    const row = classListEl.children[i];
    row.classList.toggle('active', i === currentClassIndex);
    row.classList.toggle('complete', c >= SAMPLES_GOAL);
    row.querySelector('.cls-bar-fill').style.width = `${Math.min(100, (c / SAMPLES_GOAL) * 100)}%`;
    row.querySelector('.cls-count').textContent = `${c}/${SAMPLES_GOAL}`;
  });

  const done = CLASSES.filter(c => dataset[c].length >= SAMPLES_GOAL).length;
  totalInfoEl.textContent = `${total} samples total • ${done}/${CLASSES.length} letters complete`;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.style.display = 'block';
}

// ============================================================
// CAPTURE
// ============================================================
function captureSample() {
  if (!currentLandmarks) return false;
  const currentClass = CLASSES[currentClassIndex];
  dataset[currentClass].push(normalizeLandmarks(currentLandmarks));
  saveDataset();
  updateUI();
  return true;
}

// ============================================================
// KEYBOARD
// ============================================================
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space') {
    e.preventDefault();
    if (!isRecording) {
      isRecording = true;
      videoContainer.classList.add('recording');
    }
  } else if (e.key.toLowerCase() === 'n') {
    currentClassIndex = (currentClassIndex + 1) % CLASSES.length;
    updateUI();
  } else if (e.key.toLowerCase() === 'p') {
    currentClassIndex = (currentClassIndex - 1 + CLASSES.length) % CLASSES.length;
    updateUI();
  }
});

document.addEventListener('keyup', (e) => {
  if (e.code === 'Space') {
    e.preventDefault();
    isRecording = false;
    videoContainer.classList.remove('recording');
  }
});

// ============================================================
// BUTTONS
// ============================================================
document.getElementById('btn-next').addEventListener('click', () => {
  currentClassIndex = (currentClassIndex + 1) % CLASSES.length;
  updateUI();
});

document.getElementById('btn-prev').addEventListener('click', () => {
  currentClassIndex = (currentClassIndex - 1 + CLASSES.length) % CLASSES.length;
  updateUI();
});

document.getElementById('btn-clear').addEventListener('click', () => {
  const currentClass = CLASSES[currentClassIndex];
  if (confirm(`Delete all ${dataset[currentClass].length} samples for "${currentClass}"?`)) {
    dataset[currentClass] = [];
    saveDataset();
    updateUI();
  }
});

document.getElementById('btn-reset').addEventListener('click', () => {
  if (confirm('Delete ALL recorded samples for every letter? This cannot be undone.')) {
    CLASSES.forEach(c => dataset[c] = []);
    saveDataset();
    updateUI();
  }
});

document.getElementById('btn-download').addEventListener('click', () => {
  const payload = {
    classes: CLASSES,
    normalization: "wrist-origin, max-distance-scaled, flat 63 values (x,y,z per landmark)",
    recordedAt: new Date().toISOString(),
    counts: Object.fromEntries(CLASSES.map(c => [c, dataset[c].length])),
    data: dataset
  };
  const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'nsl_landmarks_letters.json';
  a.click();
  URL.revokeObjectURL(url);
});

// ============================================================
// MEDIAPIPE + CAMERA
// ============================================================
async function loadHandLandmarker() {
  try {
    statusBar.textContent = 'Loading hand tracking model...';
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
    tryStartDetection();
  } catch (err) {
    console.error('Model loading error:', err);
    showError(`Failed to load model: ${err.message}`);
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
      tryStartDetection();
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

function tryStartDetection() {
  if (cameraReady && handLandmarker) {
    statusBar.textContent = 'Ready — hold SPACE to record';
    detectLoop();
  }
}

function drawLandmarks(landmarks) {
  ctx.strokeStyle = isRecording ? '#dc2626' : '#ffcd00';
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
      currentLandmarks = results.landmarks[0];
      drawLandmarks(currentLandmarks);

      const currentClass = CLASSES[currentClassIndex];
      if (isRecording) {
        const now = performance.now();
        if (now - lastCaptureTime >= CAPTURE_INTERVAL_MS) {
          lastCaptureTime = now;
          captureSample();
        }
        statusBar.textContent = `RECORDING "${currentClass}" — keep varying your hand angle and distance`;
      } else {
        statusBar.textContent = `Hand detected — hold SPACE to record "${currentClass}"`;
      }
    } else {
      currentLandmarks = null;
      statusBar.textContent = isRecording
        ? 'No hand detected — recording paused'
        : 'Show your hand to the camera';
    }
  }

  requestAnimationFrame(detectLoop);
}

// ============================================================
// INIT
// ============================================================
loadDataset();
buildClassList();
updateUI();
startCamera();
loadHandLandmarker();