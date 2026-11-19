/**
 * NSL Data Recorder — record.js
 * Improvements:
 *  - Hand quality indicator (landmark confidence)
 *  - Auto-augmentation (small noise) for better generalization
 *  - Keyboard shortcuts (1-9 for class jump)
 *  - Visual recording feedback
 *  - Better localStorage management
 */

import {
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// ─── CONFIG ────────────────────────────────────────────────────
const CLASSES            = ["B","C","D","I","L","O","V","W","Y"];
const SAMPLES_GOAL       = 50;
const CAPTURE_INTERVAL_MS = 120;   // ~8 samples/sec while holding
const AUGMENT_NOISE       = 0.015; // small noise magnitude for augmentation
const STORAGE_KEY         = 'nsl_dataset_letters';

// ─── DOM REFS ──────────────────────────────────────────────────
const video          = document.getElementById('video');
const canvas         = document.getElementById('canvas');
const ctx            = canvas.getContext('2d');
const targetLetter   = document.getElementById('target-letter');
const sampleCount    = document.getElementById('sample-count');
const targetProgress = document.getElementById('target-progress-fill');
const targetRefImg   = document.getElementById('target-ref-img');
const videoContainer = document.getElementById('video-container');
const qualityBadge   = document.getElementById('quality-badge');
const classList      = document.getElementById('class-list');
const totalSamplesEl = document.getElementById('total-samples');
const statusBar      = document.getElementById('status-bar');
const augmentCheck   = document.getElementById('augment-check');

// Buttons
const btnRecord   = document.getElementById('btn-record');
const btnPrev     = document.getElementById('btn-prev');
const btnNext     = document.getElementById('btn-next');
const btnClear    = document.getElementById('btn-clear');
const btnDownload = document.getElementById('btn-download');
const btnReset    = document.getElementById('btn-reset');

// ─── STATE ─────────────────────────────────────────────────────
let handLandmarker = null;
let currentClassIdx = 0;
let isRecording = false;
let recordingTimer = null;
let dataset = {};

// ─── HAND CONNECTIONS ──────────────────────────────────────────
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];

// ─── NORMALIZE LANDMARKS ───────────────────────────────────────
function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];

  const translated = landmarks.map(p => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z
  }));

  let maxDist = 0;
  for (const p of translated) {
    const d = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (d > maxDist) maxDist = d;
  }
  if (maxDist === 0) maxDist = 1;

  const features = [];
  for (const p of translated) {
    features.push(p.x / maxDist, p.y / maxDist, p.z / maxDist);
  }
  return features;
}

// ─── AUGMENTATION ──────────────────────────────────────────────
/**
 * Add small random noise to normalized landmarks.
 * This helps the model generalize to slight hand variations.
 */
function augmentFeatures(features) {
  return features.map(v => v + (Math.random() - 0.5) * 2 * AUGMENT_NOISE);
}

// ─── LARGEST HAND SELECTION ────────────────────────────────────
function selectLargestHand(results) {
  if (!results.landmarks || results.landmarks.length === 0) return null;
  if (results.landmarks.length === 1) return results.landmarks[0];

  let bestIdx = 0;
  let bestArea = 0;

  for (let i = 0; i < results.landmarks.length; i++) {
    const lm = results.landmarks[i];
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    for (const p of lm) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const area = (maxX - minX) * (maxY - minY);
    if (area > bestArea) {
      bestArea = area;
      bestIdx = i;
    }
  }

  return results.landmarks[bestIdx];
}

// ─── HAND QUALITY CHECK ────────────────────────────────────────
function getHandQuality(landmarks) {
  if (!landmarks) return 'none';

  // Check if hand fills enough of the frame
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }

  const handWidth = maxX - minX;
  const handHeight = maxY - minY;
  const handArea = handWidth * handHeight;

  // Check if hand is well-centered
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const offCenter = Math.abs(centerX - 0.5) + Math.abs(centerY - 0.5);

  if (handArea < 0.02) return 'bad';   // too small / too far
  if (handArea > 0.5)  return 'bad';   // too close / cut off
  if (offCenter > 0.6) return 'ok';    // off-center
  if (handArea < 0.04) return 'ok';    // a bit small

  return 'good';
}

// ─── DRAWING ───────────────────────────────────────────────────
function drawLandmarks(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks || landmarks.length === 0) return;

  const w = canvas.width;
  const h = canvas.height;

  // Draw connections
  const color = isRecording ? 'rgba(220, 38, 38, 0.8)' : 'rgba(255, 205, 0, 0.6)';
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  for (const [a, b] of HAND_CONNECTIONS) {
    const pA = landmarks[a];
    const pB = landmarks[b];
    ctx.beginPath();
    ctx.moveTo(pA.x * w, pA.y * h);
    ctx.lineTo(pB.x * w, pB.y * h);
    ctx.stroke();
  }

  // Draw points
  for (const p of landmarks) {
    ctx.fillStyle = isRecording ? '#dc2626' : 'white';
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

// ─── DATA MANAGEMENT ───────────────────────────────────────────
function loadDataset() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      dataset = JSON.parse(stored);
    }
  } catch (e) {
    console.warn('Failed to load dataset from localStorage:', e);
  }

  // Ensure all classes exist
  for (const cls of CLASSES) {
    if (!dataset[cls]) dataset[cls] = [];
  }
}

function saveDataset() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(dataset));
  } catch (e) {
    console.warn('localStorage save failed (might be full):', e);
    statusBar.textContent = '⚠ Storage full — download your data and reset.';
  }
}

function getClassCount(cls) {
  return (dataset[cls] || []).length;
}

function getTotalCount() {
  let total = 0;
  for (const cls of CLASSES) {
    total += getClassCount(cls);
  }
  return total;
}

// ─── UI UPDATES ────────────────────────────────────────────────
function updateUI() {
  const cls = CLASSES[currentClassIdx];
  const count = getClassCount(cls);

  targetLetter.textContent = cls;
  sampleCount.textContent = `${count} / ${SAMPLES_GOAL}`;
  targetProgress.style.width = Math.min(100, (count / SAMPLES_GOAL) * 100) + '%';
  targetRefImg.src = `images/letters/${cls}.jpg`;
  targetRefImg.alt = `Reference for letter ${cls}`;

  // Update class list
  classList.innerHTML = '';
  for (let i = 0; i < CLASSES.length; i++) {
    const c = CLASSES[i];
    const n = getClassCount(c);
    const pct = Math.min(100, (n / SAMPLES_GOAL) * 100);

    const li = document.createElement('li');
    li.className = 'class-item' + (i === currentClassIdx ? ' active' : '');
    li.innerHTML = `
      <span class="letter">${c}</span>
      <span class="mini-bar"><span class="mini-bar-fill" style="width:${pct}%"></span></span>
      <span class="sample-count">${n}</span>
    `;
    li.addEventListener('click', () => {
      currentClassIdx = i;
      updateUI();
    });
    classList.appendChild(li);
  }

  totalSamplesEl.textContent = getTotalCount();
}

// ─── RECORDING ─────────────────────────────────────────────────
let currentLandmarks = null;

function captureOneSample() {
  if (!currentLandmarks) return;

  const cls = CLASSES[currentClassIdx];
  const features = normalizeLandmarks(currentLandmarks);

  // Store the original
  if (!dataset[cls]) dataset[cls] = [];
  dataset[cls].push(features);

  // Optionally add augmented copy
  if (augmentCheck.checked) {
    dataset[cls].push(augmentFeatures(features));
  }

  saveDataset();
  updateUI();
}

function startRecording() {
  if (isRecording) return;
  isRecording = true;
  videoContainer.classList.add('recording');
  btnRecord.textContent = '■ Recording...';
  btnRecord.classList.add('recording');

  // Capture immediately, then on interval
  captureOneSample();
  recordingTimer = setInterval(captureOneSample, CAPTURE_INTERVAL_MS);
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  videoContainer.classList.remove('recording');
  btnRecord.textContent = '● Record';
  btnRecord.classList.remove('recording');

  if (recordingTimer) {
    clearInterval(recordingTimer);
    recordingTimer = null;
  }
}

// ─── DOWNLOAD ──────────────────────────────────────────────────
function downloadJSON() {
  const exportData = {
    version: 2,
    classes: CLASSES,
    normalization: 'wrist-origin, max-distance scaled, 63-value flat vector (x,y,z × 21 landmarks)',
    augmented: augmentCheck.checked,
    counts: {},
    data: {}
  };

  for (const cls of CLASSES) {
    exportData.counts[cls] = getClassCount(cls);
    exportData.data[cls] = dataset[cls] || [];
  }

  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `nsl_landmarks_${new Date().toISOString().slice(0,10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── DETECT LOOP ───────────────────────────────────────────────
function detectLoop() {
  if (video.readyState >= 2 && handLandmarker) {
    const results = handLandmarker.detectForVideo(video, performance.now());
    const landmarks = selectLargestHand(results);

    currentLandmarks = landmarks;

    if (landmarks) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      drawLandmarks(landmarks);

      // Quality indicator
      const quality = getHandQuality(landmarks);
      qualityBadge.textContent = quality === 'good' ? '✓ Good' : quality === 'ok' ? '~ OK' : '✗ Bad';
      qualityBadge.className = 'quality-badge ' + quality;
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      qualityBadge.textContent = 'No hand';
      qualityBadge.className = 'quality-badge';
    }
  }

  requestAnimationFrame(detectLoop);
}

// ─── INIT ──────────────────────────────────────────────────────
async function init() {
  loadDataset();

  // Start camera
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(resolve => { video.onloadedmetadata = resolve; });
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    statusBar.textContent = 'Camera ready, loading hand tracker...';
  } catch (err) {
    statusBar.textContent = 'Camera error: ' + err.message;
    return;
  }

  // Load MediaPipe
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2
  });

  statusBar.textContent = 'Ready — select a letter and hold SPACE or the Record button to capture samples.';
  updateUI();
  detectLoop();
}

// ─── EVENT LISTENERS ───────────────────────────────────────────

// Record button (mouse)
btnRecord.addEventListener('mousedown', startRecording);
btnRecord.addEventListener('mouseup', stopRecording);
btnRecord.addEventListener('mouseleave', stopRecording);

// Record button (touch)
btnRecord.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
btnRecord.addEventListener('touchend', stopRecording);
btnRecord.addEventListener('touchcancel', stopRecording);

// Nav
btnPrev.addEventListener('click', () => {
  currentClassIdx = (currentClassIdx - 1 + CLASSES.length) % CLASSES.length;
  updateUI();
});

btnNext.addEventListener('click', () => {
  currentClassIdx = (currentClassIdx + 1) % CLASSES.length;
  updateUI();
});

// Clear current class
btnClear.addEventListener('click', () => {
  const cls = CLASSES[currentClassIdx];
  if (confirm(`Clear all ${getClassCount(cls)} samples for letter "${cls}"?`)) {
    dataset[cls] = [];
    saveDataset();
    updateUI();
  }
});

// Download
btnDownload.addEventListener('click', downloadJSON);

// Reset all
btnReset.addEventListener('click', () => {
  if (confirm('Reset ALL recorded data? This cannot be undone.')) {
    for (const cls of CLASSES) dataset[cls] = [];
    saveDataset();
    updateUI();
  }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  if (e.code === 'Space') {
    e.preventDefault();
    startRecording();
  } else if (e.key === 'n' || e.key === 'N') {
    currentClassIdx = (currentClassIdx + 1) % CLASSES.length;
    updateUI();
  } else if (e.key === 'p' || e.key === 'P') {
    currentClassIdx = (currentClassIdx - 1 + CLASSES.length) % CLASSES.length;
    updateUI();
  } else if (e.key >= '1' && e.key <= '9') {
    const idx = parseInt(e.key) - 1;
    if (idx < CLASSES.length) {
      currentClassIdx = idx;
      updateUI();
    }
  }
});

document.addEventListener('keyup', (e) => {
  if (e.code === 'Space') {
    stopRecording();
  }
});

// ─── START ─────────────────────────────────────────────────────
init();
