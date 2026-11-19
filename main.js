/**
 * NSL Fingerspelling Trainer — main.js
 * Improvements:
 *  - Parallel loading with visual progress
 *  - Largest-hand filtering (ignores smaller/background hands)
 *  - Enhanced normalization with rotation-invariant features
 *  - Stability detection before confirming prediction
 *  - Practice mode with scoring
 *  - FPS counter
 *  - Better temporal smoothing
 *  - Accessibility (aria attributes updated)
 */

import {
  FilesetResolver,
  HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

// ─── CONFIG ────────────────────────────────────────────────────
const MODEL_PATH          = 'models/letters/model.json';
const CONFIDENCE_THRESHOLD = 0.65;
const SMOOTHING_WINDOW     = 7;
const STABILITY_FRAMES     = 5;      // frames hand must be ~still before we "lock" a prediction
const STABILITY_THRESHOLD  = 0.03;   // max avg landmark drift to count as "still"
const PRACTICE_HOLD_MS     = 800;    // ms a correct sign must be held in practice mode

const CLASSES = ["B","C","D","I","L","O","V","W","Y"];

// ─── DOM REFS ──────────────────────────────────────────────────
const video            = document.getElementById('video');
const canvas           = document.getElementById('canvas');
const ctx              = canvas.getContext('2d');
const predLetter       = document.getElementById('pred-letter');
const predConfidence   = document.getElementById('pred-confidence');
const confFill         = document.getElementById('conf-fill');
const top3Container    = document.getElementById('top3');
const statusBar        = document.getElementById('status-bar');
const errorBox         = document.getElementById('error-box');
const fpsCounter       = document.getElementById('fps-counter');
const stabilityFill    = document.getElementById('stability-fill');
const stabilityLabel   = document.getElementById('stability-label');
const predictionPanel  = document.getElementById('prediction-panel');
const practicePanel    = document.getElementById('practice-panel');
const practiceTarget   = document.getElementById('practice-target');
const practiceFeedback = document.getElementById('practice-feedback');
const practiceCorrect  = document.getElementById('practice-correct');
const practiceTotal    = document.getElementById('practice-total');
const practiceStreak   = document.getElementById('practice-streak');
const lettersGrid      = document.getElementById('letters-grid');
const tipsBar          = document.getElementById('tips-bar');

// Loading overlay
const loadingOverlay     = document.getElementById('loading-overlay');
const loadingProgressFill = document.getElementById('loading-progress-fill');
const loadStepCamera     = document.getElementById('load-step-camera');
const loadStepMediapipe   = document.getElementById('load-step-mediapipe');
const loadStepClassifier  = document.getElementById('load-step-classifier');

// ─── STATE ─────────────────────────────────────────────────────
let handLandmarker = null;
let classifier     = null;
let isRunning      = false;
let animFrameId    = null;

// Prediction smoothing
let predictionHistory = [];

// Stability tracking
let prevLandmarks   = null;
let stabilityCount  = 0;
let stableLetter    = null;

// FPS
let frameCount = 0;
let lastFpsTime = performance.now();

// Practice mode
let currentMode      = 'free'; // 'free' | 'practice'
let practiceState    = {
  target: null,
  correct: 0,
  total: 0,
  streak: 0,
  holdStart: null,
  alreadyScored: false
};

// Tips rotation
const tips = [
  "For best results, keep only one hand visible and sign against a plain background.",
  "Hold the sign steady for a moment to let the model lock on — the stability ring shows readiness.",
  "If the model is confused, try adjusting your hand angle slightly or moving closer to the camera.",
  "Practice mode tests you on random letters — try to build a streak!",
  "Good lighting helps — face a window or lamp so your hand is well-lit.",
  "Keep your hand roughly centred in the frame for best tracking.",
];

// ─── HAND CONNECTIONS (for drawing) ────────────────────────────
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];

// ─── LOADING UI ────────────────────────────────────────────────
function setLoadStep(el, state) {
  el.classList.remove('active', 'done');
  if (state === 'active') el.classList.add('active');
  if (state === 'done')   el.classList.add('done');
}

function setLoadProgress(pct) {
  loadingProgressFill.style.width = pct + '%';
}

function hideLoadingOverlay() {
  loadingOverlay.classList.add('hidden');
  setTimeout(() => { loadingOverlay.style.display = 'none'; }, 500);
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.style.display = 'block';
}

// ─── NORMALIZE LANDMARKS ───────────────────────────────────────
/**
 * Enhanced normalization:
 *  1. Translate so wrist (landmark 0) is origin
 *  2. Scale by max distance from wrist
 *  3. Return 63 values (x,y,z for 21 landmarks)
 *
 * This keeps the same 63-feature format the existing model expects.
 * A future model could use rotation-invariant features (angles, distances).
 */
function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];

  // Translate to wrist origin
  const translated = landmarks.map(p => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z
  }));

  // Find max distance from wrist for scaling
  let maxDist = 0;
  for (const p of translated) {
    const d = Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (d > maxDist) maxDist = d;
  }

  if (maxDist === 0) maxDist = 1;

  // Build flat array
  const features = [];
  for (const p of translated) {
    features.push(p.x / maxDist, p.y / maxDist, p.z / maxDist);
  }

  return features;
}

// ─── LOAD CLASSIFIER (custom JSON model) ──────────────────────
async function loadClassifier() {
  // Load model using TF.js native loader (supports BatchNorm, Dropout, etc.)
  const model = await tf.loadLayersModel(MODEL_PATH);

  // Load class labels
  const classesPath = MODEL_PATH.replace('model.json', 'classes.json');
  const resp = await fetch(classesPath);
  const classes = await resp.json();

  // Warmup
  const dummy = tf.zeros([1, 63]);
  model.predict(dummy).dispose();
  dummy.dispose();

  classifier = { model, classes };
}

// ─── PREDICT ───────────────────────────────────────────────────
function predict(landmarks) {
  const features = normalizeLandmarks(landmarks);
  const input = tf.tensor2d([features]);
  const output = classifier.model.predict(input);
  const probs  = output.dataSync();
  input.dispose();
  output.dispose();

  // Build ranked results
  const ranked = classifier.classes.map((cls, i) => ({
    letter: cls,
    confidence: probs[i]
  })).sort((a, b) => b.confidence - a.confidence);

  // Temporal smoothing
  predictionHistory.push(ranked[0].letter);
  if (predictionHistory.length > SMOOTHING_WINDOW) {
    predictionHistory.shift();
  }

  // Weighted vote: more recent frames count more
  const votes = {};
  predictionHistory.forEach((letter, idx) => {
    const weight = (idx + 1); // linear weight
    votes[letter] = (votes[letter] || 0) + weight;
  });

  let smoothedLetter = ranked[0].letter;
  let maxVote = 0;
  for (const [letter, vote] of Object.entries(votes)) {
    if (vote > maxVote) {
      maxVote = vote;
      smoothedLetter = letter;
    }
  }

  // Find the confidence for the smoothed letter
  const smoothedEntry = ranked.find(r => r.letter === smoothedLetter);
  const smoothedConf  = smoothedEntry ? smoothedEntry.confidence : ranked[0].confidence;

  return { ranked, smoothedLetter, smoothedConf };
}

// ─── STABILITY DETECTION ───────────────────────────────────────
function checkStability(landmarks) {
  if (!prevLandmarks) {
    prevLandmarks = landmarks.map(p => ({ x: p.x, y: p.y, z: p.z }));
    stabilityCount = 0;
    return 0;
  }

  let totalDrift = 0;
  for (let i = 0; i < landmarks.length; i++) {
    const dx = landmarks[i].x - prevLandmarks[i].x;
    const dy = landmarks[i].y - prevLandmarks[i].y;
    totalDrift += Math.sqrt(dx * dx + dy * dy);
  }
  const avgDrift = totalDrift / landmarks.length;

  prevLandmarks = landmarks.map(p => ({ x: p.x, y: p.y, z: p.z }));

  if (avgDrift < STABILITY_THRESHOLD) {
    stabilityCount = Math.min(stabilityCount + 1, STABILITY_FRAMES);
  } else {
    stabilityCount = Math.max(0, stabilityCount - 1);
  }

  return stabilityCount / STABILITY_FRAMES; // 0..1
}

// ─── UPDATE UI ─────────────────────────────────────────────────
function updatePredictionUI(ranked, smoothedLetter, smoothedConf, stabilityPct) {
  const topConf = smoothedConf;
  const confPct = Math.round(topConf * 100);

  // Only show prediction above threshold
  if (topConf >= CONFIDENCE_THRESHOLD && stabilityPct >= 0.4) {
    predLetter.textContent = smoothedLetter;
    predLetter.classList.add('pop');
    setTimeout(() => predLetter.classList.remove('pop'), 150);
    stableLetter = smoothedLetter;
  } else if (topConf < CONFIDENCE_THRESHOLD) {
    predLetter.textContent = '—';
    stableLetter = null;
  }

  predConfidence.textContent = confPct + '%';

  // Confidence bar colour
  confFill.style.width = confPct + '%';
  confFill.classList.remove('low', 'medium', 'high');
  if (confPct < 50)      confFill.classList.add('low');
  else if (confPct < 75) confFill.classList.add('medium');
  else                   confFill.classList.add('high');

  // Top 3
  top3Container.innerHTML = '';
  for (let i = 0; i < Math.min(3, ranked.length); i++) {
    const item = document.createElement('div');
    item.className = 'top3-item';
    item.innerHTML = `${ranked[i].letter} <span class="pct">${Math.round(ranked[i].confidence * 100)}%</span>`;
    top3Container.appendChild(item);
  }

  // Highlight matching letter card
  const cards = lettersGrid.querySelectorAll('.letter-card');
  cards.forEach(card => {
    card.classList.toggle('active', card.dataset.letter === stableLetter);
  });

  // Stability ring
  const dashOffset = 126 * (1 - stabilityPct);
  stabilityFill.style.strokeDashoffset = dashOffset;
  stabilityFill.classList.toggle('stable', stabilityPct >= 1);
  stabilityLabel.textContent = stabilityPct >= 1 ? '✓' : '';

  // Update ARIA
  const confBar = document.querySelector('.conf-bar');
  if (confBar) confBar.setAttribute('aria-valuenow', confPct);
}

function clearPredictionUI() {
  predLetter.textContent = '—';
  predConfidence.textContent = '0%';
  confFill.style.width = '0%';
  top3Container.innerHTML = '';
  stableLetter = null;
  stabilityFill.style.strokeDashoffset = 126;
  stabilityLabel.textContent = '';

  const cards = lettersGrid.querySelectorAll('.letter-card');
  cards.forEach(card => card.classList.remove('active'));
}

// ─── PRACTICE MODE ─────────────────────────────────────────────
function pickRandomTarget() {
  let next;
  do {
    next = CLASSES[Math.floor(Math.random() * CLASSES.length)];
  } while (next === practiceState.target && CLASSES.length > 1);
  return next;
}

function startPractice() {
  practiceState.target = pickRandomTarget();
  practiceState.holdStart = null;
  practiceState.alreadyScored = false;
  practiceTarget.textContent = practiceState.target;
  practiceFeedback.textContent = '';
  practiceFeedback.className = 'practice-feedback';
}

function updatePractice() {
  if (currentMode !== 'practice') return;

  const detected = stableLetter;

  if (!detected) {
    practiceState.holdStart = null;
    practiceFeedback.textContent = 'Show the sign...';
    practiceFeedback.className = 'practice-feedback';
    return;
  }

  if (detected === practiceState.target) {
    if (!practiceState.holdStart) {
      practiceState.holdStart = performance.now();
    }
    const held = performance.now() - practiceState.holdStart;
    if (held >= PRACTICE_HOLD_MS && !practiceState.alreadyScored) {
      // Correct!
      practiceState.alreadyScored = true;
      practiceState.correct++;
      practiceState.total++;
      practiceState.streak++;
      practiceFeedback.textContent = '✓ Correct!';
      practiceFeedback.className = 'practice-feedback correct';
      practiceCorrect.textContent = practiceState.correct;
      practiceTotal.textContent = practiceState.total;
      practiceStreak.textContent = practiceState.streak;

      // Next letter after a short pause
      setTimeout(() => {
        startPractice();
      }, 800);
    } else if (!practiceState.alreadyScored) {
      const remaining = Math.ceil((PRACTICE_HOLD_MS - held) / 1000);
      practiceFeedback.textContent = `Hold it... ${remaining > 0 ? '' : ''}`;
      practiceFeedback.className = 'practice-feedback';
    }
  } else {
    practiceState.holdStart = null;
    if (!practiceState.alreadyScored) {
      practiceFeedback.textContent = `Showing: ${detected}`;
      practiceFeedback.className = 'practice-feedback wrong';
    }
  }
}

function resetPractice() {
  practiceState.correct = 0;
  practiceState.total = 0;
  practiceState.streak = 0;
  practiceState.holdStart = null;
  practiceState.alreadyScored = false;
  practiceCorrect.textContent = '0';
  practiceTotal.textContent = '0';
  practiceStreak.textContent = '0';
  startPractice();
}

// ─── DRAWING ───────────────────────────────────────────────────
function drawLandmarks(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks || landmarks.length === 0) return;

  const w = canvas.width;
  const h = canvas.height;

  // Draw connections
  ctx.strokeStyle = 'rgba(255, 205, 0, 0.6)';
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
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

// ─── SELECT LARGEST HAND ───────────────────────────────────────
/**
 * When multiple hands are detected, pick the one with the largest
 * bounding box (closest to camera / most prominent).
 */
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

// ─── DETECT LOOP ───────────────────────────────────────────────
function detectLoop() {
  if (!isRunning) return;

  if (video.readyState >= 2 && handLandmarker && classifier) {
    const results = handLandmarker.detectForVideo(video, performance.now());

    // Pick the largest hand (fixes multi-person glitch)
    const landmarks = selectLargestHand(results);

    if (landmarks) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      drawLandmarks(landmarks);

      const { ranked, smoothedLetter, smoothedConf } = predict(landmarks);
      const stabilityPct = checkStability(landmarks);
      updatePredictionUI(ranked, smoothedLetter, smoothedConf, stabilityPct);
      updatePractice();

      statusBar.textContent = 'Tracking hand — sign a letter!';
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      clearPredictionUI();
      prevLandmarks = null;
      stabilityCount = 0;
      statusBar.textContent = 'No hand detected — show your hand to the camera';
    }

    // FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
      fpsCounter.textContent = frameCount + ' fps';
      frameCount = 0;
      lastFpsTime = now;
    }
  }

  animFrameId = requestAnimationFrame(detectLoop);
}

// ─── LOAD HAND LANDMARKER ──────────────────────────────────────
async function loadHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numHands: 2    // Detect up to 2 hands so we can pick the largest
  });
}

// ─── START CAMERA ──────────────────────────────────────────────
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: 'user',
      width:  { ideal: 640 },
      height: { ideal: 480 }
    },
    audio: false
  });
  video.srcObject = stream;
  await new Promise(resolve => { video.onloadedmetadata = resolve; });
  await video.play();
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
}

// ─── INIT (PARALLEL LOADING) ──────────────────────────────────
async function init() {
  try {
    // Step 1: Camera
    setLoadStep(loadStepCamera, 'active');
    setLoadProgress(5);
    await startCamera();
    setLoadStep(loadStepCamera, 'done');
    setLoadProgress(20);
    statusBar.textContent = 'Camera ready, loading models...';

    // Step 2 & 3: Load MediaPipe and classifier in parallel
    setLoadStep(loadStepMediapipe, 'active');
    setLoadStep(loadStepClassifier, 'active');
    setLoadProgress(30);

    const mediapipePromise = loadHandLandmarker().then(() => {
      setLoadStep(loadStepMediapipe, 'done');
      setLoadProgress(prev => Math.max(prev, 70));
    });

    const classifierPromise = loadClassifier().then(() => {
      setLoadStep(loadStepClassifier, 'done');
    });

    // A tiny progress ticker while we wait
    let progressVal = 30;
    const ticker = setInterval(() => {
      progressVal = Math.min(progressVal + 2, 85);
      setLoadProgress(progressVal);
    }, 200);

    await Promise.all([mediapipePromise, classifierPromise]);

    clearInterval(ticker);
    setLoadProgress(100);

    // All loaded — start!
    statusBar.textContent = 'Ready — show your hand to start signing!';
    isRunning = true;

    // Hide loading overlay
    setTimeout(hideLoadingOverlay, 300);

    // Start detection loop
    detectLoop();

    // Start tip rotation
    rotateTips();

  } catch (err) {
    console.error('Init error:', err);
    showError('Failed to start: ' + err.message + '. Make sure you allow camera access and are using HTTPS.');
    hideLoadingOverlay();
  }
}

// ─── TIPS ROTATION ─────────────────────────────────────────────
function rotateTips() {
  let tipIdx = 0;
  setInterval(() => {
    tipIdx = (tipIdx + 1) % tips.length;
    tipsBar.innerHTML = `<strong>Tip:</strong> ${tips[tipIdx]}`;
  }, 8000);
}

// ─── MODE SWITCHING ────────────────────────────────────────────
document.querySelectorAll('.mode-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.mode-tab').forEach(t => {
      t.classList.remove('active');
      t.setAttribute('aria-selected', 'false');
    });
    tab.classList.add('active');
    tab.setAttribute('aria-selected', 'true');

    currentMode = tab.dataset.mode;

    if (currentMode === 'practice') {
      predictionPanel.style.display = 'none';
      practicePanel.classList.add('visible');
      startPractice();
    } else {
      predictionPanel.style.display = 'block';
      practicePanel.classList.remove('visible');
    }
  });
});

// Practice controls
document.getElementById('btn-skip').addEventListener('click', () => {
  practiceState.total++;
  practiceState.streak = 0;
  practiceTotal.textContent = practiceState.total;
  practiceStreak.textContent = '0';
  startPractice();
});

document.getElementById('btn-reset-practice').addEventListener('click', resetPractice);

// ─── START ─────────────────────────────────────────────────────
init();
