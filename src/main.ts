import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';
import FFT from 'fft.js';

// ─── FFmpeg Setup ───
const ffmpeg = new FFmpeg();
let ffmpegLoaded = false;

// ─── State ───
let selectedFile: File | null = null;
let audioData: Float32Array | null = null;
let audioDuration = 0;
let detectedRallies: Rally[] = [];
let detectedPeaks: number[] = [];
let onsetEnvelope: Float32Array | null = null;

// ─── UI Elements ───
const logDiv       = document.getElementById('log') as HTMLDivElement;
const uploadInput  = document.getElementById('videoUpload') as HTMLInputElement;
const startBtn     = document.getElementById('startBtn') as HTMLButtonElement;
const resultDiv    = document.getElementById('result') as HTMLDivElement;
const dropzone     = document.getElementById('dropzone') as HTMLDivElement;
const dropzoneText = document.getElementById('dropzone-text') as HTMLParagraphElement;
const progressContainer = document.getElementById('progressContainer') as HTMLDivElement;
const progressFill = document.getElementById('progressFill') as HTMLDivElement;
const statusText   = document.getElementById('statusText') as HTMLDivElement;
const paramPanel   = document.getElementById('paramPanel') as HTMLDivElement;
const timelineSec  = document.getElementById('timelineSection') as HTMLDivElement;
const rallyList    = document.getElementById('rallyList') as HTMLDivElement;
const rallyCards   = document.getElementById('rallyCards') as HTMLDivElement;
const tooltip      = document.getElementById('tooltip') as HTMLDivElement;

// Canvas
const waveCanvas   = document.getElementById('waveformCanvas') as HTMLCanvasElement;
const timelineCanvas= document.getElementById('timelineCanvas') as HTMLCanvasElement;
const wCtx = waveCanvas.getContext('2d')!;
const tCtx = timelineCanvas.getContext('2d')!;

// Sliders
const deltaSlider  = document.getElementById('deltaSlider') as HTMLInputElement;
const gapSlider     = document.getElementById('gapSlider') as HTMLInputElement;
const minHitsSlider = document.getElementById('minHitsSlider') as HTMLInputElement;
const deltaVal      = document.getElementById('deltaVal') as HTMLSpanElement;
const gapVal        = document.getElementById('gapVal') as HTMLSpanElement;
const minHitsVal    = document.getElementById('minHitsVal') as HTMLSpanElement;
const statRallies   = document.getElementById('statRallies') as HTMLSpanElement;
const statHits      = document.getElementById('statHits') as HTMLSpanElement;
const statDuration  = document.getElementById('statDuration') as HTMLSpanElement;

// ─── Types ───
interface Rally {
  start: number;   // seconds
  end: number;     // seconds
  hits: number;
  hitTimes: number[];
}

// ─── Drag & Drop ───
dropzone.addEventListener('click', () => uploadInput.click());
uploadInput.addEventListener('change', (e) => handleFileSelect((e.target as HTMLInputElement).files?.[0]));
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => {
  dropzone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); }, false);
});
['dragenter', 'dragover'].forEach(ev => dropzone.addEventListener(ev, () => dropzone.classList.add('dragover')));
['dragleave', 'drop'].forEach(ev => dropzone.addEventListener(ev, () => dropzone.classList.remove('dragover')));
dropzone.addEventListener('drop', (e) => handleFileSelect(e.dataTransfer?.files?.[0]));

function handleFileSelect(file?: File) {
  if (file && file.type.startsWith('video/')) {
    selectedFile = file;
    dropzoneText.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
    dropzone.style.borderColor = '#10b981';
    dropzone.style.backgroundColor = '#ecfdf5';
    startBtn.disabled = false;
    // Reset UI
    paramPanel.classList.remove('visible');
    timelineSec.classList.remove('visible');
    rallyList.classList.remove('visible');
    audioData = null;
    detectedRallies = [];
    detectedPeaks = [];
    onsetEnvelope = null;
  } else {
    alert('Please select a valid video file.');
  }
}

// ─── Helpers ───
function log(msg: string, type: 'info'|'warn'|'err' = 'info') {
  const p = document.createElement('p');
  p.textContent = msg;
  if (type === 'warn') p.className = 'warn';
  if (type === 'err')  p.className = 'err';
  logDiv.appendChild(p);
  logDiv.scrollTop = logDiv.scrollHeight;
  console.log(msg);
}

function updateProgress(step: string, percent: number) {
  progressContainer.classList.add('visible');
  statusText.textContent = step;
  progressFill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
}

// ─── FFT / Onset Detection ───
function computeOnsetStrength(data: Float32Array, sr: number): Float32Array {
  const hopLength = 512;
  const nFft = 512;
  const numFrames = Math.floor((data.length - nFft) / hopLength) + 1;
  const diffs = new Float32Array(numFrames);
  const f = new FFT(nFft);
  const out = f.createComplexArray();
  const realInput = new Float32Array(nFft);
  const win = new Float32Array(nFft);
  for (let i = 0; i < nFft; i++)
    win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (nFft - 1)));
  let prevMags = new Float32Array(nFft / 2 + 1);
  for (let i = 0; i < numFrames; i++) {
    const start = i * hopLength;
    for (let j = 0; j < nFft; j++) realInput[j] = data[start + j] * win[j];
    f.realTransform(out, Array.from(realInput));
    let flux = 0;
    for (let k = 0; k <= nFft / 2; k++) {
      const re = out[2*k], im = out[2*k+1];
      const mag = Math.sqrt(re*re + im*im);
      const diff = mag - prevMags[k];
      if (diff > 0) flux += diff;
      prevMags[k] = mag;
    }
    diffs[i] = flux;
  }
  return diffs;
}

function peakPick(x: Float32Array, preMax: number, postMax: number, preAvg: number, postAvg: number, delta: number, wait: number): number[] {
  const peaks: number[] = [];
  for (let i = preMax; i < x.length - postMax; i++) {
    let isMax = true;
    for (let j = i - preMax; j <= i + postMax; j++) if (x[j] > x[i]) { isMax = false; break; }
    if (isMax) {
      let sum = 0, count = 0;
      for (let j = Math.max(0, i - preAvg); j <= Math.min(x.length - 1, i + postAvg); j++) { sum += x[j]; count++; }
      if (x[i] >= (sum / count) + delta)
        if (peaks.length === 0 || i - peaks[peaks.length - 1] >= wait) peaks.push(i);
    }
  }
  return peaks;
}

function detectRallies(peaks: number[], sr: number, hopLength: number, maxInterval: number, minHits: number): Rally[] {
  // peaks are in frames
  const times = peaks.map(p => p * hopLength / sr);
  if (times.length === 0) return [];
  const rallies: Rally[] = [];
  let current: number[] = [times[0]];
  for (let i = 1; i < times.length; i++) {
    if (times[i] - current[current.length - 1] <= maxInterval) {
      current.push(times[i]);
    } else {
      if (current.length >= minHits) {
        const firstHit = current[0];
        const lastHit  = current[current.length - 1];
        rallies.push({ start: Math.max(0, firstHit - 1.5), end: lastHit + 2.5, hits: current.length, hitTimes: [...current] });
      }
      current = [times[i]];
    }
  }
  if (current.length >= minHits) {
    const firstHit = current[0], lastHit = current[current.length - 1];
    rallies.push({ start: Math.max(0, firstHit - 1.5), end: lastHit + 2.5, hits: current.length, hitTimes: [...current] });
  }
  return rallies;
}

// ─── Parameter Change Handler ───
function onParamsChange() {
  if (!audioData || !onsetEnvelope) return;
  const delta = parseFloat(deltaSlider.value);
  const maxGap = parseFloat(gapSlider.value);
  const minHits = parseInt(minHitsSlider.value);

  detectedPeaks = peakPick(onsetEnvelope, 3, 3, 5, 5, delta, 10);
  detectedRallies = detectRallies(detectedPeaks, 16000, 512, maxGap, minHits);

  statRallies.textContent = String(detectedRallies.length);
  statHits.textContent = String(detectedPeaks.length);

  renderTimeline();
  renderRallyCards();
}

deltaSlider.addEventListener('input', () => {
  deltaVal.textContent = deltaSlider.value;
  onParamsChange();
});
gapSlider.addEventListener('input', () => {
  gapVal.textContent = gapSlider.value;
  onParamsChange();
});
minHitsSlider.addEventListener('input', () => {
  minHitsVal.textContent = minHitsSlider.value;
  onParamsChange();
});

// ─── Canvas Rendering ───
function setupCanvases() {
  const wrap = document.getElementById('canvasWrap')!;
  const w = wrap.clientWidth;
  waveCanvas.width   = w;
  waveCanvas.height  = 120;
  timelineCanvas.width  = w;
  timelineCanvas.height = 80;
}

function drawWaveform() {
  if (!audioData) return;
  const W = waveCanvas.width, H = waveCanvas.height;
  wCtx.clearRect(0, 0, W, H);
  wCtx.fillStyle = '#1a1a2e';
  wCtx.fillRect(0, 0, W, H);

  // Downsample to canvas width
  const blockSize = Math.max(1, Math.floor(audioData.length / W));
  wCtx.strokeStyle = '#6366f1';
  wCtx.lineWidth = 1;
  wCtx.beginPath();
  const mid = H / 2;
  for (let x = 0; x < W; x++) {
    const start = x * blockSize;
    let max = 0, min = 0;
    for (let i = 0; i < blockSize && start + i < audioData.length; i++) {
      const v = audioData[start + i];
      if (v > max) max = v;
      if (v < min) min = v;
    }
    const yMax = mid + max * mid * 0.9;
    const yMin = mid + min * mid * 0.9;
    wCtx.moveTo(x, yMax);
    wCtx.lineTo(x, yMin);
  }
  wCtx.stroke();

  // RMS energy overlay
  if (onsetEnvelope) {
    const envBlock = Math.max(1, Math.floor(onsetEnvelope.length / W));
    wCtx.strokeStyle = 'rgba(16,185,129,0.6)';
    wCtx.lineWidth = 1.5;
    wCtx.beginPath();
    for (let x = 0; x < W; x++) {
      let sum = 0;
      for (let i = 0; i < envBlock && x*envBlock+i < onsetEnvelope.length; i++)
        sum += onsetEnvelope[x*envBlock+i];
      const env = sum / envBlock;
      const y = H - env * 300;
      if (x === 0) wCtx.moveTo(x, y);
      else wCtx.lineTo(x, y);
    }
    wCtx.stroke();
  }
}

function renderTimeline() {
  if (!audioData || !onsetEnvelope) return;
  const W = timelineCanvas.width, H = timelineCanvas.height;
  tCtx.clearRect(0, 0, W, H);
  tCtx.fillStyle = '#0f0f23';
  tCtx.fillRect(0, 0, W, H);

  const dur = audioDuration;
  const toX = (t: number) => (t / dur) * W;
  const frameDur = audioData.length / 16000 / dur; // frames per second... wait, onsetEnvelope[i] = frame i

  // Draw rally segments
  const maxGap = parseFloat(gapSlider.value);
  const peaksFrames = detectedPeaks;
  const peaksTimes = peaksFrames.map(p => p * 512 / 16000);

  // Group into segments for coloring
  let inRally = false;
  let segStart = 0;

  for (let i = 0; i <= peaksTimes.length; i++) {
    const t = i < peaksTimes.length ? peaksTimes[i] : dur;
    const gap = i < peaksTimes.length ? peaksTimes[i] - (i > 0 ? peaksTimes[i-1] : 0) : maxGap + 1;
    if (!inRally && i < peaksTimes.length) {
      // Start of a new segment
      inRally = true;
      segStart = t;
    } else if (inRally && gap > maxGap) {
      // End of segment
      tCtx.fillStyle = 'rgba(16,185,129,0.18)';
      const x1 = toX(segStart), x2 = toX(i > 0 ? peaksTimes[i-1] : segStart);
      tCtx.fillRect(x1, 0, x2 - x1, H);
      inRally = false;
    }
  }
  if (inRally && peaksTimes.length > 0) {
    tCtx.fillStyle = 'rgba(16,185,129,0.18)';
    const x1 = toX(segStart), x2 = toX(peaksTimes[peaksTimes.length-1]);
    tCtx.fillRect(x1, 0, x2 - x1, H);
  }

  // Time axis ticks
  tCtx.fillStyle = '#374151';
  tCtx.font = '10px monospace';
  const tickInterval = dur > 300 ? 60 : dur > 60 ? 10 : 5;
  for (let t = 0; t <= dur; t += tickInterval) {
    const x = toX(t);
    tCtx.fillRect(x, H - 6, 1, 6);
    tCtx.fillText(formatTime(t), x + 2, H - 8);
  }

  // Draw hit markers
  for (const t of peaksTimes) {
    const x = toX(t);
    // Glow effect
    const grad = tCtx.createRadialGradient(x, H/2, 0, x, H/2, 8);
    grad.addColorStop(0, '#fbbf24');
    grad.addColorStop(1, 'transparent');
    tCtx.fillStyle = grad;
    tCtx.fillRect(x - 8, 0, 16, H);

    tCtx.beginPath();
    tCtx.arc(x, H/2, 3, 0, Math.PI * 2);
    tCtx.fillStyle = '#fbbf24';
    tCtx.fill();
  }

  // Draw rally labels
  detectedRallies.forEach((r, idx) => {
    const cx = (toX(r.start) + toX(r.end)) / 2;
    const label = `${r.hits}🎾`;
    tCtx.font = 'bold 10px monospace';
    const tw = tCtx.measureText(label).width;
    tCtx.fillStyle = 'rgba(0,0,0,0.6)';
    tCtx.fillRect(cx - tw/2 - 3, H/2 - 9, tw + 6, 14);
    tCtx.fillStyle = '#10b981';
    tCtx.fillText(label, cx - tw/2, H/2 + 3);
  });

  // Hover interaction on timelineCanvas
  timelineCanvas.onmousemove = (e) => {
    const rect = timelineCanvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseT = (mouseX / W) * dur;
    let found = false;
    for (const r of detectedRallies) {
      if (mouseT >= r.start && mouseT <= r.end) {
        tooltip.style.display = 'block';
        tooltip.style.left = `${mouseX + 10}px`;
        tooltip.style.top = `${20}px`;
        tooltip.textContent = `🎾 ${r.hits} hits | ${r.end - r.start:.1f}s | ${formatTime(r.start)} → ${formatTime(r.end)}`;
        found = true;
        break;
      }
    }
    if (!found) tooltip.style.display = 'none';
  };
  timelineCanvas.onmouseleave = () => tooltip.style.display = 'none';
}

function renderRallyCards() {
  rallyCards.innerHTML = '';
  if (detectedRallies.length === 0) return;
  detectedRallies.sort((a, b) => b.hits - a.hits);
  detectedRallies.slice(0, 12).forEach((r, i) => {
    const card = document.createElement('div');
    card.className = 'rally-card';
    card.innerHTML = `
      <div class="card-title">Rally #${i+1}</div>
      <div class="card-meta">⏱ ${(r.end - r.start).toFixed(1)}s &nbsp; 🎾 ${r.hits} hits</div>
      <div class="card-meta" style="opacity:0.7">${formatTime(r.start)} — ${formatTime(r.end)}</div>
    `;
    card.addEventListener('click', () => {
      const W = timelineCanvas.width;
      const xStart = (r.start / audioDuration) * W;
      const xEnd   = (r.end   / audioDuration) * W;
      tCtx.clearRect(0, 0, W, timelineCanvas.height);
      renderTimeline();
      // Highlight selected
      tCtx.strokeStyle = '#f59e0b';
      tCtx.lineWidth = 3;
      tCtx.strokeRect(xStart, 2, xEnd - xStart, timelineCanvas.height - 4);
    });
    rallyCards.appendChild(card);
  });
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

// ─── FFmpeg Loading ───
async function loadFFmpeg() {
  if (ffmpegLoaded) return;
  log('Loading FFmpeg (this takes a moment)...');
  ffmpeg.on('log', ({ message }) => {
    if (message.includes('frame=') || message.includes('size=')) return;
    log(message);
  });
  ffmpeg.on('progress', ({ progress }) => {
    if (progress > 0 && progress <= 1)
      progressFill.style.width = `${progress * 100}%`;
  });
  const baseURL = new URL(import.meta.env.BASE_URL, window.location.origin).href;
  const cb = '?v=' + Date.now();
  await ffmpeg.load({
    coreURL: `${baseURL}ffmpeg-core.js${cb}`,
    wasmURL: `${baseURL}ffmpeg-core.wasm${cb}`,
  });
  ffmpegLoaded = true;
  log('FFmpeg loaded successfully!');
}

// ─── Main Processing ───
startBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  startBtn.disabled = true;
  resultDiv.innerHTML = '';
  logDiv.innerHTML = '';
  logDiv.classList.add('visible');
  progressContainer.classList.add('visible');

  try {
    updateProgress('Loading FFmpeg Environment...', 5);
    await loadFFmpeg();

    updateProgress('Loading Video File...', 15);
    log('Writing video to virtual memory...');
    await ffmpeg.writeFile('input.mp4', await fetchFile(selectedFile));

    updateProgress('Extracting 16kHz Audio...', 30);
    log('Extracting audio for acoustic analysis...');
    await ffmpeg.exec(['-y', '-i', 'input.mp4', '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav']);

    updateProgress('Reading Audio Data...', 45);
    log('Reading audio buffer...');
    const rawData = await ffmpeg.readFile('temp.wav');
    const audioBuffer = new AudioContext({ sampleRate: 16000 }).decodeAudioData((rawData as Uint8Array).buffer.slice(0));
    const decoded = await audioBuffer;
    audioData = decoded.getChannelData(0);
    audioDuration = audioData.length / 16000;
    statDuration.textContent = formatTime(audioDuration);

    updateProgress('Running Spectral Flux Analysis...', 55);
    log('Computing onset strength envelope...');
    onsetEnvelope = computeOnsetStrength(audioData, 16000);

    // Show param panel and timeline
    paramPanel.classList.add('visible');
    timelineSec.classList.add('visible');
    setupCanvases();
    drawWaveform();

    // Auto-detect environment noise level for smart defaults
    const sensitivePeaks = peakPick(onsetEnvelope, 3, 3, 5, 5, 1.0, 10);
    const hitsPerMin = sensitivePeaks.length / (audioDuration / 60);
    if (hitsPerMin > 60) {
      log(`⚠️  High noise court detected (${hitsPerMin.toFixed(0)} raw hits/min). Using Anti-Interference Mode.`, 'warn');
      deltaSlider.value = '5.0';
      gapSlider.value = '4.0';
    } else {
      log(`✅  Quiet environment detected (${hitsPerMin.toFixed(0)} raw hits/min). Using High-Sensitivity Mode.`);
      deltaSlider.value = '2.0';
    }
    deltaVal.textContent = deltaSlider.value;
    gapVal.textContent = gapSlider.value;
    minHitsVal.textContent = minHitsSlider.value;

    // Initial detection
    onParamsChange();
    log(`🎾 Initial scan: ${detectedPeaks.length} hits found, ${detectedRallies.length} rallies detected.`);

    if (detectedRallies.length === 0) {
      log('⚠️  No rallies found with current settings. Try lowering Delta or Max Gap.', 'warn');
      startBtn.disabled = false;
      return;
    }

    updateProgress('Extracting Top Rallies...', 70);
    detectedRallies.sort((a, b) => (b.end - b.start) - (a.end - a.start));
    const topRallies = detectedRallies.slice(0, Math.min(5, detectedRallies.length));
    rallyList.classList.add('visible');

    let concatList = '';
    for (let i = 0; i < topRallies.length; i++) {
      const r = topRallies[i];
      const clipName = `clip_${i}.mp4`;
      const dur = r.end - r.start;
      updateProgress(`Clipping Rally ${i+1}/${topRallies.length} (${r.hits} hits)...`, 70 + (25 / topRallies.length) * i);
      log(`✂️  Clipping Rally ${i+1}: ${r.hits} hits, ${dur.toFixed(1)}s...`);
      await ffmpeg.exec([
        '-y', '-ss', r.start.toString(), '-i', 'input.mp4', '-t', dur.toString(),
        '-c:v', 'copy', '-c:a', 'copy', clipName
      ]);
      concatList += `file '${clipName}'\n`;
    }

    updateProgress('Stitching Highlights...', 95);
    await ffmpeg.writeFile('concat.txt', concatList);
    log('Stitching all highlights together...');
    await ffmpeg.exec(['-y', '-f', 'concat', '-safe', '0', '-i', 'concat.txt', '-c', 'copy', 'highlights.mp4']);

    updateProgress('Done!', 100);
    log('✅ Compilation ready! Preparing download...');
    const finalData = await ffmpeg.readFile('highlights.mp4');
    const blob = new Blob([(finalData as Uint8Array).buffer], { type: 'video/mp4' });
    const url = URL.createObjectURL(blob);
    resultDiv.innerHTML = `
      <h3 style="margin-bottom:0.5rem">🎉 ${detectedRallies.length} Rallies Found — Top ${topRallies.length} Compiled</h3>
      <p style="color:#6b7280;margin-bottom:1rem;font-size:0.9rem">You can adjust parameters above and re-analyze, or download the result below.</p>
      <video src="${url}" controls preload="metadata"></video>
      <br/>
      <a href="${url}" class="download-btn" download="tennis_highlights.mp4">Download Highlights (${topRallies.length} rallies)</a>
    `;
  } catch (err: any) {
    updateProgress('Error', 0);
    log('ERROR: ' + (err.message || String(err)), 'err');
    console.error(err);
  } finally {
    startBtn.disabled = false;
  }
});

// ─── Window Resize ───
window.addEventListener('resize', () => {
  if (audioData) {
    setupCanvases();
    drawWaveform();
    renderTimeline();
  }
});
