import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';
import FFT from 'fft.js';

const ffmpeg = new FFmpeg();
let ffmpegLoaded = false;

let selectedFile: File | null = null;
let audioData: Float32Array | null = null;
let audioDuration = 0;
let onsetEnvelope: Float32Array | null = null;
let rmsEnvelope: Float32Array | null = null;
let detectedRallies: Rally[] = [];
let detectedPeaks: number[] = [];
let analysisDone = false;
let extractionDone = false;
let currentView: 'grid' | 'list' = 'grid';
let rallyClips: Map<number, { blob: Blob; url: string }> = new Map();
let activeCardId: number | null = null;
let activeClipIdx: number | null = null;

interface Rally {
  id: number; start: number; end: number; hits: number; hitTimes: number[]; rmsEnergy: number;
}

const logDiv = document.getElementById('log') as HTMLDivElement;
const uploadInput = document.getElementById('videoUpload') as HTMLInputElement;
const startBtn = document.getElementById('startBtn') as HTMLButtonElement;
const dropzone = document.getElementById('dropzone') as HTMLDivElement;
const progressContainer = document.getElementById('progressContainer') as HTMLDivElement;
const progressFill = document.getElementById('progressFill') as HTMLDivElement;
const statusText = document.getElementById('statusText') as HTMLDivElement;
const paramPanel = document.getElementById('paramPanel') as HTMLDivElement;
const timelineSec = document.getElementById('timelineSection') as HTMLDivElement;
const rallyList = document.getElementById('rallyList') as HTMLDivElement;
const rallyCardsEl = document.getElementById('rallyCards') as HTMLDivElement;
const clipGridView = document.getElementById('clipGridView') as HTMLDivElement;
const clipListView = document.getElementById('clipListView') as HTMLDivElement;
const clipCardsEl = document.getElementById('clipCards') as HTMLDivElement;
const clipListEl = document.getElementById('clipListView') as HTMLDivElement;
const viewToggle = document.getElementById('viewToggle') as HTMLDivElement;
const gridViewBtn = document.getElementById('gridViewBtn') as HTMLButtonElement;
const listViewBtn = document.getElementById('listViewBtn') as HTMLButtonElement;
const rallyCountEl = document.getElementById('rallyCount') as HTMLSpanElement;
const clipCountEl = document.getElementById('clipCount') as HTMLSpanElement;
const tooltip = document.getElementById('tooltip') as HTMLDivElement;
const waveCanvas = document.getElementById('waveformCanvas') as HTMLCanvasElement;
const timelineCanvas = document.getElementById('timelineCanvas') as HTMLCanvasElement;
const wCtx = waveCanvas.getContext('2d')!;
const tCtx = timelineCanvas.getContext('2d')!;

const deltaSlider = document.getElementById('deltaSlider') as HTMLInputElement;
const gapSlider = document.getElementById('gapSlider') as HTMLInputElement;
const energySlider = document.getElementById('energySlider') as HTMLInputElement;

const deltaVal = document.getElementById('deltaVal') as HTMLSpanElement;
const gapVal = document.getElementById('gapVal') as HTMLSpanElement;
const energyVal = document.getElementById('energyVal') as HTMLSpanElement;
const statRallies = document.getElementById('statRallies') as HTMLSpanElement;
const statHits = document.getElementById('statHits') as HTMLSpanElement;
const statDuration = document.getElementById('statDuration') as HTMLSpanElement;

function log(msg: string, type: 'info'|'warn'|'err' = 'info') {
  const p = document.createElement('p');
  p.textContent = msg;
  if (type === 'warn') p.className = 'warn';
  if (type === 'err') p.className = 'err';
  logDiv.appendChild(p);
  logDiv.scrollTop = logDiv.scrollHeight;
  console.log(msg);
}

function updateProgress(step: string, pct: number) {
  progressContainer.classList.add('visible');
  statusText.textContent = step;
  progressFill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
}

function fmtTime(s: number): string {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

function computeOnsetStrength(data: Float32Array): Float32Array {
  const hopLength = 512, nFft = 512;
  const numFrames = Math.floor((data.length - nFft) / hopLength) + 1;
  const diffs = new Float32Array(numFrames);
  const f = new FFT(nFft);
  const out = f.createComplexArray();
  const realInput = new Float32Array(nFft);
  const win = new Float32Array(nFft);
  for (let i = 0; i < nFft; i++) win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (nFft - 1)));
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

function computeRMS(data: Float32Array): Float32Array {
  const hopLength = 512, nFft = 512;
  const numFrames = Math.floor((data.length - nFft) / hopLength) + 1;
  const rms = new Float32Array(numFrames);
  for (let i = 0; i < numFrames; i++) {
    let sum = 0;
    const start = i * hopLength;
    for (let j = 0; j < nFft; j++) sum += data[start + j] * data[start + j];
    rms[i] = Math.sqrt(sum / nFft);
  }
  return rms;
}

function peakPick(x: Float32Array, delta: number, wait: number): number[] {
  const peaks: number[] = [];
  for (let i = 3; i < x.length - 3; i++) {
    let isMax = true;
    for (let j = i - 3; j <= i + 3; j++) if (x[j] > x[i]) { isMax = false; break; }
    if (isMax) {
      let sum = 0, count = 0;
      for (let j = Math.max(0, i - 5); j <= Math.min(x.length - 1, i + 5); j++) { sum += x[j]; count++; }
      if (x[i] >= (sum / count) + delta)
        if (peaks.length === 0 || i - peaks[peaks.length - 1] >= wait) peaks.push(i);
    }
  }
  return peaks;
}

function detectRallies(peaks: number[], sr: number, hopLength: number, maxInterval: number, rms: Float32Array, energyThresh: number): Rally[] {
  const times = peaks.map(p => p * hopLength / sr);
  if (times.length === 0) return [];
  const rallies: Rally[] = [];
  let current: number[] = [times[0]];
  const flush = (i: number) => {
    if (current.length >= minHits) {
      const first = current[0], last = current[current.length - 1];
      const dur = last - first;
      if (dur >= 1 && dur <= 120) {
        const midFrame = Math.floor((first + last) / 2 * sr / hopLength);
        const energy = rms[Math.max(0, Math.min(midFrame, rms.length - 1))];
        if (energy >= energyThresh) {
          rallies.push({ id: rallies.length, start: Math.max(0, first - 1.5), end: last + 2.5, hits: current.length, hitTimes: [...current], rmsEnergy: energy });
        }
      }
    }
    current = [times[i]];
  };
  for (let i = 1; i < times.length; i++) {
    if (times[i] - current[current.length - 1] <= maxInterval) current.push(times[i]);
    else flush(i);
  }
  if (current.length >= 2) {
    const first = current[0], last = current[current.length - 1];
    const dur = last - first;
    if (dur >= 1 && dur <= 120) {
      const midFrame = Math.floor((first + last) / 2 * sr / hopLength);
      const energy = rms[Math.max(0, Math.min(midFrame, rms.length - 1))];
      if (energy >= energyThresh) rallies.push({ id: rallies.length, start: Math.max(0, first - 1.5), end: last + 2.5, hits: current.length, hitTimes: [...current], rmsEnergy: energy });
    }
  }
  return rallies;
}

function onParamsChange() {
  if (!audioData || !onsetEnvelope || !rmsEnvelope) return;
  const delta = parseFloat(deltaSlider.value);
  const maxGap = parseFloat(gapSlider.value);
  const energyThresh = parseFloat(energySlider.value);
  detectedPeaks = peakPick(onsetEnvelope, delta, 10);
  detectedRallies = detectRallies(detectedPeaks, 16000, 512, maxGap, rmsEnvelope, energyThresh);
  statRallies.textContent = String(detectedRallies.length);
  statHits.textContent = String(detectedPeaks.length);
  rallyCountEl.textContent = `(${detectedRallies.length} detected)`;
  renderTimeline();
  renderRallyCards();
}

deltaSlider.addEventListener('input', () => { deltaVal.textContent = deltaSlider.value; onParamsChange(); });
gapSlider.addEventListener('input', () => { gapVal.textContent = gapSlider.value; onParamsChange(); });
energySlider.addEventListener('input', () => { energyVal.textContent = parseFloat(energySlider.value).toFixed(3); onParamsChange(); });

// View Toggle
gridViewBtn.addEventListener('click', () => {
  currentView = 'grid';
  gridViewBtn.style.cssText = 'padding:0.3rem 0.75rem;border:none;background:white;border-radius:5px;font-size:0.75rem;font-weight:600;color:#1f2937;cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,0.1);';
  listViewBtn.style.cssText = 'padding:0.3rem 0.75rem;border:none;background:transparent;border-radius:5px;font-size:0.75rem;font-weight:600;color:#6b7280;cursor:pointer;';
  clipGridView.style.display = 'block'; clipListView.style.display = 'none';
});
listViewBtn.addEventListener('click', () => {
  currentView = 'list';
  listViewBtn.style.cssText = 'padding:0.3rem 0.75rem;border:none;background:white;border-radius:5px;font-size:0.75rem;font-weight:600;color:#1f2937;cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,0.1);';
  gridViewBtn.style.cssText = 'padding:0.3rem 0.75rem;border:none;background:transparent;border-radius:5px;font-size:0.75rem;font-weight:600;color:#6b7280;cursor:pointer;';
  clipGridView.style.display = 'none'; clipListView.style.display = 'flex';
});

function setupCanvases() {
  const wrap = document.getElementById('canvasWrap')!;
  const w = wrap.clientWidth;
  waveCanvas.width = w; waveCanvas.height = 90;
  timelineCanvas.width = w; timelineCanvas.height = 68;
}

function drawWaveform() {
  if (!audioData) return;
  const W = waveCanvas.width, H = waveCanvas.height;
  wCtx.clearRect(0, 0, W, H);
  wCtx.fillStyle = '#1a1a2e'; wCtx.fillRect(0, 0, W, H);
  const blockSize = Math.max(1, Math.floor(audioData.length / W));
  const mid = H / 2;
  wCtx.strokeStyle = '#6366f1'; wCtx.lineWidth = 1; wCtx.beginPath();
  for (let x = 0; x < W; x++) {
    let max = 0, min = 0;
    const base = x * blockSize;
    for (let i = 0; i < blockSize && base + i < audioData.length; i++) { const v = audioData[base + i]; if (v > max) max = v; if (v < min) min = v; }
    wCtx.moveTo(x, mid + max * mid * 0.9); wCtx.lineTo(x, mid + min * mid * 0.9);
  }
  wCtx.stroke();
  if (onsetEnvelope) {
    const envBlock = Math.max(1, Math.floor(onsetEnvelope.length / W));
    const maxEnv = Math.max(...Array.from(onsetEnvelope.slice(0, 5000)), 1e-10);
    wCtx.strokeStyle = 'rgba(16,185,129,0.7)'; wCtx.lineWidth = 1.5; wCtx.beginPath();
    for (let x = 0; x < W; x++) {
      let sum = 0;
      for (let i = 0; i < envBlock && x*envBlock+i < onsetEnvelope.length; i++) sum += onsetEnvelope[x*envBlock+i];
      const env = (sum / envBlock) / maxEnv; const y = H - env * H * 0.88;
      if (x === 0) wCtx.moveTo(x, y); else wCtx.lineTo(x, y);
    }
    wCtx.stroke();
  }
}

function renderTimeline() {
  if (!audioData || !onsetEnvelope) return;
  const W = timelineCanvas.width, H = timelineCanvas.height;
  tCtx.clearRect(0, 0, W, H);
  tCtx.fillStyle = '#0f0f23'; tCtx.fillRect(0, 0, W, H);
  const dur = audioDuration;
  const toX = (t: number) => (t / dur) * W;
  const peaksTimes = detectedPeaks.map(p => p * 512 / 16000);
  const sorted = [...detectedRallies].sort((a, b) => a.start - b.start);
  let lastEnd = 0;
  for (const r of sorted) {
    if (r.start > lastEnd + 0.5) { tCtx.fillStyle = 'rgba(239,68,68,0.12)'; tCtx.fillRect(toX(lastEnd), 0, toX(r.start) - toX(lastEnd), H); }
    lastEnd = r.end;
  }
  if (lastEnd < dur - 0.5) { tCtx.fillStyle = 'rgba(239,68,68,0.12)'; tCtx.fillRect(toX(lastEnd), 0, toX(dur) - toX(lastEnd), H); }
  for (const r of detectedRallies) {
    const x1 = toX(r.start), x2 = toX(r.end);
    tCtx.fillStyle = 'rgba(16,185,129,0.22)'; tCtx.fillRect(x1, 0, x2 - x1, H);
    tCtx.strokeStyle = 'rgba(16,185,129,0.6)'; tCtx.lineWidth = 1; tCtx.strokeRect(x1, 0, x2 - x1, H);
    const cx = (x1 + x2) / 2; const segW = x2 - x1;
    if (segW > 28) {
      const label = `🎾 ${r.hits}`;
      const sublabel = `${(r.end - r.start).toFixed(1)}s`;
      tCtx.font = 'bold 11px monospace';
      const tw = tCtx.measureText(label).width, tw2 = tCtx.measureText(sublabel).width;
      const maxTw = Math.max(tw, tw2);
      if (maxTw + 12 < segW) {
        tCtx.fillStyle = 'rgba(0,0,0,0.6)'; tCtx.fillRect(cx - maxTw/2 - 5, H/2 - 16, maxTw + 10, 32);
        tCtx.fillStyle = '#10b981'; tCtx.textAlign = 'center'; tCtx.fillText(label, cx, H/2 - 4);
        tCtx.font = '9px monospace'; tCtx.fillStyle = '#6ee7b7'; tCtx.fillText(sublabel, cx, H/2 + 10);
      } else {
        tCtx.font = 'bold 10px monospace';
        const tw = tCtx.measureText(label).width;
        tCtx.fillStyle = 'rgba(0,0,0,0.55)'; tCtx.fillRect(cx - tw/2 - 3, H/2 - 8, tw + 6, 16);
        tCtx.fillStyle = '#10b981'; tCtx.textAlign = 'center'; tCtx.fillText(label, cx, H/2 + 4);
      }
      tCtx.textAlign = 'left';
    }
  }
  for (const t of peaksTimes) {
    const x = toX(t);
    const grad = tCtx.createRadialGradient(x, H/2, 0, x, H/2, 7);
    grad.addColorStop(0, '#fbbf24'); grad.addColorStop(1, 'transparent');
    tCtx.fillStyle = grad; tCtx.fillRect(x - 7, 0, 14, H);
    tCtx.beginPath(); tCtx.arc(x, H/2, 2.5, 0, Math.PI * 2);
    tCtx.fillStyle = '#fbbf24'; tCtx.fill();
  }
  tCtx.fillStyle = '#374151'; tCtx.font = '9px monospace';
  const tickInterval = dur > 300 ? 60 : dur > 60 ? 10 : 5;
  for (let t = 0; t <= dur; t += tickInterval) { const x = toX(t); tCtx.fillRect(x, H - 5, 1, 5); tCtx.fillText(fmtTime(t), x + 2, H - 7); }
  timelineCanvas.onmousemove = (e) => {
    const rect = timelineCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left; const mT = (mx / W) * dur;
    let found: Rally | null = null;
    for (const r of detectedRallies) { if (mT >= r.start && mT <= r.end) { found = r; break; } }
    tooltip.style.display = found ? 'block' : 'none';
    if (found) { tooltip.style.left = `${mx}px`; tooltip.style.top = '8px'; tooltip.textContent = `🎾 ${found.hits} hits · ${(found.end - found.start).toFixed(1)}s · RMS ${found.rmsEnergy.toFixed(4)} · ${fmtTime(found.start)} → ${fmtTime(found.end)}`; }
  };
  timelineCanvas.onmouseleave = () => tooltip.style.display = 'none';
  timelineCanvas.onclick = () => { if (detectedRallies.length > 0) document.getElementById(`rally-card-${detectedRallies[0].id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' }); };
}

function renderRallyCards() {
  rallyCardsEl.innerHTML = '';
  if (detectedRallies.length === 0) { rallyCardsEl.innerHTML = `<div style="grid-column:1/-1;text-align:center;padding:1.5rem;color:#9ca3af;font-size:0.88rem;">🎾 No rallies found. Adjust parameters above.</div>`; return; }
  const sorted = [...detectedRallies].sort((a, b) => (b.end - b.start) - (a.end - a.start));
  sorted.forEach((r, idx) => {
    const isLongest = idx === 0;
    const card = document.createElement('div');
    card.className = 'rally-card';
    card.id = `rally-card-${r.id}`;
    card.innerHTML = `
      <div class="card-thumb"><div class="thumb-placeholder"><span style="font-size:1.4rem">🎾</span><span>${r.hits} hits</span><span style="opacity:0.5">${(r.end - r.start).toFixed(1)}s</span></div></div>
      <div class="card-info">
        <div class="card-title"><span class="rank-badge">#${idx + 1}</span>${isLongest ? '<span class="longest-badge">★ Longest</span>' : ''}</div>
        <div class="card-meta"><span>⏱ ${(r.end - r.start).toFixed(1)}s</span><span>·</span><span>🎾 ${r.hits} hits</span><span>·</span><span>${fmtTime(r.start)} → ${fmtTime(r.end)}</span></div>
        <div class="card-meta" style="margin-top:0.25rem;opacity:0.7">RMS ${r.rmsEnergy.toFixed(4)}</div>
      </div>`;
    rallyCardsEl.appendChild(card);
  });
}

function renderClipGrid() {
  clipCardsEl.innerHTML = '';
  if (rallyClips.size === 0) return;
  const sorted = [...detectedRallies].sort((a, b) => (b.end - b.start) - (a.end - a.start));
  clipCountEl.textContent = `${rallyClips.size} clips`;
  sorted.forEach((r, idx) => {
    const clip = rallyClips.get(r.id);
    if (!clip) return;
    const isLongest = idx === 0;
    const card = document.createElement('div');
    card.className = 'clip-card';
    card.id = `clip-card-${r.id}`;
    card.innerHTML = `
      <div class="card-thumb">
        <div class="inline-player" id="player-${r.id}"><video id="video-${r.id}" src="${clip.url}" controls></video><button class="close-inline" id="close-${r.id}">✕</button></div>
        <div class="play-overlay" id="playbtn-${r.id}"><div class="play-circle">▶</div></div>
      </div>
      <div class="card-info">
        <div class="card-title"><span class="rank-badge">#${idx + 1}</span>${isLongest ? '<span class="longest-badge">★ Longest</span>' : ''}</div>
        <div class="card-meta"><span>⏱ ${(r.end - r.start).toFixed(1)}s</span><span>·</span><span>🎾 ${r.hits} hits</span><span>·</span><span>${fmtTime(r.start)} → ${fmtTime(r.end)}</span></div>
      </div>
      <div class="card-actions"><div style="flex:1"></div><a href="${clip.url}" class="dl-btn outline" download="rally_${idx+1}_${r.hits}hits.mp4">⬇</a></div>`;
    const playBtn = card.querySelector(`#playbtn-${r.id}`); const playerEl = card.querySelector(`#player-${r.id}`); const closeBtn = card.querySelector(`#close-${r.id}`); const videoEl = card.querySelector(`#video-${r.id}`) as HTMLVideoElement;
    playBtn?.addEventListener('click', (e) => { e.stopPropagation(); closeActiveCard(); activeCardId = r.id; card.classList.add('active'); if (playBtn) playBtn.setAttribute('style', 'display:none'); if (playerEl) playerEl.setAttribute('style', 'display:block'); videoEl?.play(); });
    closeBtn?.addEventListener('click', (e) => { e.stopPropagation(); closeActiveCard(); });
    videoEl?.addEventListener('ended', () => closeActiveCard());
    card.addEventListener('click', (e) => {
      const t = e.target as HTMLElement;
      if (t.closest('.dl-btn') || t.closest('.close-inline') || t.closest('a')) return;
      if (activeCardId === r.id) { closeActiveCard(); return; }
      if (activeCardId !== null) closeActiveCard();
      activeCardId = r.id; card.classList.add('active');
      const pb = card.querySelector(`#playbtn-${r.id}`); const pl = card.querySelector(`#player-${r.id}`);
      if (pb) pb.setAttribute('style', 'display:none'); if (pl) pl.setAttribute('style', 'display:block');
      (card.querySelector(`#video-${r.id}`) as HTMLVideoElement)?.play();
    });
    clipCardsEl.appendChild(card);
  });
}

function closeActiveCard() {
  if (activeCardId === null) return;
  const prev = document.getElementById(`clip-card-${activeCardId}`);
  if (prev) {
    prev.classList.remove('active');
    const pb = prev.querySelector(`[id^="playbtn-"]`); const pl = prev.querySelector(`[id^="player-"]`);
    if (pb) pb.setAttribute('style', ''); if (pl) pl.setAttribute('style', '');
    const vid = prev.querySelector('video') as HTMLVideoElement; vid?.pause(); vid!.currentTime = 0;
  }
  activeCardId = null;
}

let clipIdxMap: Map<number, number> = new Map(); // rally.id -> display index

function renderClipList() {
  clipListEl.innerHTML = '';
  clipIdxMap.clear();
  if (rallyClips.size === 0) return;
  const sorted = [...detectedRallies].sort((a, b) => (b.end - b.start) - (a.end - a.start));
  sorted.forEach((r, idx) => {
    const clip = rallyClips.get(r.id);
    if (!clip) return;
    clipIdxMap.set(r.id, idx);
    const item = document.createElement('div');
    item.className = 'clip-item';
    item.id = `clip-item-${r.id}`;
    item.innerHTML = `
      <div class="clip-item-header">
        <div class="clip-num">${idx + 1}</div>
        <div class="clip-meta">
          <div class="clip-meta-title">🎾 ${r.hits} hits ${idx === 0 ? '<span class="longest-badge">★ Longest</span>' : ''}</div>
          <div class="clip-meta-sub">⏱ ${(r.end - r.start).toFixed(1)}s · RMS ${r.rmsEnergy.toFixed(4)} · ${fmtTime(r.start)} → ${fmtTime(r.end)}</div>
        </div>
        <div class="clip-expand-icon">▼</div>
      </div>
      <div class="clip-item-body">
        <video id="clipvid-${r.id}" src="${clip.url}" controls preload="metadata"></video>
        <div class="clip-nav">
          <button class="clip-nav-btn" id="prev-btn-${r.id}" ${idx === 0 ? 'disabled' : ''}>◀ Prev</button>
          <button class="clip-nav-btn" id="next-btn-${r.id}" ${idx === sorted.length - 1 ? 'disabled' : ''}>Next ▶</button>
        </div>
        <div class="clip-actions"><a href="${clip.url}" class="dl-btn outline" download="rally_seq_${idx+1}_${r.hits}hits.mp4" style="flex:0 0 auto;padding:0.4rem 0.8rem;">⬇ Download</a></div>
      </div>`;
    // Expand/collapse
    item.querySelector('.clip-item-header')?.addEventListener('click', () => {
      // Pause all other videos first
      document.querySelectorAll('.clip-item video').forEach(v => (v as HTMLVideoElement).pause());
      const isExpanded = item.classList.contains('expanded');
      document.querySelectorAll('.clip-item.expanded').forEach(el => el.classList.remove('expanded'));
      if (!isExpanded) {
        item.classList.add('expanded');
        activeClipIdx = idx;
        const vid = document.getElementById(`clipvid-${r.id}`) as HTMLVideoElement;
        if (vid) { vid.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); vid.play().catch(() => {}); }
      } else {
        activeClipIdx = null;
      }
    });
    // Prev button
    item.querySelector(`#prev-btn-${r.id}`)?.addEventListener('click', (e) => {
      e.stopPropagation();
      const vid = document.getElementById(`clipvid-${r.id}`) as HTMLVideoElement;
      if (vid) vid.pause();
      const prevR = sorted[idx - 1];
      if (prevR) {
        document.querySelectorAll('.clip-item.expanded').forEach(el => el.classList.remove('expanded'));
        const prevItem = document.getElementById(`clip-item-${prevR.id}`);
        if (prevItem) {
          prevItem.classList.add('expanded');
          activeClipIdx = idx - 1;
          const prevVid = document.getElementById(`clipvid-${prevR.id}`) as HTMLVideoElement;
          if (prevVid) { prevVid.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); prevVid.play().catch(() => {}); }
        }
      }
    });
    // Next button
    item.querySelector(`#next-btn-${r.id}`)?.addEventListener('click', (e) => {
      e.stopPropagation();
      const vid = document.getElementById(`clipvid-${r.id}`) as HTMLVideoElement;
      if (vid) vid.pause();
      const nextR = sorted[idx + 1];
      if (nextR) {
        document.querySelectorAll('.clip-item.expanded').forEach(el => el.classList.remove('expanded'));
        const nextItem = document.getElementById(`clip-item-${nextR.id}`);
        if (nextItem) {
          nextItem.classList.add('expanded');
          activeClipIdx = idx + 1;
          const nextVid = document.getElementById(`clipvid-${nextR.id}`) as HTMLVideoElement;
          if (nextVid) { nextVid.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); nextVid.play().catch(() => {}); }
        }
      }
    });
    clipListEl.appendChild(item);
  });
}

async function loadFFmpeg() {
  if (ffmpegLoaded) return;
  log('Loading FFmpeg...');
  ffmpeg.on('log', ({ message }) => { if (!message.includes('frame=') && !message.includes('size=')) log(message); });
  ffmpeg.on('progress', ({ progress }) => { if (progress > 0 && progress <= 1) progressFill.style.width = `${progress * 100}%`; });
  const baseURL = new URL(import.meta.env.BASE_URL, window.location.origin).href;
  const cb = '?v=' + Date.now();
  await ffmpeg.load({ coreURL: `${baseURL}ffmpeg-core.js${cb}`, wasmURL: `${baseURL}ffmpeg-core.wasm${cb}` });
  ffmpegLoaded = true; log('FFmpeg ready.');
}

// Upload
dropzone.addEventListener('click', () => { if (!dropzone.classList.contains('minimized')) uploadInput.click(); });
uploadInput.addEventListener('change', (e) => handleFileSelect((e.target as HTMLInputElement).files?.[0]));
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => { dropzone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); }, false); });
['dragenter', 'dragover'].forEach(ev => dropzone.addEventListener(ev, () => dropzone.classList.add('dragover')));
['dragleave', 'drop'].forEach(ev => dropzone.addEventListener(ev, () => dropzone.classList.remove('dragover')));
dropzone.addEventListener('drop', (e) => handleFileSelect(e.dataTransfer?.files?.[0]));

function handleFileSelect(file?: File) {
  if (file && file.type.startsWith('video/')) {
    selectedFile = file; analysisDone = false; extractionDone = false;
    rallyClips.forEach(({ url }) => URL.revokeObjectURL(url)); rallyClips.clear(); activeCardId = null;
    audioData = null; onsetEnvelope = null; rmsEnvelope = null; detectedRallies = []; detectedPeaks = [];
    dropzone.classList.add('minimized');
    dropzone.innerHTML = `<div class="icon">🎾</div><p>${file.name} <span style="opacity:0.6;font-size:0.75em">(${(file.size/1024/1024).toFixed(1)} MB)</span></p><button class="browse-btn" id="changeFileBtn">Change</button>`;
    document.getElementById('changeFileBtn')?.addEventListener('click', (e) => { e.stopPropagation(); uploadInput.click(); });
    paramPanel.classList.remove('visible'); timelineSec.classList.remove('visible');
    rallyList.classList.remove('visible'); viewToggle.style.display = 'none';
    clipGridView.style.display = 'none'; clipListView.style.display = 'none';
    progressContainer.classList.remove('visible');
    startBtn.textContent = '⏳ Analyzing...'; startBtn.disabled = true;
    logDiv.innerHTML = ''; logDiv.classList.remove('visible');
    runAnalysis();
  }
}

// Phase 1: Auto analysis
async function runAnalysis() {
  if (!selectedFile) return;
  logDiv.innerHTML = ''; logDiv.classList.add('visible'); progressContainer.classList.add('visible');
  log('Starting audio analysis...');
  try {
    await loadFFmpeg();
    await ffmpeg.writeFile('input.mp4', await fetchFile(selectedFile));
    log('Extracting audio...');
    await ffmpeg.exec(['-y', '-i', 'input.mp4', '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav']);
    log('Reading audio data...');
    const rawData = await ffmpeg.readFile('temp.wav');
    const audioCtx = new AudioContext({ sampleRate: 16000 });
    const decoded = await audioCtx.decodeAudioData((rawData as Uint8Array).buffer.slice(0));
    audioData = decoded.getChannelData(0); audioDuration = audioData.length / 16000;
    statDuration.textContent = fmtTime(audioDuration);
    log('Computing onset & energy...');
    onsetEnvelope = computeOnsetStrength(audioData);
    rmsEnvelope = computeRMS(audioData);
    paramPanel.classList.add('visible'); timelineSec.classList.add('visible');
    rallyList.classList.add('visible'); viewToggle.style.display = 'flex';
    setupCanvases(); drawWaveform();
    const sensitivePeaks = peakPick(onsetEnvelope, 1.0, 10);
    const hitsPerMin = sensitivePeaks.length / (audioDuration / 60);
    if (hitsPerMin > 60) { log(`⚠️  High noise court (${hitsPerMin.toFixed(0)} hits/min) — Anti-Interference`, 'warn'); deltaSlider.value = '6.0'; deltaVal.textContent = '6.0'; }
    else { log(`✅  Quiet environment (${hitsPerMin.toFixed(0)} hits/min)`); deltaSlider.value = '2.0'; deltaVal.textContent = '2.0'; }
    energyVal.textContent = parseFloat(energySlider.value).toFixed(3);
    onParamsChange();
    log(`🎾 ${detectedPeaks.length} hits → ${detectedRallies.length} rallies.`);
    analysisDone = true;
    startBtn.textContent = '✂️  Extract All Clips'; startBtn.disabled = false;
  } catch (err: any) {
    log('ERROR: ' + (err.message || String(err)), 'err'); console.error(err); startBtn.disabled = false;
    startBtn.textContent = '🔄 Retry Analysis';
  }
}

// Phase 2: Extract clips
startBtn.addEventListener('click', async () => {
  if (analysisDone && !extractionDone) {
    // Extract clips
    if (detectedRallies.length === 0) { log('No rallies to extract.', 'warn'); return; }
    startBtn.disabled = true; log('Extracting clips...');
    try {
      const sorted = [...detectedRallies].sort((a, b) => (b.end - b.start) - (a.end - a.start));
      const top = sorted.slice(0, 10);
      for (let i = 0; i < top.length; i++) {
        const r = top[i];
        const clipName = `clip_${r.id}.mp4`;
        const dur = r.end - r.start;
        updateProgress(`Clipping Rally ${i+1}/${top.length}...`, 50 + (45 / top.length) * i);
        log(`✂️  Rally ${i+1}: ${r.hits} hits, ${dur.toFixed(1)}s...`);
        await ffmpeg.exec(['-y', '-ss', r.start.toString(), '-i', 'input.mp4', '-t', dur.toString(), '-c:v', 'copy', '-c:a', 'copy', clipName]);
        const clipData = await ffmpeg.readFile(clipName);
        const blob = new Blob([(clipData as Uint8Array).buffer], { type: 'video/mp4' });
        const url = URL.createObjectURL(blob);
        rallyClips.set(r.id, { blob, url });
        renderClipGrid();
        renderClipList();
      }
      extractionDone = true;
      updateProgress('All Done!', 100);
      log(`✅ ${rallyClips.size} clips ready!`);
      // Switch to video display mode
      rallyList.style.display = 'none';
      clipGridView.style.display = currentView === 'grid' ? 'block' : 'none';
      clipListView.style.display = currentView === 'list' ? 'flex' : 'none';
      startBtn.textContent = '🔄  Re-Analyze'; startBtn.disabled = false;
    } catch (err: any) {
      log('ERROR: ' + (err.message || String(err)), 'err'); console.error(err); startBtn.disabled = false;
    }
  } else {
    // Re-analyze
    analysisDone = false; extractionDone = false;
    rallyClips.forEach(({ url }) => URL.revokeObjectURL(url)); rallyClips.clear(); activeCardId = null;
    clipGridView.style.display = 'none'; clipListView.style.display = 'none';
    rallyList.style.display = 'block';
    clipCardsEl.innerHTML = ''; clipListEl.innerHTML = '';
    startBtn.textContent = '⏳ Analyzing...'; startBtn.disabled = true;
    runAnalysis();
  }
});

window.addEventListener('resize', () => { if (audioData) { setupCanvases(); drawWaveform(); renderTimeline(); } });
