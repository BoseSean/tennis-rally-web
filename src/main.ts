import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';
import FFT from 'fft.js';

const ffmpeg = new FFmpeg();
let ffmpegLoaded = false;

// UI Elements
const logDiv = document.getElementById('log') as HTMLDivElement;
const uploadInput = document.getElementById('videoUpload') as HTMLInputElement;
const startBtn = document.getElementById('startBtn') as HTMLButtonElement;
const resultDiv = document.getElementById('result') as HTMLDivElement;
const dropzone = document.getElementById('dropzone') as HTMLDivElement;
const dropzoneText = document.getElementById('dropzone-text') as HTMLParagraphElement;
const progressContainer = document.getElementById('progressContainer') as HTMLDivElement;
const progressFill = document.getElementById('progressFill') as HTMLDivElement;
const statusText = document.getElementById('statusText') as HTMLDivElement;

let selectedFile: File | null = null;

// Drag and drop logic
dropzone.addEventListener('click', () => uploadInput.click());
uploadInput.addEventListener('change', (e) => handleFileSelect((e.target as HTMLInputElement).files?.[0]));

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
  }, false);
});

['dragenter', 'dragover'].forEach(ev => dropzone.classList.add('dragover'));
['dragleave', 'drop'].forEach(ev => dropzone.classList.remove('dragover'));

dropzone.addEventListener('drop', (e) => handleFileSelect(e.dataTransfer?.files?.[0]));

function handleFileSelect(file?: File) {
  if (file && file.type.startsWith('video/')) {
    selectedFile = file;
    dropzoneText.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
    dropzone.style.borderColor = '#10b981';
    dropzone.style.backgroundColor = '#ecfdf5';
    startBtn.disabled = false;
  } else {
    alert('Please select a valid video file.');
  }
}

function log(msg: string) {
  const p = document.createElement('p');
  p.textContent = msg;
  logDiv.appendChild(p);
  logDiv.scrollTop = logDiv.scrollHeight;
  console.log(msg);
}

function updateProgress(step: string, percent: number) {
  progressContainer.style.display = 'block';
  statusText.textContent = step;
  progressFill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
}

async function loadFFmpeg() {
  if (ffmpegLoaded) return;
  log('Loading FFmpeg (this takes a moment)...');
  
  ffmpeg.on('log', ({ message }) => {
    console.log(message);
    if (message.includes('frame=') || message.includes('size=')) {
        // Just log the ffmpeg progress briefly
    } else {
        log(message);
    }
  });
  
  ffmpeg.on('progress', ({ progress }) => {
    // Only update progress visually if we are in an ffmpeg step
    if (progress > 0 && progress <= 1) {
      progressFill.style.width = `${progress * 100}%`;
    }
  });

  const baseURL = new URL(import.meta.env.BASE_URL, window.location.origin).href;
  log('Fetching local FFmpeg core...');
  await ffmpeg.load({
    coreURL: `${baseURL}ffmpeg-core.js`,
    wasmURL: `${baseURL}ffmpeg-core.wasm`,
  });
  ffmpegLoaded = true;
  log('FFmpeg loaded successfully!');
}

function computeOnsetStrength(audioData: Float32Array, sampleRate: number): Float32Array {
  const hopLength = 512;
  const nFft = 512;
  const numFrames = Math.floor((audioData.length - nFft) / hopLength) + 1;
  const diffs = new Float32Array(numFrames);
  
  const f = new FFT(nFft);
  const out = f.createComplexArray();
  const realInput = new Float32Array(nFft);
  
  const window = new Float32Array(nFft);
  for (let i = 0; i < nFft; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (nFft - 1)));
  }

  let prevMags = new Float32Array(nFft / 2 + 1);
  
  for (let i = 0; i < numFrames; i++) {
    const start = i * hopLength;
    for (let j = 0; j < nFft; j++) {
      realInput[j] = audioData[start + j] * window[j];
    }
    
    f.realTransform(out, Array.from(realInput)); 
    
    let currentFlux = 0;
    for (let k = 0; k <= nFft / 2; k++) {
      const re = out[2 * k];
      const im = out[2 * k + 1];
      const mag = Math.sqrt(re * re + im * im);
      
      const diff = mag - prevMags[k];
      if (diff > 0) currentFlux += diff;
      
      prevMags[k] = mag;
    }
    diffs[i] = currentFlux;
  }
  return diffs;
}

function peakPick(x: Float32Array, preMax: number, postMax: number, preAvg: number, postAvg: number, delta: number, wait: number): number[] {
  const peaks: number[] = [];
  for (let i = preMax; i < x.length - postMax; i++) {
    let isMax = true;
    for (let j = i - preMax; j <= i + postMax; j++) {
      if (x[j] > x[i]) {
        isMax = false;
        break;
      }
    }
    if (isMax) {
      let sum = 0;
      let count = 0;
      for (let j = Math.max(0, i - preAvg); j <= Math.min(x.length - 1, i + postAvg); j++) {
        sum += x[j];
        count++;
      }
      const avg = sum / count;
      if (x[i] >= avg + delta) {
        if (peaks.length === 0 || i - peaks[peaks.length - 1] >= wait) {
          peaks.push(i);
        }
      }
    }
  }
  return peaks;
}

function detectRallies(peaks: number[], sr: number, hopLength: number): {start: number, end: number, hits: number}[] {
  const times = peaks.map(p => p * hopLength / sr);
  if (times.length === 0) return [];
  
  const rallies: {start: number, end: number, hits: number}[] = [];
  let currentRally = [times[0]];
  
  for (let i = 1; i < times.length; i++) {
    if (times[i] - currentRally[currentRally.length - 1] <= 4.5) {
      currentRally.push(times[i]);
    } else {
      if (currentRally.length >= 3) {
        rallies.push({
          start: Math.max(0, currentRally[0] - 2),
          end: currentRally[currentRally.length - 1] + 4,
          hits: currentRally.length
        });
      }
      currentRally = [times[i]];
    }
  }
  if (currentRally.length >= 3) {
    rallies.push({
      start: Math.max(0, currentRally[0] - 2),
      end: currentRally[currentRally.length - 1] + 4,
      hits: currentRally.length
    });
  }
  return rallies;
}

startBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  
  startBtn.disabled = true;
  resultDiv.innerHTML = '';
  logDiv.innerHTML = '';
  logDiv.style.display = 'block';
  
  try {
    updateProgress('Loading FFmpeg Environment...', 5);
    await loadFFmpeg();
    
    updateProgress('Loading Video File...', 15);
    log('Writing video file to virtual memory...');
    await ffmpeg.writeFile('input.mp4', await fetchFile(selectedFile));
    
    updateProgress('Extracting 16kHz Audio...', 30);
    log('Extracting audio track for acoustic analysis (this is very fast)...');
    await ffmpeg.exec(['-y', '-i', 'input.mp4', '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav']);
    
    updateProgress('Analyzing Acoustic Signature...', 50);
    log('Reading extracted audio data...');
    const audioData = await ffmpeg.readFile('temp.wav');
    
    const audioCtx = new window.AudioContext({ sampleRate: 16000 });
    const audioBuffer = await audioCtx.decodeAudioData((audioData as Uint8Array).buffer.slice(0));
    const channelData = audioBuffer.getChannelData(0); 
    
    log('Running Spectral Flux Onset Detection...');
    const onsets = computeOnsetStrength(channelData, 16000);
    
    // Adaptive thresholding: Calculate hit density on highly sensitive pass to determine noise profile
    const sensitivePeaks = peakPick(onsets, 3, 3, 5, 5, 1.0, 10);
    const durationMins = channelData.length / 16000 / 60;
    const hitsPerMin = sensitivePeaks.length / durationMins;
    
    let delta = 2.0;
    log(`Environment scan: ${hitsPerMin.toFixed(1)} raw hits/min.`);
    if (hitsPerMin > 60) {
      log(`🎾 NOISY COURT detected. Using Anti-Interference Mode (Delta=5.0).`);
      delta = 5.0;
    } else {
      log(`🎾 QUIET COURT detected. Using High-Sensitivity Mode (Delta=2.0).`);
    }
    
    updateProgress('Clipping Rallies...', 60);
    log(`Extracting valid rallies...`);
    const peaks = peakPick(onsets, 3, 3, 5, 5, delta, 10);
    const rallies = detectRallies(peaks, 16000, 512);
    
    log(`✅ Found ${rallies.length} valid rallies!`);
    
    if (rallies.length === 0) {
      updateProgress('No Rallies Found', 100);
      log('No rallies found. Try another video.');
      startBtn.disabled = false;
      return;
    }
    
    rallies.sort((a, b) => (b.end - b.start) - (a.end - a.start));
    const topRallies = rallies.slice(0, Math.min(3, rallies.length));
    
    let concatList = '';
    for (let i = 0; i < topRallies.length; i++) {
      const r = topRallies[i];
      const clipName = `clip_${i}.mp4`;
      const dur = r.end - r.start;
      updateProgress(`Extracting Highlight ${i + 1} of ${topRallies.length}...`, 60 + (30 / topRallies.length) * i);
      log(`Clipping Highlight ${i + 1} (${r.hits} hits, ${dur.toFixed(1)}s)...`);
      
      await ffmpeg.exec([
        '-y', '-ss', r.start.toString(), '-i', 'input.mp4', '-t', dur.toString(), 
        '-c:v', 'copy', '-c:a', 'copy', clipName
      ]);
      concatList += `file '${clipName}'\n`;
    }
    
    updateProgress('Stitching Final Video...', 95);
    await ffmpeg.writeFile('concat.txt', concatList);
    log('Stitching highlights together...');
    await ffmpeg.exec(['-y', '-f', 'concat', '-safe', '0', '-i', 'concat.txt', '-c', 'copy', 'highlights.mp4']);
    
    updateProgress('Ready!', 100);
    log('Done! Preparing download link...');
    const finalData = await ffmpeg.readFile('highlights.mp4');
    const blob = new Blob([(finalData as Uint8Array).buffer], { type: 'video/mp4' });
    const url = URL.createObjectURL(blob);
    
    resultDiv.innerHTML = `
      <h3>🎉 Success! AI Found ${rallies.length} Rallies</h3>
      <p style="color:#6b7280; margin-top:-10px;">Top ${topRallies.length} longest rallies compiled below:</p>
      <video src="${url}" controls preload="metadata"></video>
      <br/>
      <a href="${url}" class="download-btn" download="tennis_highlights.mp4">Download Highlights</a>
    `;
    
  } catch (err: any) {
    updateProgress('Error Occurred', 0);
    log('ERROR: ' + (err.message || String(err))); console.error(err);
  } finally {
    startBtn.disabled = false;
  }
});