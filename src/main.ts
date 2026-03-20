import './style.css'
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';
import FFT from 'fft.js';

const ffmpeg = new FFmpeg();
let ffmpegLoaded = false;

const logDiv = document.getElementById('log') as HTMLDivElement;
const uploadInput = document.getElementById('videoUpload') as HTMLInputElement;
const startBtn = document.getElementById('startBtn') as HTMLButtonElement;
const resultDiv = document.getElementById('result') as HTMLDivElement;

function log(msg: string) {
  const p = document.createElement('p');
  p.textContent = msg;
  logDiv.appendChild(p);
  logDiv.scrollTop = logDiv.scrollHeight;
  console.log(msg);
}

async function loadFFmpeg() {
  if (ffmpegLoaded) return;
  log('Loading FFmpeg (this takes a moment)...');
  
  ffmpeg.on('log', ({ message }) => {
    console.log(message);
  });
  
  ffmpeg.on('progress', ({ progress, time }) => {
    // just to keep console alive, maybe update a bar
  });

  await ffmpeg.load({
    coreURL: 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.js',
    wasmURL: 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.wasm'
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
  
  // Hanning window
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
    
    // We must pass normal array or Float64Array to fft.js, but Float32Array usually works if typed properly or converted
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
  const file = uploadInput.files?.[0];
  if (!file) {
    alert('Please select a video file first.');
    return;
  }
  
  startBtn.disabled = true;
  resultDiv.innerHTML = '';
  
  try {
    await loadFFmpeg();
    log('Writing video file to memory...');
    await ffmpeg.writeFile('input.mp4', await fetchFile(file));
    
    log('Extracting 16kHz mono audio (this is fast)...');
    await ffmpeg.exec(['-y', '-i', 'input.mp4', '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'temp.wav']);
    
    log('Reading audio data for analysis...');
    const audioData = await ffmpeg.readFile('temp.wav');
    
    // Decode Audio using Web Audio API
    const audioCtx = new window.AudioContext({ sampleRate: 16000 });
    const audioBuffer = await audioCtx.decodeAudioData((audioData as Uint8Array).buffer.slice(0));
    const channelData = audioBuffer.getChannelData(0); // Float32Array
    
    log('Analyzing acoustic signature (Spectral Flux Onset Detection)...');
    const onsets = computeOnsetStrength(channelData, 16000);
    
    // Adaptive thresholding: Calculate hit density on highly sensitive pass to determine noise profile
    const sensitivePeaks = peakPick(onsets, 3, 3, 5, 5, 1.0, 10);
    const durationMins = channelData.length / 16000 / 60;
    const hitsPerMin = sensitivePeaks.length / durationMins;
    
    let delta = 2.0; // Quiet default
    log(`Environment scan: ${hitsPerMin.toFixed(1)} raw hits/min.`);
    if (hitsPerMin > 60) {
      log('🎾 High density detected. Switching to Anti-Noise Mode (Interference filtering).');
      delta = 5.0; // Loud default
    } else {
      log('🎾 Quiet court detected. Using High-Sensitivity Mode.');
    }
    
    log(`Extracting Rallies (Delta=${delta.toFixed(1)})...`);
    const peaks = peakPick(onsets, 3, 3, 5, 5, delta, 10);
    const rallies = detectRallies(peaks, 16000, 512);
    
    log(`✅ Found ${rallies.length} valid rallies! Generating highlights...`);
    
    if (rallies.length === 0) {
      log('No rallies found. Try another video.');
      startBtn.disabled = false;
      return;
    }
    
    // Pick Top 3 longest rallies to avoid OOM in browser memory during concat
    rallies.sort((a, b) => (b.end - b.start) - (a.end - a.start));
    const topRallies = rallies.slice(0, Math.min(3, rallies.length));
    
    let concatList = '';
    for (let i = 0; i < topRallies.length; i++) {
      const r = topRallies[i];
      const clipName = `clip_${i}.mp4`;
      const dur = r.end - r.start;
      log(`Clipping Highlight ${i + 1} (${r.hits} hits, ${dur.toFixed(1)}s)...`);
      
      // Cut with re-encoding to smaller size for speed and memory in browser
      await ffmpeg.exec([
        '-y', '-ss', r.start.toString(), '-i', 'input.mp4', '-t', dur.toString(), 
        '-vf', 'scale=854:480', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '30', 
        '-c:a', 'aac', '-b:a', '128k', clipName
      ]);
      concatList += `file '${clipName}'\n`;
    }
    
    await ffmpeg.writeFile('concat.txt', concatList);
    log('Stitching highlights together...');
    await ffmpeg.exec(['-y', '-f', 'concat', '-i', 'concat.txt', '-c', 'copy', 'highlights.mp4']);
    
    log('Done! Preparing download link...');
    const finalData = await ffmpeg.readFile('highlights.mp4');
    const blob = new Blob([(finalData as Uint8Array).buffer], { type: 'video/mp4' });
    const url = URL.createObjectURL(blob);
    
    resultDiv.innerHTML = `
      <h3>🎉 Success!</h3>
      <p>Total Rallies Detected: ${rallies.length}</p>
      <video src="${url}" controls width="100%"></video>
      <br/><br/>
      <a href="${url}" download="tennis_highlights.mp4" style="padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">Download Highlights</a>
    `;
    
  } catch (err: any) {
    log('ERROR: ' + err.message);
  } finally {
    startBtn.disabled = false;
  }
});

