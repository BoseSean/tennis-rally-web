# 🎾 Tennis Rally Clipper

Auto-extract tennis rally clips from match videos using audio onset detection. Pure client-side — no server, no upload, fully private.

**Live:** [https://BoseSean.github.io/tennis-rally-web/](https://BoseSean.github.io/tennis-rally-web/)

---

## How It Works

1. **Upload** a tennis video (MP4, MOV, etc.)
2. **Analysis** runs automatically — audio onset detection finds each hit
3. **Tune** parameters in real-time on the timeline
4. **Extract** clips with one click

---

## Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Delta** | 0.5 – 20 | Hit detection sensitivity. Higher = more sensitive to quieter sounds |
| **Max Gap** | 1.5 – 15s | Maximum silence between hits before a rally ends |
| **Min Hits** | 2 – 20 | Minimum number of hits to qualify as a rally |
| **Energy Floor** | 0.001 – 0.01 | RMS energy threshold — ignores very quiet segments |

---

## Architecture

```
Video File
    ↓
FFmpeg WASM (browser) — extract PCM audio
    ↓
Web Audio API — decode to Float32Array
    ↓
Custom TS onset detection (librosa-style spectral flux)
    ↓
Peak picking → rally segments → timeline
    ↓
FFmpeg WASM — extract each rally clip (stream-copy, no re-encode)
    ↓
Blob URLs → inline video players
```

### Key techniques

- **Spectral flux onset detection** — detects energy increases across FFT bins frame-to-frame
- **Adaptive Delta** — auto-switches between quiet (Δ=2.0) and noisy (Δ=5.0) courts based on hit density
- **RMS energy filtering** — rejects segments where audio energy is too low
- **Chunked file loading** — 128MB streaming chunks avoid browser memory OOM for large (2GB+) files
- **Stream-copy extraction** — `-c copy` to avoid transcoding in the browser

---

## Development

```bash
git clone https://github.com/BoseSean/tennis-rally-web.git
cd tennis-rally-web
npm install
npm run dev       # dev server at localhost:5173
npm run build     # production build → dist/
npm run deploy    # build + push to gh-pages
```

### Tech Stack

- **Vite** + Vanilla TypeScript (no framework overhead)
- **@ffmpeg/core** + **@ffmpeg/ffmpeg** v0.12.x — browser-side video/audio processing
- **Web Audio API** — FFT and onset analysis
- **Canvas API** — waveform and timeline visualization
- **GitHub Pages** — zero-cost hosting

---

## Project Structure

```
tennis-rally-web/
├── index.html          # Single HTML entry point
├── src/
│   └── main.ts         # All logic: audio analysis, FFmpeg, UI rendering
├── public/
│   ├── ffmpeg-core.js  # FFmpeg core (ESM build)
│   └── ffmpeg-core.wasm
└── package.json
```

---

## Audio Analysis Algorithm

1. Decode video → PCM WAV (FFmpeg, 16kHz mono)
2. Compute STFT (2048-point FFT, 512 hop)
3. **Onset strength envelope** = sum of positive spectral differences across bins
4. **RMS energy envelope** = rolling root-mean-square of audio frames
5. **Peak picking** on onset envelope using adaptive Delta threshold
6. **Rally grouping** — merge peaks separated by ≤ Max Gap
7. **Filter** rallies with < Min Hits or RMS below Energy Floor

---

## Known Limits

- **Large files (2GB+):** Uses 128MB chunked streaming to avoid JS heap OOM. Very large files may still exceed FFmpeg WASM memory ceiling (~4GB single-threaded). Recommend pre-compressingExtremely with HandBrake/FFmpeg CLI if crashes persist.
- **Audio quality:** Built for tennis matches with clear hit sounds. Ambient crowd noise may affect detection accuracy.
- **Mobile:** Functional but not optimized — desktop Chrome recommended for best performance.

---

## License

MIT
