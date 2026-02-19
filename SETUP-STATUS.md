# Dyslexic Writer - Setup Status

**Date:** 2026-02-14
**System:** Windows 11, Intel Arc B580 (12 GB VRAM), 16 GB RAM

---

## ✅ Completed

- [x] Git installed and configured
  - Username: jburnford
  - Email: cljim22@gmail.com
- [x] GitHub extensions installed in VSCode
  - GitHub Pull Requests and Issues (v0.128.0)
  - GitLens (v17.10.0)
  - Git Graph (v1.30.0)
- [x] Repository cloned: `C:\Users\cljim\dyslexic-writer`
- [x] Latest updates pulled from main branch
- [x] Python 3.14.3 installed (working, but terminal needs restart to pick up PATH)
- [x] Modelfile created for Ollama (`C:\Users\cljim\dyslexic-writer\Modelfile`)
- [x] **Ollama** installed
- [x] **Intel Arc GPU Drivers** installed
- [x] **Node.js** installed
- [x] **huggingface-hub** installed (`pip install huggingface-hub`)
- [x] **Model downloaded** - v3-Qwen2.5-7B-q4_k_m.gguf (4.4 GB)
- [x] **Ollama model created** - `dyslexic-writer`
- [x] **Model tested** - Successfully fixes spelling errors ✓

---

## ⏳ In Progress

- [ ] **None** - Ready to install dependencies and run the app!

---

## ❌ Still Needed

### None - Model is ready! Next steps below:

---

## 📦 Install Dependencies

### Python Backend (Flask + TTS)
```bash
cd C:\Users\cljim\dyslexic-writer\app
pip install -r requirements.txt
```

Dependencies:
- Flask (web server)
- piper-tts (text-to-speech)
- soundfile/sounddevice (audio)
- requests, numpy, flask-cors

### Node.js Frontend (React + Vite)
```bash
cd C:\Users\cljim\dyslexic-writer
npm install
```

---

## 🚀 Running the App

### Start Backend Server
```bash
cd C:\Users\cljim\dyslexic-writer\app
python server.py --backend ollama --model dyslexic-writer
```

Server will run at http://127.0.0.1:5000

### Start Frontend (separate terminal)
```bash
cd C:\Users\cljim\dyslexic-writer
npm run dev
```

Frontend will run at http://localhost:5173 (or similar)

---

## 🔧 System Info

**GPU:** Intel Arc B580 Graphics (12 GB VRAM)
- Supports the v3-Qwen2.5-7B model (4.4 GB VRAM required)
- Ollama will use Intel GPU backend automatically
- Expected VRAM usage: ~5-6 GB during inference

**RAM:** 16 GB (meets minimum requirement)

**Model Choice:** v3-Qwen2.5-7B-q4_k_m (4.4 GB)
- Best quality option
- 69% exact match, 85% error correction rate
- Fits comfortably in 12 GB VRAM

---

## 🐛 Troubleshooting

### Python not found after installation
**Fix:** Restart VSCode terminal (trash can icon → new terminal) or restart VSCode completely to pick up updated PATH

### "model not found" when running Ollama
**Fix:** Make sure you ran `ollama create` from the directory containing the GGUF file

### Out of memory
**Fix:** Use the smaller v2-SmolLM2-1.7B model (1 GB) - update Modelfile `FROM` line and download different GGUF

---

## 📝 Next Steps

1. ✅ **Restart VSCode terminal** to pick up Python PATH
2. ⏳ Wait for Ollama download to complete
3. ⏳ Wait for Intel Arc driver download to complete
4. ⬜ Install Node.js
5. ⬜ Install huggingface-cli and download model
6. ⬜ Create Ollama model
7. ⬜ Install Python dependencies
8. ⬜ Install Node.js dependencies
9. ⬜ Test the app!

---

## 📚 Key Files

- `SETUP-LOCAL.md` - Official local setup instructions
- `README.md` - Project overview and architecture
- `app/server.py` - Flask backend server
- `app/requirements.txt` - Python dependencies
- `package.json` - Node.js dependencies
- `Modelfile` - Ollama model configuration (already created)

---

## 🔗 Useful Links

- Ollama: https://ollama.com/download/windows
- Python: https://www.python.org/downloads/
- Node.js: https://nodejs.org/
- Intel Arc Drivers: https://www.intel.com/content/www/us/en/download/785597/
- Model Repository: https://huggingface.co/jburnford/dyslexic-writer-spelling

---

**Resume from here when you restart!**
