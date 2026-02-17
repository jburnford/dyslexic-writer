# Deploying Dyslexic Writer

How to share and demo the app. Since Dyslexic Writer runs a 7B-parameter LLM for spelling correction, it needs a GPU at runtime.

## App Stack

| Component | Tech | GPU? |
|-----------|------|------|
| Frontend | React + Vite | No |
| Backend | Flask (Python) + Piper TTS | No |
| Spelling model | Qwen2.5-7B via Ollama | Yes (~5-6 GB VRAM) |

---

## Option 1: Local Demo (Simplest)

Bring the laptop. Everything already runs on the Intel Arc B580.

**Share on the same Wi-Fi:**

```bash
# Start backend
cd app && python server.py --backend ollama --model dyslexic-writer

# Start frontend (exposed to LAN)
npm run dev -- --host
```

Then share `http://<your-local-ip>:5173` with visitors. Works without internet.

---

## Option 2: ngrok Tunnel (Public URL, Local GPU)

Expose the running local app to the internet so anyone can try it from a shareable link.

### Setup

```bash
# Install ngrok
# Download from https://ngrok.com/download or:
choco install ngrok   # Windows
brew install ngrok     # Mac

# Sign up for free account and add auth token
ngrok config add-authtoken <your-token>
```

### Run

```bash
# Terminal 1: start backend
cd app && python server.py --backend ollama --model dyslexic-writer

# Terminal 2: start frontend
npm run dev -- --host

# Terminal 3: tunnel the frontend
ngrok http 5173
```

ngrok gives you a URL like `https://abc123.ngrok-free.app` — put this on a poster QR code.

**Pros:** Free, easy, your GPU does all the work, nothing to deploy.
**Cons:** Requires your laptop to stay on and connected to the internet.

---

## Option 3: Hugging Face Spaces (Free GPU, Permanent Link)

Deploy as a Docker Space on Hugging Face with free ZeroGPU access.

### What's Needed

1. Create a Space at `huggingface.co/new-space` (Docker SDK)
2. Write a `Dockerfile` that installs Ollama + the model + Flask + the React build
3. Replace Ollama with `llama-cpp-python` (simpler in a container)
4. Shareable URL: `huggingface.co/spaces/jburnford/dyslexic-writer`

### Pros/Cons

| Pros | Cons |
|------|------|
| Free GPU tier (ZeroGPU) | Cold starts (~30s first load) |
| Permanent shareable URL | Need to rework model loading |
| Looks great for college apps | Free tier has usage limits |

---

## Option 4: Modal (Serverless GPU, Pay-per-Use)

Run the Flask backend + model as a serverless GPU function. Pay only when someone uses it.

### What's Needed

1. `pip install modal` and set up an account
2. Write a Modal app that loads the GGUF model with `llama-cpp-python`
3. Deploy the React frontend to Vercel or Netlify (free)
4. Point the frontend at the Modal API URL

### Pricing

- A10G GPU: ~$0.50/hr, billed per second
- A science fair demo day might cost $2-5 total

---

## Recommendation: Science Fair Day

**Use Option 1 + Option 2 together:**

1. Demo the app live on your laptop (reliable, works even if Wi-Fi drops)
2. Run ngrok so visitors can try it on their phones via a QR code
3. Your poster/display can include the ngrok URL and a QR code

**After the fair**, consider deploying to Hugging Face Spaces for a permanent demo link you can share with anyone.
