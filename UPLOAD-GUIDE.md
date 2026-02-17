# Upload Qwen3-4B Weights from Nibi to Hugging Face Hub

Follow these steps from the Nibi cluster to upload the fine-tuned Qwen3-4B model (98% accuracy) so the Hugging Face Space can use it with free GPU.

## 1. SSH into Nibi

```bash
ssh nibi.alliancecan.ca
```

## 2. Install huggingface-hub (if needed)

```bash
pip install --user huggingface-hub
```

## 3. Log in to Hugging Face

Go to https://huggingface.co/settings/tokens and create a token with **write** access, then:

```bash
huggingface-cli login
# Paste your token when prompted
```

## 4. Upload the model

```bash
huggingface-cli upload jburnford/dyslexic-writer-qwen3-4b \
  ~/projects/def-jic823/dyslexic-writer/training/outputs_qwen3/Qwen3-4B/ \
  --repo-type model
```

This uploads the full Transformers weights (~7.5 GB). It may take a few minutes depending on network speed.

## 5. Verify

Visit https://huggingface.co/jburnford/dyslexic-writer-qwen3-4b and confirm you see files like:
- `config.json`
- `model.safetensors` (or `pytorch_model.bin`)
- `tokenizer.json` / `tokenizer_config.json`
- `generation_config.json`

## What happens next

Once the model is on HF Hub, the Gradio Space in `space/` will be able to load it with ZeroGPU (free H200 GPU) for fast inference at full quality.
