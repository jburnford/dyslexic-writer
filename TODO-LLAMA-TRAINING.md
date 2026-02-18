# Llama-3.2-3B Training - Continuation Notes

## Status
- **Llama gate access**: Approved on HuggingFace, but the **write token** (`hf_bjGd...`) may not have gate access
- The original read token (`hf_Zcmx...`) had access confirmed, but the write token keeps getting 403
- All other models are trained and deployed

## The Problem
Gate access on HuggingFace is tied to the token used to accept the license. The write token was created after accepting, so it may not inherit the access. Fix: either re-accept the license while logged in with the write token's session, or use the read token for downloading.

## Fix Option 1: Use read token for training (recommended)
On Nibi, save the original read token before submitting:
```bash
echo '<YOUR_READ_TOKEN>' > ~/.cache/huggingface/token
```
Then submit:
```bash
cd ~/dyslexic-writer/training
sbatch train_v3_large.slurm
```
After training, switch back to write token for uploads:
```bash
echo '<YOUR_WRITE_TOKEN>' > ~/.cache/huggingface/token
```

## Fix Option 2: Re-accept license
1. Go to https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
2. Make sure you're logged in as `jburnford`
3. Click "Agree and access repository" again
4. Resubmit the job (write token should work after)

## Fix Option 3: Use ungated mirror
Change the model in `train_v3_large.slurm`:
```
--model unsloth/Llama-3.2-3B-Instruct
```
Same weights, no gate.

## After Llama Trains Successfully

### 1. Run Bob eval
Update `eval_bob_story.slurm` to add:
```python
("v3-Llama-3.2-3B", Path("/scratch/jic823/dyslexic-writer/outputs_v3/Llama-3.2-3B-Instruct")),
```
Then: `sbatch eval_bob_story.slurm`

### 2. Convert to GGUF
Update `convert_to_gguf.slurm` to add:
```
"v3-Llama-3.2-3B|/scratch/jic823/dyslexic-writer/outputs_v3/Llama-3.2-3B-Instruct"
```
Then: `sbatch convert_to_gguf.slurm`

Expected GGUF size: ~2GB (Q4_K_M) — fits easily on 12GB VRAM.

### 3. Upload to HuggingFace
```bash
module load python/3.11
echo '<YOUR_WRITE_TOKEN>' > ~/.cache/huggingface/token
HF_TOKEN=$(cat ~/.cache/huggingface/token) python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ['HF_TOKEN'])
api.upload_file(
    path_or_fileobj='/scratch/jic823/dyslexic-writer/gguf_models/v3-Llama-3.2-3B-q4_k_m.gguf',
    path_in_repo='v3-Llama-3.2-3B-q4_k_m.gguf',
    repo_id='jburnford/dyslexic-writer-spelling',
    token=os.environ['HF_TOKEN'],
)
print('Uploaded!')
"
```

### 4. Update SETUP-LOCAL.md
Add Llama to the model table and update instructions.

## Current Model Leaderboard (Bob Story Eval)

| Model | Size (GGUF Q4) | Exact Match | Errors Fixed | FP | HuggingFace |
|-------|---------------|-------------|--------------|-----|-------------|
| v3-Qwen2.5-7B | 4.4 GB | 69.2% | 22/26 (85%) | 0 | uploaded |
| v2-SmolLM2-1.7B | 1.0 GB | 61.5% | 21/26 (81%) | 1 | uploaded |
| v3-Llama-3.2-3B | ~2 GB | TBD | TBD | TBD | pending |

## Files on Nibi
- Training script: `~/dyslexic-writer/training/train_v3_large.slurm`
- Training code: `~/dyslexic-writer/training/finetune_v2.py`
- Training data: `/scratch/jic823/dyslexic-writer/synthetic_train_clean_instruction.jsonl`
- Eval data: `/scratch/jic823/dyslexic-writer/synthetic_eval_clean_instruction.jsonl`
- Model outputs: `/scratch/jic823/dyslexic-writer/outputs_v3/`
- GGUF models: `/scratch/jic823/dyslexic-writer/gguf_models/`
- Logs: `/scratch/jic823/dyslexic-writer/logs/`
