# Phi-1.5 QLoRA Fine-Tuning on Alpaca (TRL)
This repo fine-tunes a lightweight LLM (microsoft/phi-1_5) on the Alpaca instruction dataset using QLoRA (LoRA + 4-bit quantization) with the TRL `SFTTrainer`.

## Features
- Dataset: tatsu-lab/alpaca via 🤗 Datasets
- Formatting: ### Instruction / ### Input / ### Response prompt template
- Model: AutoModelForCausalLM (Phi-1.5), 4-bit NF4 quantization (bitsandbytes)
- PEFT: LoRA (r=16, α=16, dropout=0.1) for parameter-efficient fine-tuning
- Trainer: TRL SFTTrainer with SFTConfig (packing, grad accumulation)
- Precision: bf16 (fallback fp16), context length 1024
- Output: checkpoints + final adapter in outputs/phi_1_5_alpaca_qlora/best_model

## Requirements
- Python 3.10+
- CUDA-capable GPU recommended
- Key libraries:
- torch, transformers, datasets
- trl, peft, bitsandbytes
Example requirements.txt
```bash
torch
transformers>=4.40.0
datasets
trl>=0.8.0
peft>=0.10.0
bitsandbytes
```
## How it Works (Overview)
1. Load data: load_dataset('tatsu-lab/alpaca')
2. Split: 95% train / 5% valid
3. Format each example into:
```shell
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}

```
4. Model: Load microsoft/phi-1_5 with 4-bit NF4 quantization
5. PEFT: Apply LoRA adapters
6. Train: TRL SFTTrainer with packing, grad accumulation, and set epochs/batch size from Pr
7. Save: Write final model + tokenizer to outputs/.../best_model

## Quickstart
```bahs
# 1) Create & activate env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run training
python main.py

```
## Config (edit in `__main__`)
```python
Pr.batch_size = 1
Pr.epochs = 1
Pr.bf16 = True            # set False and use Pr.fp16=True if needed
Pr.gradient_accumulation_steps = 16
Pr.context_length = 1024
Pr.learning_rate = 2e-4
Pr.model_name = 'microsoft/phi-1_5'
Pr.out_dir = 'outputs/phi_1_5_alpaca_qlora'

```

## Outputs
Outputs
- outputs/phi_1_5_alpaca_qlora/best_model/
 - LoRA-adapted model weights & tokenizer (ready for inference or further training)

## Notes
- Uses 4-bit quantization (NF4 + double quant) for memory efficiency.
- Sets tokenizer.pad_token = eos_token to avoid padding issues.
- If you see memory errors, reduce batch_size or increase gradient_accumulation_steps.

