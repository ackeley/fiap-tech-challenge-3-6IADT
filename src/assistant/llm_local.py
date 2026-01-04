from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_DIR = BASE_DIR / "lora_adapter"

def load_llm_cpu():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SEM device_map para evitar conflito com pipeline
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
    model.eval()

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return gen
