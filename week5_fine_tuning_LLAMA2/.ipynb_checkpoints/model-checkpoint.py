'''A function that fetches a large language model and configures it for k-bit quantization.
The function returns the peft configured model.'''


import transformers as t # provides pre-trained models for various tasks.
import peft # parameter efficient fine tuning: custom library related to LORA (Layer-wise Optimal Rank Adaptation). Enables efficient adaption of pre-trained
            # language models (PLMs) to various downstream applications without fine-tuning all the model's parameters. 
import torch
import os

def get_model():
  #pick the model to be fine tuned
  NAME = "NousResearch/Llama-2-7b-hf" 
  #check if the code is running in a distributed data parallel (DDP) setting
  is_ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
  #get the rank of different available devices to organize how the model is replicated across multiple devices. 
  device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if is_ddp else None
  #load the chosen pre-trained model. load_in_8bit is for model weights. float16 is for model parameters
  m = t.AutoModelForCausalLM.from_pretrained(NAME, load_in_8bit=True, torch_dtype=torch.float16, device_map=device_map)
  #prepare the model for K-bit quantization to reduce memory requirements and to make computational operations more efficeint
  m = peft.prepare_model_for_kbit_training(m)
  #define LoRA-specific parameters
  config = peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.01, bias="none", task_type="CAUSAL_LM")
  #wrap the base model to get a trainable PeftModel
  peft_model = peft.get_peft_model(m, config)
  return peft_model
