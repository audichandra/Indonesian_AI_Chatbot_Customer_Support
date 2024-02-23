from datasets import load_dataset, Dataset, load_metric
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaTokenizer
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorWithPadding
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import csv
import re
import nltk
import pickle
import os
import time
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from sacrebleu.metrics import BLEU, CHRF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss, accuracy_score
from peft import PeftModel, PeftConfig
import sacrebleu
#import logging
import bitsandbytes as bnb
#from torch.nn import DataParallel
#logging.basicConfig(level=logging.INFO)


app = FastAPI()

# Load your model
model_chat = "audichandra/Gajah-7B"
tokenizer = LlamaTokenizer.from_pretrained("Ichsan2895/Merak-7B-v4")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = AutoModelForCausalLM.from_pretrained(model_chat, torch_dtype=torch.float16
                                              #, device_map="auto", pad_token_id=0 
                                              #, attn_implementation="flash_attention_2"  
                                              , cache_dir="/workspace").to(device)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    elapsed_time: float 

def format_response(response: str) -> str:
    # Add a newline before and after each list item
    formatted_response = re.sub(r'(\d+\.)', r'\n\1\n', response)

    # Add a newline before the start of the numbered list
    formatted_response = re.sub(r'(\:\s*)\n(\d+\.)', r'\1\n\n\2', formatted_response)

    # Remove unwanted newlines following list numbers
    formatted_response = re.sub(r'(\d+\.)\n\s+', r'\1 ', formatted_response)

    # Identify the last item of the list and add an extra paragraph break if needed
    formatted_response = re.sub(r'(\d+\.\s.*?\.)\s+(\w)', r'\1\n\n\2', formatted_response)

    # Clean up: Ensure there are not more than two consecutive newlines
    formatted_response = re.sub(r'\n{3,}', '\n\n', formatted_response)
    formatted_response = formatted_response.strip()

    return formatted_response

def adjust_list_spacing(formatted_response: str) -> str:
    # Correctly format each list item with a single line break
    # Replace multiple newlines with a single newline after list numbers
    adjusted_response = re.sub(r'(\d+\.)\s*\n+', r'\1 ', formatted_response)

    # Add a newline after each list item
    adjusted_response = re.sub(r'(\d+\.[^\n]*)(?=\n\d+|$)', r'\1\n', adjusted_response)

    # Special handling for the last list item: Check if there is more content after it
    last_item_match = re.search(r'(\d+\.\s.*?)(?=\n\d+|$)', adjusted_response, re.DOTALL)
    if last_item_match:
        last_item = last_item_match.group(1)
        # If the last item is followed by more content, add an extra paragraph break
        if re.search(r'\.\s', last_item.split('.')[1]):
            adjusted_response = re.sub(last_item + r'\n', last_item + '\n\n', adjusted_response, 1, re.DOTALL)

    return adjusted_response.strip() 

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.time()

    # Generate response using your model
    chat = [
      {"role": "system", "content": "Ada yang bisa saya bantu?"},
      {"role": "user", "content": request.question},
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)

    inputs = inputs.to(device)  # Ensure inputs are on the same device as the model

    with torch.no_grad():
        outputs = model1.generate(**inputs, max_new_tokens=512)
        
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Clear cache after model generates response
    #if torch.cuda.is_available():
        #torch.cuda.empty_cache()

    assistant_start = f'''{request.question} \n assistant\n '''
    response_start = response.find(assistant_start)
    response_formatted = format_response(response[response_start + len(assistant_start):].strip())
    final_response = adjust_list_spacing(response_formatted)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return ChatResponse(answer=final_response, elapsed_time=elapsed_time)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)