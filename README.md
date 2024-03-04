# Indonesian_AI_Chatbot_Customer_Support

![Gajah_7-B](https://github.com/audichandra/Indonesian_AI_Chatbot_Customer_Support/blob/main/img/gajah%207b.jpg)

## Table of Contents
- [Description](#description)
- [File Structure](#file-structure)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Description
This project develops an Indonesian AI chatbot tailored for customer service applications. The chatbot is trained with Qlora by [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) from [Merak 7-B](https://huggingface.co/Ichsan2895/Merak-7B-v4) as the base model to understand and respond to customer queries effectively in Indonesian. The dataset needed to train the base model is taken from [Bitext](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) customer support LLM Chatbot training dataset which will be translated into Bahasa Indonesia by using [Opus](https://huggingface.co/Helsinki-NLP/opus-mt-en-id). This repository contains resources for data translation, model training, and a training model loading API with a Gradio frontend for easy interaction and testing for demonstration purposes of our Indonesian AI Chatbot Customer Support: [Gajah 7-B](https://huggingface.co/audichandra/Gajah-7B).

The 3 process: 
1. Training data translation
2. Base model training
3. API and demo frontend

## File Structure
- `img/`: Contains the image file for the example that are used in readme.md
- `requirements.txt`: List of all necessary Python libraries.
- `data_translation/`: Scripts and resources for translating and preparing datasets.
- `model_training/`: Code and resources for training the AI model.
- `API_frontend/`: API for loading the trained model, integrated with a Gradio frontend for demonstration and testing purposes.

## Getting Started

### Prerequisites
- Python 3.10
- Pytorch => 2.0 
- Jupyter Notebook (optional)

### Installation
To set up the environment:
*notes: this project is being run on runpod cloud gpu* 

```bash
git clone https://github.com/audichandra/Indonesian_AI_Chatbot_Customer_Support.git
cd Indonesian_AI_Chatbot_Customer_Support
pip install -r requirements.txt
```
1. Data translation: For this part, you can just use the **Translation Bitext.ipynb** (optional if you have your own datasets in your desired language) 
2. Model training: as mentioned, Axolotl will be used with Qlora method to train the base model into Gajah-7B with the command below (The **qlora stuff.ipynb** and **qlora API s.ipynb** are optional, it only intended to test the model capabilities in answering queries)
```bash
git clone -b main --depth 1 https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install packaging
pip install -e '.[flash-attn,deepspeed]'
pip install -U git+https://github.com/huggingface/peft.git 
pip install peft==0.6.0
pip install gradio -U
export TRANSFORMERS_CACHE=/workspace
```
*if there are some dependencies problem especially xformers, you can replace the setup.py in axolotl folder with the ones from **model_training** folder and run this command*
```bash
pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118
pip install -e '.[flash-attn,deepspeed]'
pip install -U git+https://github.com/huggingface/peft.git 
pip install peft==0.6.0
pip install gradio -U
export TRANSFORMERS_CACHE=/workspace
```
Replace the scripts of each followings with the ones from **model_training** folder:
- both qlora.yml and config.yml in the examples/mistral/ folder path
- zero2.json in deepspeed folder path (this is optional and depends on whether you will use deepspeed zero 2 or 3, TLDR: zero 2 speed up the training process while zero 3 is taking longer due to more detailed training)
- *some of the files path directories might need to be modified according to your current directories*
```bash
accelerate launch scripts/finetune.py examples/mistral/qlora.yml --deepspeed deepspeed/zero2.json
```
**This training process takes around 3 hours with 70% of training data and zero 2 with 8x a40 cloud gpu**

3. API and frontend: run both API script and Gradio frontend script on separate terminal process
```bash
cd..
cd API_frontend
uvicorn fast_s:app --host 0.0.0.0 --port 8000
```
```bash
python gradio_app.py
```
**This API and frontend process are loaded on 1x RTX a5000 with 24gb VRAM and 29gb RAM (you can use the gpu with less VRAM and RAM with BitsandBytes 4-bit Quantization but it may compromise the result)** 

## Results

![Screenshot of the frontend](https://github.com/audichandra/Indonesian_AI_Chatbot_Customer_Support/blob/main/img/gradioex5.png)

Some of the asked Indonesian questions are also answered in the **qlora API s.ipynb** in model_training folder 

## Acknowledgements
- **Authors**: Audi Chandra. Indonesian AI Chatbot Customer Service: Gajah-7B can be accessed on [Hugging Face](https://huggingface.co/audichandra/Gajah-7B). 
- **License**: [MIT License](https://github.com/audichandra/Selenium_Webscraping_Kalibrr/blob/main/LICENSE).
- **Base Model**: Special thanks to Muhammad Ichsan for developing "Merak-7B: The LLM for Bahasa Indonesia," a foundational model for our project, published on Hugging Face in 2023. You can explore this model further [here](https://huggingface.co/Ichsan2895/Merak-7B-v4).
- **Training Dataset**: Our model training leveraged the "Bitext Customer Support LLM Chatbot Training Dataset," available on Hugging Face. We appreciate the dataset compilers for their valuable contribution. Access the original dataset [here](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).
- **Machine Learning Translation (English to Indonesian)**: We utilized the machine translation model by Opus, specifically "opus-mt-en-id" for translating English to Indonesian. This significantly aided our data preparation process. Kudos to the team at Helsinki NLP for this resource, available on [Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-en-id).
- **Axolotl**: Our model training process was streamlined by Axolotl, an innovative tool by the OpenAccess AI Collective. We are grateful for their groundbreaking work in the field of machine learning. Learn more about Axolotl [here](https://github.com/OpenAccess-AI-Collective/axolotl).
