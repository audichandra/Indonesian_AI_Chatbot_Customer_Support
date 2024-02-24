# Indonesian_AI_Chatbot_Customer_Support

## Table of Contents
- [Description](#description)
- [File Structure](#file-structure)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Description
This project develops an Indonesian AI chatbot tailored for customer service applications. The chatbot is trained with Qlora by [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) from [Merak 7-B](https://huggingface.co/Ichsan2895/Merak-7B-v4) as the base model to understand and respond to customer queries effectively in Indonesian. The dataset needed to train the base model is taken from [Bitext](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) customer support LLM Chatbot training dataset which will be translated into Bahasa Indonesia by using [Opus](https://huggingface.co/Helsinki-NLP/opus-mt-en-id). This repository contains resources for data translation, model training and a training model loading API with a Gradio frontend for easy interaction and testing for demonstration purposes.

## File Structure
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

```bash
git clone https://github.com/your-username/your-repository.git

1. Clone this repository:  
   `git clone <repo-link>`
2. Navigate to the directory:  
   `cd <repo-name>`
3. Install the required packages:  
   `pip install -r requirements.txt`



## Usage
After you installed the required packages, you can navigate into `app.py` file manually and run it. Then, open your browser and navigate to http://127.0.0.1:5000 to see the visualized job listing data.
For a detailed explanation and code walkthrough, please refer to [Selenium Webscraping Kalibrr Notebook](https://github.com/audichandra/Selenium_Webscraping_Kalibrr/blob/main/Selenium%20web%20scraping%20Kalibrr.ipynb).

## Results
Below are some visual results obtained from the scraped data:

![Job Distribution by Location](https://github.com/audichandra/Selenium_Webscraping_Kalibrr/blob/main/img/dfg3.png)

This graph shows the distribution of job postings by Indonesia top 10 areas, indicating the cities with the highest demand 

![Job Posting Period Distribution](https://github.com/audichandra/Selenium_Webscraping_Kalibrr/blob/main/img/dfg1.png)

The above visualization gives insights into how long the companies will open their job postings  

![Job Posting Distribution based on Data Roles](https://github.com/audichandra/Selenium_Webscraping_Kalibrr/blob/main/img/dfg2.png)

## Acknowledgements
- **Authors**: Audi Chandra  
- **License**: [MIT License](https://github.com/audichandra/Selenium_Webscraping_Kalibrr/blob/main/LICENSE) 
- A nod to [**Kalibrr**](https://www.kalibrr.id/id-ID/job-board/te/data/1) for providing a platform filled with rich job posting data.
- Heartfelt gratitude to [**Algoritma Data Science School**](https://gitlab.com/algoritma4students/academy-python/capstone/web_scraping) for making available the base example of the project and providing a learning opportunity.
