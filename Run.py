import subprocess
import sys


# Function to install nltk
def install_nltk():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])


# Check if nltk is installed and install if not
try:
    import nltk
except ImportError:
    install_nltk()
    import nltk

nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set device to CPU
device = torch.device('cpu')

# Load the saved model and tokenizer
saved_model_dir = "./summarization_model"
tokenizer = AutoTokenizer.from_pretrained(saved_model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(saved_model_dir)
model.to(device)


# Function to split text into sentences
def split_into_sentences(text):
    return sent_tokenize(text)


# Function to split sentences into chunks
def chunk_sentences(sentences, max_tokens=512):
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        if current_length + len(tokens) <= max_tokens:
            current_chunk.append(sentence)
            current_length += len(tokens)
        else:
            yield ' '.join(current_chunk)
            current_chunk = [sentence]
            current_length = len(tokens)
    if current_chunk:
        yield ' '.join(current_chunk)


# Define generation parameters
generation_kwargs = {
    "num_beams": 4,
    "max_length": 150,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
    "forced_bos_token_id": model.config.bos_token_id,
}


# Function to generate summaries
def generate_summary(text):
    sentences = split_into_sentences(text)
    summaries = []
    for chunk in chunk_sentences(sentences):
        inputs = tokenizer([chunk], max_length=512, truncation=True, return_tensors="pt").to(device)
        summary_ids = model.generate(inputs["input_ids"], **generation_kwargs)
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    # Combine all chunk summaries into a single text
    final_summary = " ".join(summaries)

    # Generate a summary of the combined chunk summaries
    inputs = tokenizer([final_summary], max_length=512, truncation=True, return_tensors="pt").to(device)
    summary_ids = model.generate(inputs["input_ids"], **generation_kwargs)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Interactive part
while True:
    input_article = input("Please enter an article to summarize (or type 'exit' to quit): ")
    if input_article.lower() == 'exit':
        print("Exiting the summarizer. Goodbye!")
        break
    summary = generate_summary(input_article)
    print("\nGenerated Summary:")
    print(summary)
    print("\n")
