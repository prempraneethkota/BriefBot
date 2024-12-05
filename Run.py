import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set device to CPU
device = torch.device('cpu')

# Load the saved model and tokenizer
saved_model_dir = "./summarization_model"
tokenizer = AutoTokenizer.from_pretrained(saved_model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(saved_model_dir)
model.to(device)

# Define a function to generate summaries


def generate_summary(text):
    inputs = tokenizer([text], max_length=512, truncation=True, return_tensors="pt").to(device)
    # Explicitly set generation parameters
    generation_kwargs = {
        "num_beams": 4,
        "max_length": 150,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "forced_bos_token_id": model.config.bos_token_id,
    }
    summary_ids = model.generate(
        inputs["input_ids"],
        **generation_kwargs
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


while True:
    input_article = input("Please enter an article to summarize (or type 'exit' to quit): ")
    if input_article.lower() == 'exit':
        break
    print("Generated Summary:")
print(generate_summary(input_article))
print("\n")