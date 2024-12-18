import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator)
import os
import psutil

# Function to monitor memory usage


def limit_memory_usage(max_usage_percent=85):
    process = psutil.Process(os.getpid())
    max_usage = (max_usage_percent / 100) * psutil.virtual_memory().total
    if process.memory_info().rss > max_usage:
        raise MemoryError("Memory usage exceeded limit.")

# Set device to CPU


device = torch.device('cpu')

# Load the dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset['train'].select(range(250))  # Using a smaller subset for demonstration
valid_data = dataset['validation'].select(range(150))

# Load pre-trained tokenizer and model
model_name = "facebook/bart-base"  # Using a smaller model for faster training
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move the model to CPU
model.to(device)

# Preprocess function for tokenizing data


def preprocess_function(examples):
    inputs = tokenizer(examples["article"], max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(examples["highlights"], max_length=128, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

# Apply preprocessing to datasets


train_dataset = train_data.map(preprocess_function, batched=True)
valid_dataset = valid_data.map(preprocess_function, batched=True)

# Set dataset format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
valid_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Custom data collator function to handle labels efficiently


def collate_fn(batch):
    batch = default_data_collator(batch)
    if "labels" in batch:
        batch["labels"] = torch.tensor(np.array(batch["labels"]), dtype=torch.int64)
    return batch

# Training arguments with increased logging verbosity


training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=3,  # Adjust batch size for CPU usage
    per_device_eval_batch_size=3,   # Adjust eval batch size for CPU usage
    weight_decay=0.01,
    save_total_limit=4,
    num_train_epochs=4,
    predict_with_generate=True,
    logging_dir='./logs',
    logging_steps=10,  # Log every 10 steps
    report_to="all",  # Report to all available logging integrations (e.g., console, TensorBoard)
    logging_first_step=True,  # Log the very first step as well
)

# Trainer with memory monitoring


class MemoryLimitedSeq2SeqTrainer(Seq2SeqTrainer):
    def training_step(self, *args, **kwargs):
        limit_memory_usage(max_usage_percent=85)
        return super().training_step(*args, **kwargs)


trainer = MemoryLimitedSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./summarization_model")
tokenizer.save_pretrained("./summarization_model")

# Generate summary for a new article


def generate_summary(text):
    inputs = tokenizer([text], max_length=512, truncation=True, return_tensors="pt").to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=150,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example usage
input_article = """
Lung cancer is a type of cancer that begins in the lungs. Your lungs are two spongy organs in your chest that take in oxygen when you inhale and release carbon dioxide when you exhale.
Lung cancer is the leading cause of cancer deaths worldwide. People who smoke have the greatest risk of lung cancer, though lung cancer can also occur in people who have never smoked.
The risk of lung cancer increases with the length of time and number of cigarettes you've smoked. If you quit smoking, even after smoking for many years, you can significantly reduce your chances of developing lung cancer.
"""
print("Generated Summary:")
print(generate_summary(input_article))