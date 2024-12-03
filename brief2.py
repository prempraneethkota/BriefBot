from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertModel
import os
from torch.cuda.amp import autocast, GradScaler

# Load the dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset['train']
valid_data = dataset['validation']

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def preprocess_function(examples):
    inputs = tokenizer(
        examples["article"], max_length=512, truncation=True, padding="max_length"
    )
    targets = tokenizer(
        examples["highlights"], max_length=128, truncation=True, padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs


train_dataset = train_data.map(preprocess_function, batched=True)
valid_dataset = valid_data.map(preprocess_function, batched=True)

train_dataset = train_dataset.remove_columns(["article", "highlights"])
valid_dataset = valid_dataset.remove_columns(["article", "highlights"])
train_dataset.set_format("torch")
valid_dataset.set_format("torch")


class SummarizationModel(nn.Module):
    def __init__(self, vocab_size):
        super(SummarizationModel, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.decoder = nn.GRU(input_size=768, hidden_size=768, num_layers=2, batch_first=True)
        self.fc = nn.Linear(768, vocab_size)

    def forward(self, input_ids, attention_mask, labels=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Use the hidden states of the encoder as the initial hidden states for the decoder
        decoder_outputs, _ = self.decoder(encoder_hidden_states)

        if labels is not None:
            # Ensure decoder outputs match the length of labels
            decoder_outputs = decoder_outputs[:, :labels.size(1), :]
            logits = self.fc(decoder_outputs)

            print(f"logits shape before view: {logits.shape}")
            # Flatten logits to match the expected input to CrossEntropyLoss
            logits = logits.view(-1, logits.size(-1))
            print(f"logits shape after view: {logits.shape}")

            # Flatten labels to match logits shape
            labels = labels.view(-1)
            print(f"labels shape after view: {labels.shape}")

            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            calculated_loss = loss_fn(logits, labels)
            return calculated_loss, logits
        else:
            logits = self.fc(decoder_outputs)
            return logits


train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4)

# Renamed the outer vocab_size variable to avoid shadowing
tokenizer_vocab_size = tokenizer.vocab_size
model = SummarizationModel(vocab_size=tokenizer_vocab_size)
optimizer = AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()  # For mixed precision training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training with gradient accumulation and mixed precision
epochs = 3
accumulation_steps = 4
for epoch in range(epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        batch_input_ids = batch["input_ids"].to(device)
        batch_attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)

        with autocast():
            loss, _ = model(batch_input_ids, batch_attention_mask, batch_labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        if step % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Step {step}, Loss: {loss.item()}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss}")

# Save the final model
model_save_path = "./summarization_model"
os.makedirs(model_save_path, exist_ok=True)
torch.save(model.state_dict(), f"{model_save_path}/model.pt")
tokenizer.save_pretrained(model_save_path)

print("Model and tokenizer saved successfully.")
