from datasets import load_dataset
from transformers import AutoTokenizer, BertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim  # Use optim from torch
import os

# Load the CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Split the dataset into training and validation
train_data = dataset['train']
valid_data = dataset['validation']

# Define tokenizer (basic tokenizer to preprocess the text)
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

# Apply preprocessing to the dataset
train_dataset = train_data.map(preprocess_function, batched=True)
valid_dataset = valid_data.map(preprocess_function, batched=True)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(["article", "highlights"])
valid_dataset = valid_dataset.remove_columns(["article", "highlights"])
train_dataset.set_format("torch")
valid_dataset.set_format("torch")

# Define the SummarizationModel
class SummarizationModel(nn.Module):
    def __init__(self, vocab_size):
        super(SummarizationModel, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.decoder = nn.GRU(input_size=768, hidden_size=768, num_layers=2, batch_first=True)
        self.fc = nn.Linear(768, vocab_size)

    def forward(self, input_ids, attention_mask, labels=None):
        # Encoder
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Decoder
        decoder_outputs, _ = self.decoder(encoder_hidden_states)
        logits = self.fc(decoder_outputs)

        # Compute loss if labels are provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        return logits

# Create DataLoaders with a smaller batch size
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Reduced batch size
valid_loader = DataLoader(valid_dataset, batch_size=4)  # Reduced batch size

# Initialize the model, optimizer, and device
vocab_size = tokenizer.vocab_size
model = SummarizationModel(vocab_size=vocab_size)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)  # Use torch.optim.AdamW instead
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # Learning rate scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Directory for saving model
model_save_path = "./summarization_model"
os.makedirs(model_save_path, exist_ok=True)  # Create the directory if it doesn't exist

# Training loop
epochs = 3
gradient_accumulation_steps = 2  # Accumulate gradients over a few steps to simulate a larger batch size
for epoch in range(epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    print(f"Starting epoch {epoch + 1}/{epochs}...")

    for i, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        loss, _ = model(input_ids, attention_mask, labels)

        # Backpropagate and accumulate gradients
        loss.backward()

        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            scheduler.step()  # Update learning rate
            optimizer.zero_grad()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:  # Print every 100 batches
            print(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item()}")

    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_loader)}")

    # Free up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, _ = model(input_ids, attention_mask, labels)
            total_val_loss += loss.item()

    print(f"Validation Loss after Epoch {epoch + 1}: {total_val_loss / len(valid_loader)}")

# Save the model and tokenizer
torch.save(model.state_dict(), f"{model_save_path}/model.pt")
tokenizer.save_pretrained(model_save_path)

print("Model saved successfully.")