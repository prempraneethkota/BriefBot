from datasets import load_dataset

# Load the CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Split the dataset into training and validation
train_data = dataset['train']
valid_data = dataset['validation']

print("Sample training example:")
print(train_data[0])  # Example structure: {'article': ..., 'highlights': ...}
from transformers import AutoTokenizer

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
import torch
import torch.nn as nn
from transformers import BertModel

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
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8)

# Initialize the model, optimizer, and device
vocab_size = tokenizer.vocab_size
model = SummarizationModel(vocab_size=vocab_size)
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")
# Save the model and tokenizer
model_save_path = "./summarization_model"
torch.save(model.state_dict(), f"{model_save_path}/model.pt")
tokenizer.save_pretrained(model_save_path)

print("Model saved successfully.")