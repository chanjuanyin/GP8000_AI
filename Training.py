import os
os.environ["USE_TF"] = "0"  # Ensure transformers uses PyTorch only

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Step 0: Ensure the output directory exists
output_dir = "./flood-dialoGPT"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 1: Define Sample Conversations
conversations = [
    {"input": "What should I do during a flood?", "response": "Stay indoors, avoid flooded areas, and listen to emergency services."},
    {"input": "How can I prepare for a flood?", "response": "Keep emergency supplies, know your evacuation routes, and stay informed."},
    {"input": "What are the common signs of a flood?", "response": "Rising water levels, continuous rain, and overflowing rivers."},
]

# Step 2: Create a Custom Dataset Class
class FloodDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.conversations = conversations
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        input_text = self.conversations[idx]["input"]
        response_text = self.conversations[idx]["response"]
        combined = f"{input_text} [SEP] {response_text}"
        
        # Tokenize and encode the text
        encoded = self.tokenizer.encode_plus(
            combined,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        
        # To calculate loss, we need labels; the labels will be the same as the input_ids
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# Step 3: Load Model and Tokenizer
model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Step 4: Prepare the Dataset
train_dataset = FloodDataset(conversations, tokenizer)

# Step 5: Set Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_steps=10,  # Save more frequently for testing
    save_total_limit=2,  # Save only 2 checkpoints at most
    logging_dir='./logs',
    logging_steps=10,  # Log every 10 steps
)

# Step 6: Initialize Trainer and Start Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Step 7: Save the Model and Tokenizer
trainer.save_model(output_dir)  # Save the final model
tokenizer.save_pretrained(output_dir)
