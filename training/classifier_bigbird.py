import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torch.nn import CrossEntropyLoss
from torch import nn


# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Reading data now")

# Load the dataset
data = pd.read_csv('allsides-df-processed.csv')

data = data.dropna(subset=['content'])
data = data.dropna(subset=['bias'])

# Specify the columns you want to keep
columns_to_keep = ['bias', 'content']
filtered_df = data[columns_to_keep]

# Filter and rename columns
df = filtered_df[filtered_df['bias'].isin(['Left', 'Right', 'Center'])]
df.rename(columns={'content': 'text', 'bias': 'label'}, inplace=True)

df['label'] = df['label'].str.lower()

# Define label mapping
label_map = {"left": 0, "right": 1, "center": 2}

# Function to preprocess and map labels
def preprocess_labels(data):
    data['label'] = data['label'].map(label_map)
    return data

df = preprocess_labels(df)
df['text'] = df['text'].astype(str)


# Split the dataset into train, validation, and test sets with a seed for reproducibility
train_df, temp_df = train_test_split(df[['text', 'label']], test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Add class weights
y_train = [item['label'] for item in train_dataset]

class_labels = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float) 

print("Class weights:", class_weights)

#Custom Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        
        device = model.device
        label_weights = class_weights.to(device)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # Compute the loss with class weights
        loss_fct = nn.CrossEntropyLoss(weight=label_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
 

# Load the tokenizer
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base', attn_implementation="sdpa")

# Tokenize the dataset
def tokenize_function(examples):
    tokenized = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=1024)
    if len(tokenized['input_ids']) > 512:
        print("Token length exceeds 512")
        print(len(tokenized['input_ids']))
    return tokenized

print("Tokenizing data now")

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set the format of the datasets for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

print("Loading the model now")

# Load the model
model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', num_labels=3)
model.to(device)

Learning_Rate=2e-5
batch_size_train = 32
batch_size_eval = 32
epoch = 5
Weight_Decay = 0.001
lr_schedular = 'cosine'

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=Learning_Rate,
    per_device_train_batch_size=batch_size_train,
    per_device_eval_batch_size=batch_size_eval,
    num_train_epochs=epoch,
    weight_decay=Weight_Decay,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=1,
    metric_for_best_model="macro_f1",
    load_best_model_at_end=True,
    lr_scheduler_type=lr_schedular,
    gradient_checkpointing=True,
    #gradient_accumulation_steps=16,
    #fp16 = True
)
print("Hyperparameters")
print("Learning rate: ", Learning_Rate)
print("Train batch size: ", batch_size_train)
print("Eval batch size: ", batch_size_eval)
print("Epoch: ", epoch)
print("Weight decay: ", Weight_Decay)
print("Scheduler type:", lr_schedular)

# Function to compute the metrics
f1_metric = load_metric("f1", trust_remote_code=True)
accuracy_metric = load_metric("accuracy", trust_remote_code=True)

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=1)
    
    # Compute F1 score
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
    
    # Compute accuracy
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    
    return {
        'macro_f1': f1['f1'],
        'accuracy': accuracy['accuracy']
    }

# Initialize Trainer with the updated compute_metrics function
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("Training the model")

# Train the model
trainer.train()

print("Printing results")

# Evaluate the model
results = trainer.evaluate(test_dataset)
print(results)
