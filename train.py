import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

# Configuration
SHOULD_TRAIN = False
MODEL_DIR = "./financial-sentiment-model"
FINAL_MODEL_PATH = "./financial-sentiment-model-final.pth"

# Load dataset and tokenizer
print("Loading dataset and tokenizer...")
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
print("Available dataset splits:", dataset.keys())  # Debug print to see available splits

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def find_best_checkpoint():
    if not os.path.exists(MODEL_DIR):
        return None
        
    checkpoints = [d for d in os.listdir(MODEL_DIR) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
        
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    latest_checkpoint = os.path.join(MODEL_DIR, checkpoints[-1])
    return latest_checkpoint

def load_or_train_model():
    if not SHOULD_TRAIN:
        if os.path.exists(FINAL_MODEL_PATH):
            print("Loading final model...")
            return DistilBertForSequenceClassification.from_pretrained(FINAL_MODEL_PATH)
            
        best_checkpoint = find_best_checkpoint()
        if best_checkpoint:
            print(f"Loading from checkpoint: {best_checkpoint}")
            return DistilBertForSequenceClassification.from_pretrained(best_checkpoint)
            
        print("No model or checkpoint found. Please either:")
        print("1. Enable training (SHOULD_TRAIN = True)")
        print("2. Provide a valid model path")
        print("3. Ensure checkpoints exist in the model directory")
        raise FileNotFoundError("No model available")
    
    return train_model()

def train_model():
    print("Starting model training process...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3 
    )
    
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_steps=100,
        logging_dir="./logs",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(FINAL_MODEL_PATH)
    return model

def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors=None,
        labels=examples['label']
    )

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def predict_sentiment(text, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    
    sentiment_map = {
        0: "Bearish",
        1: "Bullish",
        2: "Neutral"
    }
    return sentiment_map[prediction]

# Main execution
try:
    # Load model
    model = load_or_train_model()
    print("Model loaded successfully!")
    
    # Print dataset information
    print("\nDataset Information:")
    for split_name in dataset.keys():
        print(f"Split: {split_name}, Size: {len(dataset[split_name])}")
        print(f"Columns: {dataset[split_name].column_names}")
    
    # Use validation set for testing predictions since test set might not exist
    eval_split = 'validation' if 'validation' in dataset else 'train'
    print(f"\nUsing {eval_split} split for predictions")
    
    # Test predictions
    print("\nTesting predictions on sample data:")
    eval_data = dataset[eval_split]
    
    for i in range(min(5, len(eval_data))):
        example = eval_data[i]
        text = example['text']
        true_label = example['label']
        sentiment = predict_sentiment(text, model)
        print(f"\nText: {text}")
        print(f"True Label: {true_label}")
        print(f"Predicted Sentiment: {sentiment}")
        
except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    print("\nDetailed error information:")
    print(traceback.format_exc())