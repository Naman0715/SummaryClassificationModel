"""
Dialogue Summarization Training Script

This script fine-tunes a T5 model on the SAMSum dataset for abstractive dialogue summarization.
It loads the dataset, tokenizes it, sets up training arguments, and trains the model using Hugging Face Transformers.
"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import evaluate
import numpy as np

# =============================================================================
# DATASET LOADING
# =============================================================================

# Load the SAMSum dataset from Hugging Face Hub
# This dataset contains dialogues and their human-written summaries
dataset = load_dataset("knkarthick/samsum")

# =============================================================================
# MODEL AND TOKENIZER SETUP
# =============================================================================

# Specify the pre-trained model checkpoint to fine-tune
# FLAN-T5-small is a smaller, efficient version of T5 fine-tuned for instruction following
checkpoint = "google/flan-t5-small"

# Load the tokenizer that matches the model
# This will handle text encoding/decoding consistently with the model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the pre-trained model
# AutoModelForSeq2SeqLM automatically selects the right architecture for sequence-to-sequence tasks
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

# Define a function to preprocess the dataset
# This converts raw text dialogues and summaries into tokenized inputs suitable for the model
def preprocess_function(examples):
    # Extract dialogues and summaries from the dataset examples
    inputs = examples["dialogue"]
    targets = examples["summary"]
    
    # Tokenize the input dialogues
    # max_length=512 ensures inputs don't exceed model's context window
    # truncation=True cuts off longer texts, padding="max_length" pads shorter ones
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize the target summaries (these will be the labels for training)
    # max_length=128 is sufficient for most summaries
    labels = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    # Add the tokenized labels to the model inputs
    # The model will learn to generate these token sequences
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Apply the preprocessing function to the entire dataset
# batched=True processes multiple examples at once for efficiency
# This transforms the raw dataset into a format ready for training
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# =============================================================================
# EVALUATION METRICS
# =============================================================================

# Load the ROUGE metric for evaluating summarization quality
# ROUGE measures overlap between generated and reference summaries
rouge = evaluate.load("rouge")
print("ROUGE metric loaded successfully!")

# Define a function to compute metrics during evaluation
# This will be called by the Trainer during validation
def compute_metrics(eval_pred):
    # Unpack the predictions and labels from the evaluation predictions
    predictions, labels = eval_pred
    
    # Decode the predicted token IDs back to text
    # skip_special_tokens=True removes padding and special tokens
    decoded_preds = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True
    )
    
    # Replace -100 labels (used for padding) with pad token ID
    # This is necessary because the model uses -100 to ignore padding tokens during loss calculation
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode the reference labels back to text
    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )
    
    # Compute ROUGE scores comparing predictions to references
    # use_stemmer=True applies stemming for better matching
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    return result

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Set up training arguments - these control all aspects of the training process
training_args = TrainingArguments(
    # Directory to save model checkpoints and logs
    output_dir="./model",
    
    # Learning rate for the optimizer
    learning_rate=2e-5,
    
    # Batch size for training and evaluation per device
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    
    # Weight decay for regularization (prevents overfitting)
    weight_decay=0.01,
    
    # Maximum number of checkpoints to keep
    save_total_limit=2,
    
    # Number of training epochs
    num_train_epochs=1,
    
    # Log training metrics every N steps
    logging_steps=50
)

# =============================================================================
# DATA COLLATION AND TRAINING SETUP
# =============================================================================

# Create a data collator that handles dynamic padding and prepares batches
# DataCollatorForSeq2Seq is specifically designed for sequence-to-sequence tasks
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize the Trainer with all necessary components
trainer = Trainer(
    # The model to train
    model=model,
    
    # Training arguments/configuration
    args=training_args,
    
    # Training dataset (tokenized dialogues and summaries)
    train_dataset=tokenized_dataset["train"],
    
    # Validation dataset for evaluation during training
    eval_dataset=tokenized_dataset["validation"],
    
    # Data collator for batching
    data_collator=data_collator,
    
    # Function to compute evaluation metrics
    compute_metrics=compute_metrics
)

# =============================================================================
# MODEL TRAINING
# =============================================================================

# Start the training process
# This will fine-tune the model on the dialogue summarization task
trainer.train()

# =============================================================================
# MODEL SAVING
# =============================================================================


# Save the fine-tuned model and tokenizer to disk
# This allows loading the trained model later for inference
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

# Print completion message
print("Training completed successfully!")