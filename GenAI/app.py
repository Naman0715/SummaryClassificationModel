# Import Gradio library for creating web-based user interfaces
# Gradio allows us to quickly build and share ML model interfaces without frontend expertise
import gradio as gr

# Import transformers library components:
# - AutoTokenizer: Automatically loads the correct tokenizer for the model
# - AutoModelForSeq2SeqLM: Loads a sequence-to-sequence model for text summarization
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ========== MODEL INITIALIZATION ==========
# Define the path to the pre-trained T5 model directory
model_path = "./model"

# Load the pre-trained T5 model from the specified path
# This model has been fine-tuned for conversational summarization tasks
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Load the corresponding tokenizer that matches the model
# The tokenizer converts raw text into token IDs that the model expects
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ========== SUMMARIZATION FUNCTION ==========
# This function takes user input text and returns an AI-generated summary
def summarize_text(text):
    # Tokenize the input text into token IDs and attention masks
    # return_tensors="pt" converts to PyTorch tensors
    # truncation=True automatically truncates long sequences to model's max length
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    # Generate summary tokens using the model
    # **inputs unpacks the tokenized input dictionary
    # max_new_tokens=50 limits the generated summary length to 50 tokens for conciseness
    outputs = model.generate(
        **inputs,
        max_new_tokens=50
    )

    # Decode the generated token IDs back into human-readable text
    # skip_special_tokens=True removes special tokens, keeping only the actual summary text
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return the generated summary
    return summary

# ========== GRADIO WEB INTERFACE ==========
# Create a Gradio interface that connects the summarization function to a web UI
interface = gr.Interface(
    # fn: The function to run when user submits input
    fn=summarize_text,
    # inputs: A text box for users to paste conversations (10 lines tall)
    inputs=gr.Textbox(lines=10, placeholder="Enter conversation..."),
    # outputs: Display the summary as plain text
    outputs="text",
    # title: Displayed at the top of the web interface
    title="AI Conversation Summarizer",
    # description: Additional info shown below the title
    description="Generative AI Summarization Model"
)

# Launch the Gradio web server
# This starts a local web server (default: http://localhost:7860) for users to interact with the model
interface.launch()