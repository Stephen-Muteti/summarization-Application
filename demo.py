# Import necessary libraries
from transformers import pipeline

# Step 1: Choose the interface pipeline - Text Summarization
task = "summarization"

# Step 2: Create the summarization pipeline
summarization_pipeline = pipeline(task)

# Step 3: Sample example - Summarize a piece of text using the default model
sample_text = """
Hugging Face is a technology company based in New York City that specializes in natural language processing. 
They have developed the Transformers library, which is widely used for various NLP tasks.
"""

default_summary = summarization_pipeline(sample_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
print("Default Model Summary:", default_summary)

# Step 4: Change the default model to another suitable model
new_model_name = "facebook/bart-large-cnn"
summarization_pipeline_new = pipeline(task, model=new_model_name)

# Step 5: Example with the new model
new_summary = summarization_pipeline_new(sample_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
print("New Model Summary:", new_summary)




