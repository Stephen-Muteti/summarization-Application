import gradio as gr
from transformers import pipeline

# Choose the interface pipeline - Text Summarization
task = "summarization"

# defining the model to use
model_name = "facebook/bart-large-cnn"

# Create the summarization pipeline
summarization_pipeline = pipeline(task, model=model_name)

# Define the function to be used by Gradio for summarization
def summarize_text(input_text):
    summary = summarization_pipeline(input_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    return summary

# Create the Gradio interface
iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.components.Textbox(),
    outputs=gr.components.Textbox(),
    live=True,
    title="Text Summarization App",
    description="Enter some text, and the model will provide a summary."
)

# Launch the Gradio interface
iface.launch(share=True)



