import gradio as gr
import pdfminer
from pdfminer.high_level import extract_text

def read_pdf(file):
    text = extract_text(file.name)
    return text

iface = gr.Interface(
    read_pdf,
    gr.inputs.File(),
    gr.outputs.Textbox()
)
iface.launch()
