import sys
sys.stdout.reconfigure(encoding='utf-8')
import gradio as gr
from fpdf import FPDF
from datetime import datetime
import re
import librosa
import numpy as np
from transformers import pipeline
print("please enter the path of directory where files will be downloaded within:")
list_of_allowed_paths = [input()]
# Initialize ASR pipeline
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def add_section(self, title, content):
        self.set_font('Arial', 'B', 13)
        self.cell(0, 8, title, ln=True)
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 8, content)
        self.ln()
# Preprocessing function with POD handling
def preprocess_text(text):
    text = ' '.join(text.split())
    matches = list(re.finditer(r'pouch of douglas', text, re.I))
    if matches:
        last_match = matches[-1]
        findings = text[:last_match.end()].strip()
        comments = text[last_match.end():].strip()
        if not findings.endswith('.'):
            findings += '.'
        if comments:
            comments = comments.strip('.,') + '.'
        else:
            comments = "No additional comments provided."
    else:
        sentences = re.split(r'(?<=[a-zA-Z0-9])\s(?=[A-Z])', text)
        if len(sentences) > 1:
            comments = sentences[-1].strip('.,') + '.'
            findings = ' '.join(sentences[:-1]).rstrip('.,') + '.'
        else:
            findings = text.strip('.,') + '.'
            comments = "No additional comments provided."
    return findings, comments
# PDF Generation function
def generate_pdf(text):
    findings, comments = preprocess_text(text)
    pdf = PDFReport()
    pdf.add_page()
    pdf.add_section('Findings:', findings)
    pdf.add_section('Comments:', comments)
    filename = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    filepath = f'{list_of_allowed_paths[0]}/{filename}'
    pdf.output(filepath)
    return filepath
# Audio transcription function
def transcribe_audio(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    transcription = asr_pipeline({"array": np.array(audio), "sampling_rate": 16000}, return_timestamps=True)
    return transcription['text']
# Gradio Interface
with gr.Blocks() as demo:
    audio_input = gr.Audio(type="filepath", label="Upload Audio File")
    transcribed_output = gr.Textbox(label="Transcribed Text", lines=10, interactive=True)
    transfer_button = gr.Button("Transfer")
    text_input = gr.Textbox(lines=15, placeholder="Paste or transfer medical text here...", label="Medical Report Text")
    pdf_output = gr.File(label="Download PDF Report")
    audio_input.change(fn=transcribe_audio, inputs=audio_input, outputs=transcribed_output)
    transfer_button.click(lambda x: x, inputs=transcribed_output, outputs=text_input)
    generate_btn = gr.Button("Generate PDF")
    generate_btn.click(fn=generate_pdf, inputs=text_input, outputs=pdf_output)
demo.launch(allowed_paths=list_of_allowed_paths)

