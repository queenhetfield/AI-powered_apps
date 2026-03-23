import torch
import os
import gradio as gr
from transformers import pipeline

# initiate LLM instance
generator = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize the speech recognition pipeline
    
pipe = pipeline("automatic-speech-recognition",
    model = "openai/whisper-tiny.en",
    chunk_length_s = 30,
    )

def transcript_audio(audio_file):

    # Transcribe the audio file and return the result
    transcript_txt = pipe(audio_file, batch_size=8)["text"]

    # BART's token limit is ~1024 tokens ≈ 4000 characters
    if len(transcript_txt) > 4000:
        transcript_txt = transcript_txt[:4000]  # trim to safe length

	# run the chain to merge transcript text with the template and send it to the LLM
    output = generator(transcript_txt, max_length=200, min_length=50, do_sample=False)
    result = output[0]["summary_text"]

    return result

#######------------- Gradio-------------####

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn=transcript_audio, 
    inputs=audio_input, outputs=output_text, 
    title="Audio Transcription App",
    description="Upload the audio file")


iface.launch(server_name="0.0.0.0", server_port=7860)
