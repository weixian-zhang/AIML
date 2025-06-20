import torch
from transformers import pipeline
import gradio as gr

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    # from: https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX04C6EN/Testing%20speech%20to%20text.mp3
    mp3 = 'C:\\Weixian\\projects\\AIML\\certifications\\IBM Generative AI Engineering Specialization\\Testing speech to text.mp3'
    # Perform speech recognition on the audio file
    # The `batch_size=8` parameter indicates how many chunks are processed at a time
    # The result is stored in `prediction` with the key "text" containing the transcribed text
    transcript_result = pipe(audio_file, batch_size=8)["text"]

    return transcript_result


# Set up Gradio interface
audio_input = gr.Audio(sources="upload", type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn=transcript_audio, 
                     inputs=audio_input, outputs=output_text, 
                     title="Audio Transcription App",
                     description="Upload the audio file")

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=7860)