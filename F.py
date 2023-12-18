import PyPDF4
import pyttsx3
import torch
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt

def pdf_to_audio(file_name):
    
   
    engine = pyttsx3.init()

    
    summarizer = pipeline("summarization")

    pdf_reader = PyPDF4.PdfFileReader(file_name)
    text_lengths = []
    audio_lengths = []

    for page_number in range(pdf_reader.numPages):
       
        text = page.extractText()

        
        summary = summarizer(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['summary_text']

     
        engine.save_to_file(summary, f"page_{page_number + 1}.mp3")

      
        text_lengths.append(len(text))
        audio_lengths.append(len(summary))

    
    engine.runAndWait()
  
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(text_lengths) + 1), y=text_lengths, label='Original Text')
    sns.lineplot(x=range(1, len(audio_lengths) + 1), y=audio_lengths, label='Summarized Text (Audio)')
    plt.title('Text and Audio Lengths Over Pages')
    plt.xlabel('Page Number')
    plt.ylabel('Length')
    plt.legend()
    plt.show()


pdf_to_audio("File.pdf")
