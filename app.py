pip install google-generativeai

GEMINI_API_KEY= "AIzaSyDykkBaIChdFmj2NYtraLERwTtl5_ar5Do"

import google.generativeai as genai
import os
from dotenv import load_dotenv

api_key = os.getenv("AIzaSyDykkBaIChdFmj2NYtraLERwTtl5_ar5Do")

[{"name":"GEMINI_API_KEY","value":"39 chars 'AIzaSyDykkBaIChdFmj2…","type":"str"},{"name":"api_key","value":"None","type":"NoneType"}]

# Load variables from .env file
load_dotenv()

# Ensure the API key is loaded
# Instead of os.getenv(), use the GEMINI_API_KEY variable directly
api_key = GEMINI_API_KEY
if not api_key:
    raise ValueError("API key not found. Check your .env file or set GEMINI_API_KEY manually.")

# Configure the Gemini API key
genai.configure(api_key=api_key)

def generate_gemini_recommendation(entry):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"I feel {entry}. Can you give me some advice?")
    return response.text

# Test the function
print(generate_gemini_recommendation("stressed"))

!pip install streamlit

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from transformers import pipeline
import datetime
import json
import os
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.environ.get("AIzaSyDykkBaIChdFmj2NYtraLERwTtl5_ar5Do"))

# Initialize sentiment analysis and text generation pipelines
@st.cache_resource
def get_sentiment_pipeline():
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline

@st.cache_resource
def get_text_generation_pipeline():
    text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
    return text_generator

# Function to save journal entry
def save_journal_entry(entry):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    journal_entry = {"timestamp": timestamp, "entry": entry}

    try:
        with open("journal_entries.json", "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(journal_entry)

    with open("journal_entries.json", "w") as file:
        json.dump(data, file, indent=4)

    return "Journal entry saved successfully."

# Function to delete all journal entries
def delete_all_entries():
    try:
        os.remove("journal_entries.json")
        return "All journal entries have been deleted."
    except FileNotFoundError:
        return "No journal entries found to delete."

# Function to analyze sentiment
def analyze_sentiment(entry):
    sentiment_pipeline = get_sentiment_pipeline()
    result = sentiment_pipeline(entry)[0]
    return result["label"], result["score"]

# Function to generate mental health recommendation using Gemini
def generate_gemini_recommendation(entry):
    model = GenerativeModel("gemini-pro")
    response = model.generate_content(f"I feel {entry}. Can you give me some advice?")
    return response.text

# Function to generate mental health recommendation using Hugging Face
def generate_huggingface_recommendation(entry):
    text_generator = get_text_generation_pipeline()
    generated_text = text_generator(f"Give mental health advice based on: {entry}", max_length=100)[0]["generated_text"]
    return generated_text

# Streamlit App
st.title("MindFlow AI")

user_input = st.text_area("Write your journal entry:")

if st.button("Submit"):
    save_message = save_journal_entry(user_input)
    st.success(save_message)

    sentiment, confidence = analyze_sentiment(user_input)
    st.write(f"Sentiment Analysis: {sentiment} (Confidence: {confidence:.2f})")

    gemini_recommendation = generate_gemini_recommendation(user_input)
    huggingface_recommendation = generate_huggingface_recommendation(user_input)

    st.subheader("Suggestion:")
    st.write(gemini_recommendation)

    st.subheader("Mental Health Recommendation (Hugging Face GPT-Neo):")
    st.write(huggingface_recommendation)

# Add a button to delete all journal entries
if st.button("Delete All Journal Entries"):
    delete_message = delete_all_entries()
    st.warning(delete_message)

#Open streamlit
!wget -q -O - ipv4.icanhazip.com

! streamlit run app.py & npx localtunnel --port 8501