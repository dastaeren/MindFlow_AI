# MindFlow_AI

MindFlow AI is an AI-powered mood journaling application that provides sentiment analysis and mental health recommendations. The application uses machine learning models to analyze user journal entries, determine their sentiment, and generate personalized advice using Gemini and Hugging Face models.

---

## Features

- **Sentiment Analysis**: Uses a pre-trained sentiment analysis model to classify user journal entries.
- **Mental Health Recommendations**: Generates personalized mental health suggestions using:
  - **Gemini AI (Google Generative AI)**
  - **Hugging Face GPT-Neo 1.3B**
- **Journal Entry Management**: Allows users to save and delete journal entries.
- **Streamlit Web Interface**: Provides an interactive UI for users to enter and analyze journal entries.

---

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed.

Install the required dependencies:

```sh
pip install pandas seaborn matplotlib scikit-learn imblearn joblib streamlit transformers google-generativeai langchain_community huggingface_hub bitsandbytes dotenv
```

---

## Setup

### 1. Configure Google Gemini API

Sign up at [Hugging Face](https://huggingface.co/) and Google Gemini to get API keys.

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Load the API key in your Python script:

```python
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
```

### 2. Run the Streamlit App

Execute the following command in your terminal:

```sh
streamlit run app.py
```



```sh

```

---

## Usage

1. Open the Streamlit web application.
2. Enter your journal entry in the text box and click **Submit**.
3. View sentiment analysis and AI-generated mental health recommendations.

---

## File Structure

```
MindEaseAI/
│── stress_analysis.py          # List of dependencies
│── langchain_mistral_setup     # Langchain Community Setup
│── app.py                      # Streamlit application script
│── README.md                   # Project documentation
```

---

## Dependencies

- **pandas**: Data handling
- **seaborn, matplotlib**: Data visualization
- **scikit-learn, imbalanced-learn**: Machine learning and data preprocessing
- **transformers**: Hugging Face NLP models
- **google-generativeai**: Google Gemini API integration
- **streamlit**: Web application framework

---

## Acknowledgments

- OpenAI for the NLP support
- Hugging Face for powerful transformer models
- Google Gemini AI for intelligent recommendations
