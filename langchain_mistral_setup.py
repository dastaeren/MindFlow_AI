!pip install langchain_community  

!pip install huggingface_hub 

# Logging into Hugging Face Hub
from huggingface_hub import notebook_login
notebook_login()  # Prompts user to log in to Hugging Face

# Installing Additional Packages
!pip install langchain_community  
import torch  
from transformers import BitsAndBytesConfig  
from langchain import HuggingFacePipeline  
from langchain import PromptTemplate, LLMChain  
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

# Installing and Upgrading BitsAndBytes
!pip install -U bitsandbytes  

# Setting up Quantization Configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_use_double_quant=True, 
)

# Loading Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")  # Loads tokenizer for the Mistral-7B model

