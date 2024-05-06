# ManaGenie 

## Overview

This project is a chatbot implemented in Python using various libraries such as RAG (Retrieval Augmented Generation), LangChain, LlamaCPP, Faiss, Transformers, PyPDF2, and Streamlit. The chatbot provides two main features:

1. **Upload New PDFs**: Users can upload PDF documents, which the chatbot then processes and stores for later use.
2. **Chat with Bot**: Users can engage in conversation with the chatbot, utilizing either the uploaded PDFs or a combination of uploaded and pre-existing data.

Additionally, the chatbot employs LangChain to prompt for the next dialogue in the conversation, ensuring coherence and context continuity.

## Features

- **PDF Upload**: Enables users to upload PDF documents.
- **Chat Interface**: Provides a user-friendly chat interface powered by Streamlit.
- **Integration with Transformers**: Utilizes Transformer-based models for natural language processing tasks.
- **History Saving**: Utilizes LangChain to save conversation history and prompt for the next dialogue.

## Requirements

- Python 3.x
- RAG
- LangChain
- LlamaCPP
- Faiss
- Transformers
- PyPDF2
- Streamlit

## Installation to run locally on your pc

1. Clone the repository:

   ```bash
   git clone https://github.com/mohdtalal3/Chatbot.git

2. Install the required dependencies:    
    ```bash
    pip install -r requirements.txt

## Usage:
1. Navigate to the project directory:
   ```bash
   cd Files

2. Run the Streamlit app::    
    ```bash
   streamlit run app.py


## Installation to run Colab:
1. See or refer to the attached notebook `colab_run.ipynb` to run on colab.


## Project Structure
- `app.py`: Contains the main streamlit code.
- `Rag_note.ipynb`: Contains the code of the main strucutre of chat bot .
- `colab_run.ipynb`: Contains the code to run using colab .
- `Files`: Contains the files to run locally or to run using colab .
- `htmlTemplate.py`: Contains the code for the streamlt frontend .
