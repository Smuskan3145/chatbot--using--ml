# Chatbot using Machine Learning

A Machine Learning based chatbot that can interact with users, understand their queries, and provide intelligent responses using Natural Language Processing (NLP) techniques. This project demonstrates how machine learning models can be trained on a dataset of questions and answers to provide context-aware responses.

## Objectives
- Understand and process natural language input.
- Provide accurate and relevant chatbot responses.
- Showcase the use of ML algorithms in conversational AI.

## Technologies Used
Python, Pandas, Scikit-learn, NLTK/spaCy, Flask (optional for web UI).

## Dataset
The dataset (CSV file) contains user queries and their respective answers. Example columns: Question | Answer.

## How to Run
1. Clone the repository:  
   git clone https://github.com/Smuskan3145/chatbot--using--ml.git
2. Open the folder:  
   cd chatbot--using--ml
3. Install dependencies:  
   pip install -r requirements.txt
4. Run the chatbot:  
   python chatbot.py

## Workflow
Data preprocessing → Model training → Prediction → Response generation.  
Text is tokenized, cleaned, and converted into numerical vectors before training the ML model. The model predicts the best response for a given input.

## Output Example
User: Hello  
Bot: Hi there! How can I help you today?
