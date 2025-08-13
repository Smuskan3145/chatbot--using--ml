# chatbot.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------
# 1. Sample Training Data
# -------------------------
training_sentences = [
    "hello", "hi", "hey", "good morning", "good evening",
    "how are you", "what's up", "how's it going",
    "bye", "see you", "goodbye", "take care",
    "thank you", "thanks a lot", "thanks",
    "what is your name", "who are you",
    "what can you do", "help me", "can you assist me"
]

training_labels = [
    "greeting", "greeting", "greeting", "greeting", "greeting",
    "how_are_you", "how_are_you", "how_are_you",
    "farewell", "farewell", "farewell", "farewell",
    "thanks", "thanks", "thanks",
    "name", "name",
    "capabilities", "capabilities", "capabilities"
]

# -------------------------
# 2. Vectorization
# -------------------------
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_sentences)

# -------------------------
# 3. Model Training
# -------------------------
model = LogisticRegression()
model.fit(X_train, training_labels)

# -------------------------
# 4. Responses
# -------------------------
responses = {
    "greeting": "Hello! How can I help you?",
    "how_are_you": "I'm doing great, thank you! How about you?",
    "farewell": "Goodbye! Have a great day!",
    "thanks": "You're welcome!",
    "name": "I am your friendly AI chatbot 🤖",
    "capabilities": "I can answer basic questions and chat with you."
}

# -------------------------
# 5. Chat Loop
# -------------------------
print("Chatbot is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break

    # Predict intent
    user_vec = vectorizer.transform([user_input])
    intent = model.predict(user_vec)[0]

    # Respond
    print("Bot:", responses.get(intent, "Sorry, I didn't understand that."))
