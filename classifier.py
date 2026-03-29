from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

training_data = [
    # BOOK
    ("I want to book an appointment", "book"),
    ("Can I schedule a meeting", "book"),
    ("I need to make a reservation", "book"),
    ("Book me a slot for tomorrow", "book"),
    ("I want to reserve a table", "book"),
    ("Set up a meeting for me", "book"),
    ("I'd like to schedule something", "book"),
    ("Can you book me in", "book"),
    ("I need an appointment", "book"),
    ("Schedule me for next week", "book"),
    ("I want to set up a call", "book"),
    ("Can I get a time slot", "book"),
    ("I need to book a session", "book"),
    ("Reserve a spot for me", "book"),
    ("I'd like to make an appointment", "book"),
    ("Can you schedule me in for tomorrow", "book"),
    ("I want to register for a slot", "book"),
    ("Please book me an appointment", "book"),
    ("I need to get on the schedule", "book"),
    ("Set me up with an appointment please", "book"),
    ("I want to fix a time to meet", "book"),
    ("Can we schedule a time to talk", "book"),
    ("I need to plan a meeting", "book"),
    ("Put me down for an appointment", "book"),
    ("I would like to arrange a meeting", "book"),

    # CANCEL
    ("I need to cancel my appointment", "cancel"),
    ("Cancel my reservation please", "cancel"),
    ("I want to cancel my booking", "cancel"),
    ("Remove my reservation", "cancel"),
    ("Delete my appointment", "cancel"),
    ("I don't want my booking anymore", "cancel"),
    ("Please cancel my slot", "cancel"),
    ("I need to cancel my meeting", "cancel"),
    ("Can you cancel my reservation", "cancel"),
    ("I want to cancel my session", "cancel"),
    ("Cancel my booking for tomorrow", "cancel"),
    ("I need to call off my appointment", "cancel"),
    ("Please remove my booking", "cancel"),
    ("I want to undo my reservation", "cancel"),
    ("Cancel everything I booked", "cancel"),
    ("I no longer need my appointment", "cancel"),
    ("Please cancel my scheduled time", "cancel"),
    ("I want to cancel the meeting we set up", "cancel"),
    ("Can you remove my time slot", "cancel"),
    ("I need to cancel my upcoming appointment", "cancel"),
    ("Please take me off the schedule", "cancel"),
    ("I changed my mind cancel my booking", "cancel"),
    ("I want to cancel", "cancel"),
    ("Cancel it please", "cancel"),
    ("Remove my booking from the system", "cancel"),

    # ESCALATE
    ("I want to talk to a human", "escalate"),
    ("Connect me to an agent please", "escalate"),
    ("I need to speak to someone", "escalate"),
    ("Transfer me to customer service", "escalate"),
    ("I want a real person", "escalate"),
    ("Let me talk to your manager", "escalate"),
    ("I need human assistance", "escalate"),
    ("Can I speak to a representative", "escalate"),
    ("Get me a real agent", "escalate"),
    ("I want to speak to customer support", "escalate"),
    ("Transfer me to a live agent", "escalate"),
    ("I need to talk to someone real", "escalate"),
    ("Connect me with support staff", "escalate"),
    ("I want to escalate this issue", "escalate"),
    ("Can I talk to a person please", "escalate"),
    ("I need human help", "escalate"),
    ("Put me through to an operator", "escalate"),
    ("I want to speak with a manager", "escalate"),
    ("Can you transfer me to someone", "escalate"),
    ("I need a real person not a bot", "escalate"),
    ("Please connect me to support", "escalate"),
    ("I want to talk to customer care", "escalate"),
    ("Get me someone who can help", "escalate"),
    ("I want to speak to a live person", "escalate"),
    ("Transfer me now please", "escalate"),

    # GENERAL
    ("What is the capital of France", "general"),
    ("How are you today", "general"),
    ("Tell me a joke", "general"),
    ("What time is it", "general"),
    ("How does this work", "general"),
    ("What can you help me with", "general"),
    ("What is the weather like", "general"),
    ("Tell me something interesting", "general"),
    ("What is two plus two", "general"),
    ("Who are you", "general"),
    ("What is your name", "general"),
    ("How old are you", "general"),
    ("What day is it today", "general"),
    ("Can you help me", "general"),
    ("What are your features", "general"),
    ("Tell me about yourself", "general"),
    ("What can you do", "general"),
    ("Give me some information", "general"),
    ("I have a question", "general"),
    ("Can you answer something for me", "general"),
    ("What is the meaning of life", "general"),
    ("How does AI work", "general"),
    ("Tell me the news", "general"),
    ("What is machine learning", "general"),
    ("Explain something to me", "general"),
]

texts = [d[0] for d in training_data]
labels = [d[1] for d in training_data]

classifier = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])
classifier.fit(texts, labels)

def classify_intent(text: str) -> dict:
    if not text:
        return {"intent": "general", "confidence": 0.0}

    text = text.lower().strip()
    intent = classifier.predict([text])[0]
    confidence = float(np.max(classifier.predict_proba([text])))

    return {
        "intent": intent,
        "confidence": round(confidence * 100, 1)
    }