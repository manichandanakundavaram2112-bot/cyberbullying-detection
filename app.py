import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="Cyberbullying Detection",
    page_icon="🛡️",
    layout="centered"
)

# Title
st.title("🛡️ Cyberbullying Detection System")
st.write("Enter a comment below to analyze whether it contains cyberbullying language.")

# Load model
model_name = "unitary/toxic-bert"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# Bad words list
bad_words = [
    "idiot","stupid","shitty","useless","moron","dumb",
    "kill","die","hate","fool","trash"
]

# Input box
user_input = st.text_area("💬 Enter a comment")

if st.button("Analyze Comment"):

    if user_input.strip() == "":
        st.warning("Please enter a comment to analyze.")
    
    else:

        # Tokenize text
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        # Model prediction
        with torch.no_grad():
            outputs = model(**inputs)

        scores = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        max_score = max(scores)
        max_label = labels[scores.argmax()]

        text_lower = user_input.lower()
        bad_flag = any(word in text_lower for word in bad_words)

        # Cyberbullying detected
        if bad_flag or max_score > 0.60:

            st.error("🚨 Cyberbullying Detected")
            st.write(f"**Category:** {max_label}")

            st.subheader("Toxicity Scores")

            # Show bars only for toxic comments
            for i, label in enumerate(labels):
                score = float(scores[i])
                st.progress(score)
                st.write(f"{label}: {round(score*100,2)}%")

        else:

            # Safe comment
            st.success("✅ This comment appears safe.")
            st.info("No harmful language detected.")











