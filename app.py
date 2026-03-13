import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load pretrained model
model_name = "unitary/toxic-bert"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

st.set_page_config(page_title="Cyberbullying Detection", page_icon="🛡")

st.title("🛡 Cyberbullying Detection System")
st.write("This system analyzes online comments and detects potential cyberbullying using a machine learning model.")

user_input = st.text_area("Enter a comment to analyze")

if st.button("Analyze Comment"):

    if user_input.strip() == "":
        st.warning("Please enter a comment.")

    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        scores = torch.sigmoid(outputs.logits).numpy()[0]

        st.subheader("Prediction Results")

        # Show probabilities
        for i, label in enumerate(labels):
            score = float(scores[i])
            st.progress(score)
            st.write(f"{label}: {round(score*100,2)} %")

        # Determine highest probability label
        max_score = max(scores)
        max_label = labels[scores.argmax()]

        # Threshold decision
        if max_score > 0.4:
            st.error(f"⚠️ Cyberbullying detected: {max_label}")
        else:
            st.success("✅ This comment appears safe.")











