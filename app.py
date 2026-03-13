import streamlit as st
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load pretrained model
model_name = "unitary/toxic-bert"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

st.set_page_config(page_title="Cyberbullying Detection", page_icon="🛡")

st.title("🛡 Cyberbullying Detection System")
st.write("Analyze comments and detect potential cyberbullying using a DistilBERT NLP model.")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_area("Enter a comment to analyze")

if st.button("Analyze Comment"):

    if user_input.strip() == "":
        st.warning("Please enter a comment.")

    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        scores = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        st.subheader("Prediction Results")

        # Show probability bars
        for i, label in enumerate(labels):
            score = float(scores[i])
            st.progress(score)
            st.write(f"{label}: {round(score*100,2)} %")

        # Detection logic using highest probability
        max_score = max(scores)
        max_label = labels[scores.argmax()]

        st.write(f"Most likely category: **{max_label}**")

        if max_score > 0.45:
            st.error(f"⚠️ Cyberbullying detected: {max_label}")
            st.warning("⚠ Please reconsider posting harmful language.")
        else:
            st.success("✅ This comment appears safe.")

        # Save comment history
        st.session_state.history.append(user_input)

        # Download report
        report = pd.DataFrame({
            "Label": labels,
            "Score": scores
        })

        st.download_button(
            label="Download Analysis Report",
            data=report.to_csv(index=False),
            file_name="cyberbullying_analysis.csv",
            mime="text/csv"
        )

# Display history
st.subheader("Analyzed Comments History")

for comment in st.session_state.history:
    st.write(comment)











