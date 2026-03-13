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
st.write("This system analyzes online comments and detects potential cyberbullying using a machine learning model.")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
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

        # Show probability bars
        for i, label in enumerate(labels):
            score = float(scores[i])
            st.progress(score)
            st.write(f"{label}: {round(score*100,2)} %")

        # Most toxic category
        max_score = max(scores)
        max_label = labels[scores.argmax()]

        st.write(f"Most likely category: **{max_label}**")

        # Individual scores
        toxic_score = scores[0]
        severe_score = scores[1]
        obscene_score = scores[2]
        threat_score = scores[3]
        insult_score = scores[4]
        identity_score = scores[5]

        # Detection logic
        if (
            toxic_score > 0.35 or
            insult_score > 0.35 or
            obscene_score > 0.40 or
            severe_score > 0.50 or
            threat_score > 0.50 or
            identity_score > 0.50
        ):
            st.error("⚠️ Cyberbullying detected in the comment.")
            st.warning("⚠ Please reconsider posting harmful language.")
        else:
            st.success("✅ This comment appears safe.")

        # Save comment
        st.session_state.history.append(user_input)

        # Create report
        data = {
            "Label": labels,
            "Score": scores
        }

        df = pd.DataFrame(data)

        st.download_button(
            label="Download Analysis Report",
            data=df.to_csv(index=False),
            file_name="cyberbullying_analysis.csv",
            mime="text/csv"
        )

# Comment history
st.subheader("Analyzed Comments History")

for comment in st.session_state.history:
    st.write(comment)












