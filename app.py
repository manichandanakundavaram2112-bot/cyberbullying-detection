import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
model_path = "unitary/toxic-bert"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

st.title("Cyberbullying Detection System")

st.write("Enter a comment to check if it contains cyberbullying.")

user_input = st.text_area("Enter Comment")

if st.button("Analyze Comment"):

    if user_input.strip() == "":
        st.warning("Please enter a comment")

    else:

        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.sigmoid(outputs.logits)[0]

        st.subheader("Prediction Results")

        for label, prob in zip(labels, probs):

            percentage = float(prob) * 100

            st.write(f"{label} : {percentage:.2f}%")

        if max(probs) > 0.5:
            st.error("⚠️ Toxic or harmful language detected")
        else:

            st.success("Comment appears safe")
