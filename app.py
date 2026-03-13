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
st.write("This system analyzes online comments and detects potential cyberbullying using AI.")

# Initialize comment history
if "history" not in st.session_state:
    st.session_state.history = []

# Word dictionaries
bad_words = [
"idiot","stupid","shitty","useless","moron","dumb","kill","die","hate","fool","trash"
]

good_words = [
"beautiful","great","good","nice","amazing","awesome","excellent","love",
"wonderful","fantastic","brilliant","thank","thanks","well done",
"congratulations","appreciate","helpful","kind","happy","friendly"
]

# User input
user_input = st.text_area("Enter a comment to analyze")

if st.button("Analyze Comment"):

    if user_input.strip() == "":
        st.warning("Please enter a comment.")

    else:

        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        scores = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        st.subheader("Toxicity Scores")

        # Show toxicity bars
        for i, label in enumerate(labels):
            score = float(scores[i])
            st.progress(score)
            st.write(f"{label}: {round(score*100,2)} %")

        max_score = max(scores)
        max_label = labels[scores.argmax()]

        st.write(f"Most likely toxicity category: **{max_label}**")

        # Toxicity meter
        st.subheader("Overall Toxicity Meter")
        st.progress(float(max_score))

        text_lower = user_input.lower()

        bad_flag = any(word in text_lower for word in bad_words)
        good_flag = any(word in text_lower for word in good_words)

        # Final classification
        if bad_flag or max_score > 0.60:
            st.error("⚠ Cyberbullying Detected")
            st.write(f"Category: **{max_label}**")
            st.warning("Please reconsider posting harmful language.")

        elif good_flag:
            st.success("✅ Positive Comment")
            st.write("Category: **Friendly / Appreciation**")

        else:
            st.success("✅ Neutral Comment")
            st.write("Category: **Neutral / Safe Comment**")

        # Save comment
        st.session_state.history.append(user_input)

        # Create downloadable report
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

# Display comment history
st.subheader("Analyzed Comments History")

for comment in st.session_state.history:
    st.write(comment)











