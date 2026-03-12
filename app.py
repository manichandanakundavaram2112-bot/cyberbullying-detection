import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

model_path = "unitary/toxic-bert"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

st.set_page_config(page_title="Cyberbullying Detection", page_icon="🛡️")

st.title("🛡️ Cyberbullying Detection System")
st.markdown("This system analyzes online comments and detects possible **cyberbullying or toxic language** using a Machine Learning model.")

user_input = st.text_area("Enter a comment to analyze")

if st.button("Analyze Comment"):

```
inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)

scores = torch.sigmoid(outputs.logits).numpy()[0]

st.subheader("Prediction Results")

for i, label in enumerate(labels):
    st.progress(float(scores[i]))
    st.write(f"{label}: {round(scores[i]*100,2)} %")
```

