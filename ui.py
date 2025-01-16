import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Configuration
MODEL_DIR = "./financial-sentiment-model"
FINAL_MODEL_PATH = "./financial-sentiment-model-final.pth"

# Load tokenizer and model
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    if FINAL_MODEL_PATH:
        model = DistilBertForSequenceClassification.from_pretrained(FINAL_MODEL_PATH)
    else:
        raise FileNotFoundError("Final model file not found.")
    return tokenizer, model

# Predict sentiment function
def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    confidence = float(torch.max(probs))  # Confidence score
    
    sentiment_map = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
    return sentiment_map[prediction], confidence

# Generate example texts dynamically
def generate_examples(model, tokenizer):
    example_texts = [
        "The market is struggling after a sharp decline in tech stocks.",
        "Amazon stock prices hit record highs following the new product launch.",
        "The Federal Reserve's uncertain policy creates mixed feelings in the market.",
    ]
    examples = []
    for text in example_texts:
        sentiment, confidence = predict_sentiment(text, model, tokenizer)
        examples.append(f"{text} (Sentiment: {sentiment}, Confidence: {confidence:.2%})")
    return examples

# Streamlit UI
def main():
    st.title("Financial Sentiment Analysis")
    st.write("Analyze the sentiment of financial news or social media text using a pre-trained model.")

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        st.write("Example Predictions from the Model:")
        example_predictions = generate_examples(model, tokenizer)
        for example in example_predictions:
            st.write(f"- {example}")
        st.write("Or, input your own text below:")

    # User input
    user_input = st.text_area("Enter financial text for sentiment analysis:", "")
    
    if st.button("Analyze Sentiment"):
        try:
            sentiment, confidence = predict_sentiment(user_input, model, tokenizer)
            
            st.subheader("Prediction Results:")
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Confidence Score:** {confidence:.2%}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
