import torch
import pickle
from transformers import BertTokenizer
from model_sentiment import BertForSentimentAnalysis  # Make sure to import your specific model class

def load_model_state(model_path, model):
    model_state = torch.load(model_path, map_location=torch.device('cpu'))  # or 'cuda' as appropriate
    model.load_state_dict(model_state)
    model.eval() 

def run_inference(model, tokenizer, input_text):
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=128,  # Adjust as needed
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print(outputs)    
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()

if __name__ == "__main__":
    model = BertForSentimentAnalysis(pretrained_model_name='bert-base-uncased', num_labels=3)
    pickle_path = 'training_results/weights/epoch_4.pickle'  # Update this path
    load_model_state(pickle_path, model)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Adjust as necessary
    input_text = "Enter your text here"  # Your input text
    
    predicted_label = run_inference(model, tokenizer, input_text)
    print(f"Predicted Label: {predicted_label}")

