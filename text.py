import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import EncoderDecoderModel, BertTokenizerFast
from torch.utils.data import DataLoader
import numpy as np

class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, articles, summaries, tokenizer, max_input_length=512, max_output_length=150):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        summary = self.summaries[idx]

        inputs = self.tokenizer.encode_plus(
            article,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = self.tokenizer.encode(
            summary,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": labels.flatten(),
        }

def train_summarization_model(articles, summaries, model_name="bert-base-uncased", batch_size=4, epochs=3):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    dataset = SummarizationDataset(articles, summaries, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

   
    model.train()
    model.to('cuda') 

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to('cuda')
            attention_mask = batch["attention_mask"].to('cuda')
            labels = batch["labels"].to('cuda')

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=labels, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    
    model.save_pretrained("summarization_model")


articles = [
    "This is the first news article.",
    "This is the second news article.",
    "This is the third news article."
]
summaries = [
    "Summary of the first news article.",
    "Summary of the second news article.",
    "Summary of the third news article."
]


train_summarization_model(articles, summaries)
