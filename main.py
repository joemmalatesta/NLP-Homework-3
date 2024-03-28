import datasets
import torch 
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
# Load hate speech detection dataset from huggingface
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'default')   
df = dataset['train'].to_pandas()
df.describe()


# Split training and testing in 80% train and 20% test
trainDf, testDf = train_test_split(df, test_size=0.2)


# Load BERT and roBERTa tokenizers and models
bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
robertaTokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bertModel = BertForSequenceClassification.from_pretrained('bert-base-uncased')
robertaModel = RobertaForSequenceClassification.from_pretrained('roberta-base')


def encode_dataset(tokenizer, texts, labels=None, max_length=512):
    # Tokenize the text
    encodedBatch = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    # If labels are provided, return tensors of them as well
    if labels is not None:
        labelsTensor = torch.tensor(labels)
        return encodedBatch, labelsTensor
    return encodedBatch

# Encode the datasets
trainEncoded, trainLabels = encode_dataset(bertTokenizer, trainDf['text'].tolist(), trainDf['sentiment'].tolist())
testEncoded, testLabels = encode_dataset(bertTokenizer, testDf['text'].tolist(), testDf['sentiment'].tolist())


from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW

# Create TensorDatasets and DataLoaders
trainDataset = TensorDataset(trainEncoded['input_ids'], trainEncoded['attention_mask'], trainLabels)
testDataset = TensorDataset(testEncoded['input_ids'], testEncoded['attention_mask'], testLabels)

trainLoader = DataLoader(trainDataset, batch_size=8)
testLoader = DataLoader(testDataset, batch_size=8)

# Example of setting up the optimizer
optimizer = AdamW(bertModel.parameters(), lr=5e-5)
print(trainDataset, testDataset) 


###############################################
#
#
# TRAINING LOOP
#
#
################################################
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

print("checking in on a fellow")
# Number of training epochs
epochs = 4


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
bertModel.to(device)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Define the optimizer
optimizer = AdamW(bertModel.parameters(), lr=5e-5)

# Total number of training steps
total_steps = len(trainLoader) * epochs

# Set up the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=total_steps)



# Training loop
for epoch in range(epochs):
    bertModel.train()
    total_loss = 0
    predictions, true_labels = [], []

    for batch in trainLoader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        
        outputs = bertModel(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
        
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)

    avg_train_loss = total_loss / len(trainLoader)
    print(f"Average training loss: {avg_train_loss}")

    # Validation step
    bertModel.eval()
    eval_loss, eval_accuracy = 0, 0
    for batch in testLoader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = bertModel(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
        
        logits = outputs[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = accuracy_score(np.argmax(logits, axis=1).flatten(), label_ids.flatten())

        eval_accuracy += tmp_eval_accuracy

    print(f"Validation Accuracy: {eval_accuracy / len(testLoader)}")