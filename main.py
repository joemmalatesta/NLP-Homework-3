from openai import OpenAI
import datasets
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
# Load hate speech detection dataset from huggingface
dataset = datasets.load_dataset(
    'ucberkeley-dlab/measuring-hate-speech', 'default')
df = dataset['train'].to_pandas()
df.describe()


# Split training and testing in 80% train and 20% test
trainDf, testDf = train_test_split(df, test_size=0.2)
# Example remapping for binary classification
trainDf['hate_speech_score'] = trainDf['hate_speech_score'].apply(lambda x: 0 if x < 0 else 1)
testDf['hate_speech_score'] = testDf['hate_speech_score'].apply(lambda x: 0 if x < 0 else 1)

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


# # Encode the datasets
trainEncoded, trainLabels = encode_dataset(bertTokenizer, trainDf['text'].tolist(), trainDf['hate_speech_score'].tolist())
testEncoded, testLabels = encode_dataset(bertTokenizer, testDf['text'].tolist(), testDf['hate_speech_score'].tolist())
# Assuming `trainDf` and `testDf` have a column `hate_speech_score` that you wish to use as labels
trainLabels = torch.tensor(trainDf['hate_speech_score'].values.astype(int)).long()
testLabels = torch.tensor(testDf['hate_speech_score'].values.astype(int)).long()

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

# Number of training epochs
epochs = 4

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the device
bertModel.to(device)

# Define the loss function
lossFn = torch.nn.CrossEntropyLoss()

# Define the optimizer
optimizer = AdamW(bertModel.parameters(), lr=5e-5)

# Total number of training steps
totalSteps = len(trainLoader) * epochs

# Set up the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=totalSteps)

# Training loop
for epoch in range(epochs):
    bertModel.train() # Enter training mode
    totalLoss = 0
    predictions, trueLabels = [], []

    for batch in trainLoader:
        # Unpack the batch and move to the specified device
        bInputIds, bInputMask, bLabels = [t.to(device) for t in batch]
        
        optimizer.zero_grad() # Clear previous gradients

        # Forward pass
        outputs = bertModel(bInputIds, token_type_ids=None, attention_mask=bInputMask, labels=bLabels)
        
        loss = outputs.loss # Extract the loss
        totalLoss += loss.item() # Accumulate the loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Store predictions and true labels for evaluation
        logits = outputs.logits.detach().cpu().numpy()
        labelIds = bLabels.to('cpu').numpy()
        predictions.extend(logits)
        trueLabels.extend(labelIds)

    # Calculate average loss over the training data
    avgTrainLoss = totalLoss / len(trainLoader)
    print(f"Epoch {epoch+1}/{epochs} - Average training loss: {avgTrainLoss}")

    # Validation step
    bertModel.eval() # Enter evaluation mode
    evalAccuracy = 0
    for batch in testLoader:
        bInputIds, bInputMask, bLabels = [t.to(device) for t in batch]

        with torch.no_grad(): # Disable gradient calculation
            outputs = bertModel(bInputIds, token_type_ids=None, attention_mask=bInputMask)

        logits = outputs.logits.detach().cpu().numpy()
        labelIds = bLabels.to('cpu').numpy()

        # Calculate and accumulate the accuracy
        tmpEvalAccuracy = accuracy_score(np.argmax(logits, axis=1).flatten(), labelIds.flatten())
        evalAccuracy += tmpEvalAccuracy

    # Print validation accuracy
    print(f"Validation Accuracy: {evalAccuracy / len(testLoader)}")


#############################
#
#
# Zero Shot Classification / Prompt Engineering
#
#
##############################

# Take 100 examples of
# zeroShotInput = trainDf['text'].tolist()[:200]
# zeroShotLabels = trainDf['hate_speech_score'].tolist()[:200]


# prompt = f"""
# Imagine you are a researcher tasked with analyzing comments and evaluating them as either hate speech or not. 
# Hate speech is defined as any communication that belittles a person or a group on the basis of attributes such as race, religion, ethnic origin, sexual orientation, disability, or gender. 
# It can include, but is not limited to, promoting hatred, inciting violence, or suggesting harm towards specific groups or individuals. 
# When analyzing the comment, please be careful in your assessment and take into account the criteria mentioned prior. 
# Is the following comment hate speech? Your answer should not include any words other than "yes" or "no"\n\n
# """

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key="Not committing that",
# )
# mismatchCount = 0
# for index, comment in enumerate(zeroShotInput):
#     label = zeroShotLabels[index]
#     chatCompletion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt+comment,
#             }
#         ],
#         model="gpt-4-turbo-preview",
#     )
#     output = chatCompletion.choices[0].message.content
#     print(f'GPT:{output} - Label:{label}')

#     if label > 0 and output.strip().lower() == "no":
#         mismatchCount += 1
#     if label < 0 and output.strip().lower() == "yes":
#         mismatchCount += 1
# print(f'{mismatchCount} Mismatched out of {len(zeroShotInput)}')






#############################################
#
#
# BASELINES
#
#
#############################################
    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

# Creating a pipeline that first vectorizes the text and then applies logistic regression
pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(solver='liblinear')
)

# Training the model
pipeline.fit(trainDf['text'], trainDf['hate_speech_score'].astype(int))

# Predicting on the test set
predictions = pipeline.predict(testDf['text'])

# Evaluating the model
accuracy = accuracy_score(testDf['hate_speech_score'].astype(int), predictions)
f1 = f1_score(testDf['hate_speech_score'].astype(int), predictions, average='binary')

print(f"BoW + Logistic Regression Accuracy: {accuracy}")
print(f"BoW + Logistic Regression F1 Score: {f1}")


import numpy as np
randomPredictions = np.random.randint(2, size=len(testDf))
randomAccuracy = accuracy_score(testDf['hate_speech_score'].astype(int), randomPredictions)
randomF1 = f1_score(testDf['hate_speech_score'].astype(int), randomPredictions, average='binary')



majorityClass = trainDf['hate_speech_score'].astype(int).mode()[0]
majorityPredictions = [majorityClass] * len(testDf)
majorityAccuracy = accuracy_score(testDf['hate_speech_score'].astype(int), majorityPredictions)
# F1 score is context-dependent; for binary classification, it might be insightful or not, depending on class distribution.