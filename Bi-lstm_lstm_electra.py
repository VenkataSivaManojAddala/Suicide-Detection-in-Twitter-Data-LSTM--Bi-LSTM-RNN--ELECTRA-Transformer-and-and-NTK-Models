# Import necessary libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score, precision_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import wandb
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_metric

# Load dataset
suicide_detection_df = pd.read_csv('suicide_detection_final_cleaned.csv', header=0)
suicide_detection_df.drop(columns=['text'], axis=1, inplace=True)
suicide_detection_df = suicide_detection_df.rename(columns={"cleaned_text": "text"})
classes = {"suicide": 1, "non-suicide": 0}
suicide_detection_df = suicide_detection_df.replace({"class": classes})
suicide_detection_df.head()

# Split dataset into train, validation and test sets
train_text, temp_text, train_labels, temp_labels = train_test_split(suicide_detection_df['text'], suicide_detection_df['class'],
                                                                    random_state=SEED,
                                                                    test_size=0.2,
                                                                    stratify=suicide_detection_df['class'])

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=SEED,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

max_length = max([len(s.split()) for s in train_text])
max_length

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_text)

vocab_size = len(tokenizer.word_index) + 1
# Define constants
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
SEED = 4222
def tokenize_and_encode(text, max_length=62):
    """Tokenize and encode sequences."""

    # sequence encode
    encoded_docs = tokenizer.texts_to_sequences(text)
    # pad sequences
    padded_sequence = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    return padded_sequence

# Tokenize and encode sequences in all datasets
tokens_train = tokenize_and_encode(train_text)
tokens_val = tokenize_and_encode(val_text)
tokens_test = tokenize_and_encode(test_text)

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(tokens_train), torch.from_numpy(train_labels.to_numpy()))
val_data = TensorDataset(torch.from_numpy(tokens_val), torch.from_numpy(val_labels.to_numpy()))

# Sampler for sampling the data
train_sampler = RandomSampler(train_data)
val_sampler = SequentialSampler(val_data)

# DataLoader
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()[1:]
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
	return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab, embedding_dim):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = np.zeros((vocab_size, embedding_dim))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		weight_matrix[i] = embedding.get(word)

	return weight_matrix

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape[0], weights_matrix.shape[1]
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
    # emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

# load word2vec embedding from file
raw_embedding_word2vec = load_embedding('Data/embedding_word2vec.txt') # 'Data/embedding_word2vec_1.txt'
# get vectors in the right order
embedding_vectors_word2vec = get_weight_matrix(raw_embedding_word2vec, tokenizer.word_index, 300)
embedding_vectors_word2vec = np.float32(embedding_vectors_word2vec)

# load glove embedding from file
raw_embedding_glove = load_embedding('Data/glove_twitter_27B_200d.txt')

# get vectors in the right order
embedding_vectors_glove = get_weight_matrix(raw_embedding_glove, tokenizer.word_index, 200)
embedding_vectors_glove = np.float32(embedding_vectors_glove)

for arr in embedding_vectors_glove:
    for idx, i in enumerate(arr):
        if np.isnan(arr[idx]):
            arr[idx] = 0

class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, 
                 embedding_dim, hidden_dim, n_layers, 
                 dropout_rate, pre_trained=False, embedding_vectors=None):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        if pre_trained:
            self.embedding, num_embeddings, embedding_dim = create_emb_layer(embedding_vectors, True)
        else:
            # Create word embeddings from the input words
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=dropout_rate, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3) # dropout_rate
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        
        return hidden
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Instantiate the model w/ hyperparams
embedding_dim = 300
hidden_dim = 128
output_size = 1
n_layers = 2
dropout = 0.5

# Define the loss function
criterion = nn.BCEWithLogitsLoss()

# push to GPU
criterion = criterion.to(device)

model1 = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout)

print("No pre trained embedding weights")
print(model1)

print(f'Model 1 has {count_parameters(model1):,} trainable parameters')

model2 = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout, 
                       pre_trained=True, embedding_vectors=embedding_vectors_word2vec)
model2.embedding.weight.data.copy_(torch.from_numpy(embedding_vectors_word2vec))

print("With Word2Vec pre trained embedding weights")
print(model2)

print(f'Model 2 has {count_parameters(model2):,} trainable parameters')

model3 = SentimentLSTM(vocab_size, output_size, 200, hidden_dim, n_layers, dropout, 
                       pre_trained=True, embedding_vectors=embedding_vectors_glove)
model3.embedding.weight.data.copy_(torch.from_numpy(embedding_vectors_glove))

print("With gloVe pre trained embedding weights")
print(model3)

print(f'Model 3 has {count_parameters(model3):,} trainable parameters')

def binary_accuracy(preds, labels):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # convert output probabilities to predicted class (0 or 1)
    preds = torch.round(preds.squeeze())  # rounds to the nearest integer
    
    # compare predictions to true label
    correct_tensor = preds.eq(labels.float().view_as(preds))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct = np.sum(correct)

    acc = num_correct/len(correct)

    return acc

# function to train the model
def train():

    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches of train data
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('Batch {:>5,} of {:>5,}.'.format(step, len(train_dataloader)))
        
        # push the batch to gpu
        # batch = [r.to(device) for r in batch]

        inputs, labels = batch
        inputs = inputs.type(torch.LongTensor)

        # initialize hidden state
        h = model.init_hidden(len(inputs))

        # move to gpu
        inputs, labels = inputs.cuda(), labels.cuda()

        # Create new variables for the hidden state
        h = tuple([each.data for each in h])

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for current batch
        preds, h = model(inputs, h)
        
        # compute the loss between actual and predicted values
        loss = criterion(preds.squeeze(), labels.float())

        # add on to the total loss
        total_loss += loss.item()
        
        # backward pass to calculate the gradients
        loss.backward()

        # clip the gradients to 1.0. It helps in preventing the exploding gradient problem
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        # compute accuracy
        acc = binary_accuracy(preds, labels)

        # add on to the total accuracy
        total_accuracy += acc.item()

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # compute the training acc of the epoch
    avg_acc = total_accuracy / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss, accuracy and predictions
    return avg_loss, avg_acc, total_preds

# function for evaluating the model
def evaluate():

    print("\nEvaluating...")

    # deactivate dropout layers

    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('Batch {:>5,} of {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        # batch = [t.to(device) for t in batch]

        inputs, labels = batch
        inputs = inputs.type(torch.LongTensor)

        # initialize hidden state
        val_h = model.init_hidden(len(inputs))

        # move to gpu
        inputs, labels = inputs.cuda(), labels.cuda()

        # Create new variables for the hidden state
        val_h = tuple([each.data for each in val_h])

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds, val_h = model(inputs, val_h)

            # compute the validation loss between actual and predicted values
            loss = criterion(preds.squeeze(), labels.float())

            total_loss += loss.item()

            acc = binary_accuracy(preds, labels)

            total_accuracy += acc.item()
            
            preds = preds.detach().cpu().numpy()
            
            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # compute the validation acc of the epoch
    avg_acc = total_accuracy / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, avg_acc, total_preds

# define the optimizer
optimizer = optim.Adam(model1.parameters(), lr=LEARNING_RATE)

# push to GPU
model = model1.to(device)

MODEL_WEIGHTS_PATH = 'Models/lstm_model_1_saved_weights.pt'

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
model1_train_losses = []
model1_valid_losses = []

# empty lists to store training and validation acc of each epoch
model1_train_accuracies = []
model1_valid_accuracies = []

# for each epoch
for epoch in range(EPOCHS):

    print('\n Epoch {:} / {:}'.format(epoch+1, EPOCHS))
    
    # train model
    train_loss, train_acc, _ = train()

    # evaluate model
    valid_loss, valid_acc, _ = evaluate()

    # save best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    
    # append training and validation loss
    model1_train_losses.append(train_loss)
    model1_valid_losses.append(valid_loss)

    # append training and validation acc
    model1_train_accuracies.append(train_acc)
    model1_valid_accuracies.append(valid_acc)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

    print(f'\nTraining Accuracy: {train_acc:.3f}')
    print(f'Validation Accuracy: {valid_acc:.3f}')

# load weights of best model lstm
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))

# create Tensor datasets
test_data = TensorDataset(torch.from_numpy(tokens_test), torch.from_numpy(test_labels.to_numpy()))

# Sampler for sampling the data
test_sampler = SequentialSampler(test_data)

# DataLoader
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

# empty list to save the model predictions
total_preds = []

# iterate over batches
for step, batch in enumerate(test_dataloader):

    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
        print('Batch {:>5,} of {:>5,}.'.format(step, len(test_dataloader)))

    # push the batch to gpu
    # batch = [t.to(device) for t in batch]

    inputs, labels = batch
    inputs = inputs.type(torch.LongTensor)

    # initialize hidden state
    test_h = model.init_hidden(len(inputs)) # BATCH_SIZE # to discuss!!!!!!!!!

    # move to gpu
    inputs, labels = inputs.cuda(), labels.cuda()

    # Create new variables for the hidden state
    test_h = tuple([each.data for each in test_h])

    # deactivate autograd
    with torch.no_grad():

        # model predictions
        preds, test_h = model(inputs, test_h)

        # convert output probabilities to predicted class (0 or 1)
        preds = torch.round(preds.squeeze())  # rounds to the nearest integer

        preds = preds.detach().cpu().numpy()

        total_preds.append(preds)

# reshape the predictions in form of (number of samples, no. of classes)
total_preds = np.concatenate(total_preds, axis=0)

print(classification_report(test_labels, total_preds, digits=4))

model_1_test_accuracy_score = accuracy_score(test_labels, total_preds)
model_1_test_precision_score = precision_score(test_labels, total_preds)
model_1_test_recall_score = recall_score(test_labels, total_preds)
model_1_test_f1_score = f1_score(test_labels, total_preds)

# define the optimizer
optimizer = torch.optim.Adam(model2.parameters(), lr=LEARNING_RATE)

# push to GPU
model = model2.to(device)

MODEL_WEIGHTS_PATH = 'Models/lstm_model_2_saved_weights.pt'

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
model2_train_losses = []
model2_valid_losses = []

# empty lists to store training and validation acc of each epoch
model2_train_accuracies = []
model2_valid_accuracies = []

# for each epoch
for epoch in range(EPOCHS):

    print('\n Epoch {:} / {:}'.format(epoch+1, EPOCHS))
    
    # train model
    train_loss, train_acc, _ = train()

    # evaluate model
    valid_loss, valid_acc, _ = evaluate()

    # save best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    
    # append training and validation loss
    model2_train_losses.append(train_loss)
    model2_valid_losses.append(valid_loss)

    # append training and validation acc
    model2_train_accuracies.append(train_acc)
    model2_valid_accuracies.append(valid_acc)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

    print(f'\nTraining Accuracy: {train_acc:.3f}')
    print(f'Validation Accuracy: {valid_acc:.3f}')

# load weights of best model lstm
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))

# create Tensor datasets
test_data = TensorDataset(torch.from_numpy(tokens_test), torch.from_numpy(test_labels.to_numpy()))

# Sampler for sampling the data
test_sampler = SequentialSampler(test_data)

# DataLoader
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

# empty list to save the model predictions
total_preds = []

# iterate over batches
for step, batch in enumerate(test_dataloader):

    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
        print('Batch {:>5,} of {:>5,}.'.format(step, len(test_dataloader)))

    # push the batch to gpu
    # batch = [t.to(device) for t in batch]

    inputs, labels = batch
    inputs = inputs.type(torch.LongTensor)

    # initialize hidden state
    test_h = model.init_hidden(len(inputs)) # BATCH_SIZE # to discuss!!!!!!!!!

    # move to gpu
    inputs, labels = inputs.cuda(), labels.cuda()

    # Create new variables for the hidden state
    test_h = tuple([each.data for each in test_h])

    # deactivate autograd
    with torch.no_grad():

        # model predictions
        preds, test_h = model(inputs, test_h)

        # convert output probabilities to predicted class (0 or 1)
        preds = torch.round(preds.squeeze())  # rounds to the nearest integer

        preds = preds.detach().cpu().numpy()

        total_preds.append(preds)

# reshape the predictions in form of (number of samples, no. of classes)
total_preds = np.concatenate(total_preds, axis=0)

print(classification_report(test_labels, total_preds, digits=4))

model_2_test_accuracy_score = accuracy_score(test_labels, total_preds)
model_2_test_precision_score = precision_score(test_labels, total_preds)
model_2_test_recall_score = recall_score(test_labels, total_preds)
model_2_test_f1_score = f1_score(test_labels, total_preds)

# define the optimizer
optimizer = torch.optim.Adam(model3.parameters(), lr=LEARNING_RATE)

# push to GPU
model = model3.to(device)

MODEL_WEIGHTS_PATH = 'Models/lstm_model_3_saved_weights.pt'

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
model3_train_losses = []
model3_valid_losses = []

# empty lists to store training and validation acc of each epoch
model3_train_accuracies = []
model3_valid_accuracies = []

# for each epoch
for epoch in range(EPOCHS):

    print('\n Epoch {:} / {:}'.format(epoch+1, EPOCHS))
    
    # train model
    train_loss, train_acc, _ = train()

    # evaluate model
    valid_loss, valid_acc, _ = evaluate()

    # save best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    
    # append training and validation loss
    model3_train_losses.append(train_loss)
    model3_valid_losses.append(valid_loss)

    # append training and validation acc
    model3_train_accuracies.append(train_acc)
    model3_valid_accuracies.append(valid_acc)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

    print(f'\nTraining Accuracy: {train_acc:.3f}')
    print(f'Validation Accuracy: {valid_acc:.3f}')

# load weights of best model lstm
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))

# create Tensor datasets
test_data = TensorDataset(torch.from_numpy(tokens_test), torch.from_numpy(test_labels.to_numpy()))

# Sampler for sampling the data
test_sampler = SequentialSampler(test_data)

# DataLoader
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

# empty list to save the model predictions
total_preds = []

# iterate over batches
for step, batch in enumerate(test_dataloader):

    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
        print('Batch {:>5,} of {:>5,}.'.format(step, len(test_dataloader)))

    # push the batch to gpu
    # batch = [t.to(device) for t in batch]

    inputs, labels = batch
    inputs = inputs.type(torch.LongTensor)

    # initialize hidden state
    test_h = model.init_hidden(len(inputs)) # BATCH_SIZE # to discuss!!!!!!!!!

    # move to gpu
    inputs, labels = inputs.cuda(), labels.cuda()

    # Create new variables for the hidden state
    test_h = tuple([each.data for each in test_h])

    # deactivate autograd
    with torch.no_grad():

        # model predictions
        preds, test_h = model(inputs, test_h)

        # convert output probabilities to predicted class (0 or 1)
        preds = torch.round(preds.squeeze())  # rounds to the nearest integer

        preds = preds.detach().cpu().numpy()

        total_preds.append(preds)

# reshape the predictions in form of (number of samples, no. of classes)
total_preds = np.concatenate(total_preds, axis=0)

print(classification_report(test_labels, total_preds, digits=4))

model_3_test_accuracy_score = accuracy_score(test_labels, total_preds)
model_3_test_precision_score = precision_score(test_labels, total_preds)
model_3_test_recall_score = recall_score(test_labels, total_preds)
model_3_test_f1_score = f1_score(test_labels, total_preds)

table = PrettyTable()
table.field_names = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

table.add_row(['LSTM without pre trained embedding weights', 
               format(model_1_test_accuracy_score, '.4f'), 
               format(model_1_test_precision_score, '.4f'), 
               format(model_1_test_recall_score, '.4f'), 
               format(model_1_test_f1_score, '.4f')])

table.add_row(['LSTM with Word2Vec pre trained embedding weights', 
               format(model_2_test_accuracy_score, '.4f'), 
               format(model_2_test_precision_score, '.4f'), 
               format(model_2_test_recall_score, '.4f'), 
               format(model_2_test_f1_score, '.4f')])

table.add_row(['LSTM with gloVe Twitter (200d) pre trained embedding weights', 
               format(model_3_test_accuracy_score, '.4f'), 
               format(model_3_test_precision_score, '.4f'), 
               format(model_3_test_recall_score, '.4f'), 
               format(model_3_test_f1_score, '.4f')])
print(table)

# summarize history for accuracy
plt.figure(figsize=(10,8))
plt.plot(model1_train_accuracies, "r-")
plt.plot(model1_valid_accuracies, "r--")
plt.plot(model2_train_accuracies, "b-")
plt.plot(model2_valid_accuracies, "b--")
plt.plot(model3_train_accuracies, "g-")
plt.plot(model3_valid_accuracies, "g--")
plt.title('Comparison of accuracies across LSTM models')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train (model 1)', 'validation (model 1)', 'train (model 2)', 'validation (model 2)', 'train (model 3)', 'validation (model 3)'], 
           bbox_to_anchor=(1, 1))
           #loc='upper left')
plt.show()



# Electra Model




# Specify GPU
device = torch.device("cuda")

# Define constants
EPOCHS = 1
BATCH_SIZE = 6
LEARNING_RATE = 1e-5
SEED = 4222

MODEL_SAVE_PATH = "Models/electra"
MODEL_CHECKPOINT_PATH = "Models/electra_checkpoint"
MODEL_LOGGING_PATH = "Models/electra_checkpoint/logs"

WANDB_ENTITY = "gohjiayi"
WANDB_PROJECT = "suicide_detection"
WANDB_RUN = "electra"

# Load dataset
df = pd.read_csv('Data/suicide_detection_final_cleaned.csv', header=0)
df.drop(columns=['cleaned_text'], inplace=True)
df['label'] = df['label'].map({'suicide': 1, 'non-suicide': 0})
df.head()

# Split dataset into train, validation and test sets
train, temp = train_test_split(df,
                               random_state=SEED,
                               test_size=0.2,
                               stratify=df['label'])

val, test = train_test_split(temp,
                             random_state=SEED,
                             test_size=0.5,
                             stratify=temp['label'])

# Load ELECTRA tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")

def dataset_conversion(train, test, val):
  """Converts pandas dataframe to Dataset."""

  train.reset_index(drop=True, inplace=True)
  test.reset_index(drop=True, inplace=True)
  val.reset_index(drop=True, inplace=True)

  train_dataset = Dataset.from_pandas(train)
  test_dataset = Dataset.from_pandas(test)
  val_dataset = Dataset.from_pandas(val)

  return DatasetDict({"train": train_dataset,
                      "test": test_dataset,
                      "val": val_dataset})

raw_datasets = dataset_conversion(train, test, val)

def tokenize_function(dataset):
    return tokenizer(dataset["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Tokenise datasets
SAMPLE_SIZE = 20
small_train_dataset = tokenized_datasets["train"].shuffle(seed=SEED).select(range(SAMPLE_SIZE))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=SEED).select(range(SAMPLE_SIZE))
small_val_dataset = tokenized_datasets["val"].shuffle(seed=SEED).select(range(SAMPLE_SIZE))

full_train_dataset = tokenized_datasets["train"]
full_test_dataset = tokenized_datasets["test"]
full_val_dataset = tokenized_datasets["val"]

# Import ELECTRA-base pretrained model
model = AutoModelForSequenceClassification.from_pretrained("google/electra-base-discriminator", num_labels=2)

# Login wandb
wandb.login()

# Initialise wandb
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN)

# Define custom metrics for computation
def compute_metrics(eval_pred):
    metric_acc = load_metric("accuracy")
    metric_rec = load_metric("recall")
    metric_pre = load_metric("precision")
    metric_f1 = load_metric("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    recall = metric_rec.compute(predictions=predictions, references=labels)["recall"]
    precision = metric_pre.compute(predictions=predictions, references=labels)["precision"]
    f1 = metric_f1.compute(predictions=predictions, references=labels)["f1"]

    return {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}

# Define model and training parameters
training_args = TrainingArguments(
    output_dir=MODEL_CHECKPOINT_PATH,
    overwrite_output_dir = True,
    report_to = 'wandb',
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    seed=SEED,
    # evaluation_strategy="epoch",
    run_name=WANDB_RUN,
    logging_dir=MODEL_LOGGING_PATH,
    save_strategy="steps",
    save_steps=1500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_train_dataset,
    eval_dataset=full_val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Predict before fine-tuning
trainer.predict(full_test_dataset).metrics

# %%wandb # To observe training progress live

# Fine-tune model
trainer.train()

# Resume fine-tuning from checkpoint
# trainer.train(MODEL_CHECKPOINT_PATH + "/" + "checkpoint-18000")

# Terminate wandb run
wandb.finish()

# Save fine-tuned model
trainer.save_model(MODEL_SAVE_PATH)

# Evaluate fine-tuned model
trainer.evaluate()

# Predict after fine-tuning
trainer.predict(full_test_dataset).metrics

def get_training_history(wandb_run):
  """Extract key metrics from training and eval across epochs from wandb run data."""

  # Get training history from wandb
  api = wandb.Api()
  run = api.run(wandb_run)
  history = run.history()

  # Rename columns
  train_column_dict = {'train/epoch': 'epoch', 'train/loss': 'training_loss'}
  val_column_dict = {'train/epoch': 'epoch', 'eval/loss': 'validation_loss', 'eval/accuracy': 'accuracy',
                'eval/precision': 'precision', 'eval/recall': 'recall', 'eval/f1': 'f1'}

  # Train data
  train_history = history[list(train_column_dict.keys())]
  train_history.columns = [train_column_dict.get(x, x) for x in train_history.columns]
  train_history = train_history.dropna()

  # Val data
  val_history = history[list(val_column_dict.keys())]
  val_history.columns = [val_column_dict.get(x, x) for x in val_history.columns]
  val_history = val_history.dropna()

  return pd.merge(train_history, val_history, how="right", on="epoch")


# Get dataframe for training history
WANDB_RUN_ID = "1bcfrimx" # Replace with your wandb run details, found in the training cell

training_history = get_training_history(WANDB_ENTITY + "/" + WANDB_PROJECT + "/" + WANDB_RUN_ID)
training_history

# Load fine-tuned model
saved_model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

# Load trainer after fine-tune
saved_trainer = Trainer(
    model=saved_model,
    args=training_args,
    train_dataset=full_train_dataset,
    eval_dataset=full_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Predict after fine-tuning
saved_trainer.predict(full_test_dataset).metrics

# Load fine-tuned model
saved_model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

# Load trainer after fine-tune
saved_trainer = Trainer(
    model=saved_model,
    args=training_args,
    train_dataset=full_train_dataset,
    eval_dataset=full_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)



# Delete variables and empty cache
del trainer
del model
torch.cuda.empty_cache()

# Python garbage collection
import gc
gc.collect()

# Check memory allocation
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())

# Check memory summary
print(torch.cuda.memory_summary(device=None, abbreviated=False))

# Check GPU allocation and acprocesses
!nvidia-smi



# Bi-LSTM Model

class BiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, dropout):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(self.dropout(pooled))


# Main execution logic
if __name__ == "__main__":
    print("Script for LSTM, Electra, and Bi-LSTM Models")
