import torch
import torchtext
import spacy
from torchtext.data import get_tokenizer
from torch.utils.data import random_split
from torchtext.experimental.datasets import IMDB
from torch.utils.data import DataLoader
from models import MyTransformer
from tqdm import tqdm
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np


def pad_trim(data):
    ''' Pads or trims the batch of input data.

    Arguments:
        data (torch.Tensor): input batch
    Returns:
        new_input (torch.Tensor): padded/trimmed input
        labels (torch.Tensor): batch of output target labels
    '''
    data = list(zip(*data))
    # Extract target output labels
    labels = torch.tensor(data[0]).float().to(device)
    # Extract input data
    inputs = data[1]

    # Extract only the part of the input up to the MAX_SEQ_LEN point
    # if input sample contains more than MAX_SEQ_LEN. If not then
    # select entire sample and append <pad_id> until the length of the
    # sequence is MAX_SEQ_LEN
    new_input = torch.stack([torch.cat((input[:MAX_SEQ_LEN],
                                        torch.tensor([pad_id] * max(0, MAX_SEQ_LEN - len(input))).long()))
                             for input in inputs])

    return new_input, labels

def split_train_val(train_set):
    ''' Splits the given set into train and validation sets WRT split ratio
    Arguments:
        train_set: set to split
    Returns:
        train_set: train dataset
        valid_set: validation dataset
    '''
    train_num = int(SPLIT_RATIO * len(train_set))
    valid_num = len(train_set) - train_num
    generator = torch.Generator().manual_seed(SEED)
    train_set, valid_set = random_split(train_set, lengths=[train_num, valid_num],
                                        generator=generator)
    return train_set, valid_set

def load_imdb_data(batch_size = 32):
    """
    This function loads the IMDB dataset and creates train, validation and test sets.
    It should take around 15-20 minutes to run on the first time (it downloads the GloVe embeddings, IMDB dataset and extracts the vocab).
    Don't worry, it will be fast on the next runs. It is recommended to run this function before you start implementing the training logic.
    :return: train_set, valid_set, test_set, train_loader, valid_loader, test_loader, vocab, pad_id
    """
    cwd = os.getcwd()
    if not os.path.exists(cwd + '/.vector_cache'):
        os.makedirs(cwd + '/.vector_cache')
    if not os.path.exists(cwd + '/.data'):
        os.makedirs(cwd + '/.data')
    # Extract the initial vocab from the IMDB dataset
    vocab = IMDB(data_select='train')[0].get_vocab()
    # Create GloVe embeddings based on original vocab word frequencies
    glove_vocab = torchtext.vocab.Vocab(counter=vocab.freqs,
                                        max_size=MAX_VOCAB_SIZE,
                                        min_freq=MIN_FREQ,
                                        vectors=torchtext.vocab.GloVe(name='6B'))
    # Acquire 'Spacy' tokenizer for the vocab words
    tokenizer = get_tokenizer('spacy', 'en_core_web_sm')
    # Acquire train and test IMDB sets with previously created GloVe vocab and 'Spacy' tokenizer
    train_set, test_set = IMDB(tokenizer=tokenizer, vocab=glove_vocab)
    vocab = train_set.get_vocab()  # Extract the vocab of the acquired train set
    pad_id = vocab['<pad>']  # Extract the token used for padding

    train_set, valid_set = split_train_val(train_set)  # Split the train set into train and validation sets

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=pad_trim)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, collate_fn=pad_trim)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=pad_trim)

    return train_set, valid_set, test_set, train_loader, valid_loader, test_loader, vocab, pad_id


def evaluat_model(model , data_loader, criterion ,desc_kind , ephot_num):
    """
    This function should evaluate the model using the given data loader and the criterion.
    It should return the average loss
    :param model: The model to evaluate
    :param data_loader: The data loader to use
    :param criterion: The loss criterion
    :return: The average loss

    """

    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()

        # Initialize the total loss 
        total_loss = 0
        
        # initialize the number of correct predictions , and total number of predictions
        total_correct = 0
        total_predictions = 0
        
        # Iterate over the batches
        for inputs_embeddings, labels in tqdm(data_loader, desc=desc_kind + ephot_num , total=len(data_loader)):
            # Extract the inputs and the labels
            inputs_embeddings, labels = inputs_embeddings.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs_embeddings)

            # compute the number of correct predictions
            
            # get the predicted labels by the model  
            predicted = (torch.sigmoid(outputs) > 0.5).int()
            
            # compute the number of correct predictions
            total_correct += (predicted == labels.unsqueeze(1)).sum().item()

            # add the total number of predictions
            total_predictions += labels.size(0)

            # Compute the loss
            loss = criterion(outputs.squeeze(), labels.float())

            # Add the loss to the total loss
            total_loss += loss.item() * inputs_embeddings.size(0)

    # Compute the accuracy
    accuracy = total_correct / total_predictions

    # Compute the average loss
    avg_loss = total_loss / len(data_loader.dataset)

    return avg_loss , accuracy
 
 

def train_transformer(model, train_loader, valid_loader,test_loader ,optimizer, criterion, num_of_epochs):
    """
    This function should train the transformer model using the given train and validation loaders and the optimizer and criterion.
    It should also Plot the training and validation loss for each epoch. Additionally, report the test accuracy at the end of training.
    :param model: The transformer model
    :param train_loader: The train data loader
    :param valid_loader: The validation data loader
    :param optimizer: The optimizer
    :param criterion: The loss criterion
    :param num_of_epochs: The number of epochs to train
    :return: None
    """
    # initialize the lists for the loss for train , validation and test
    train_loss_lst = []
    valid_loss_lst = []
    test_loss_lst = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_of_epochs):

        # Set the model to training mode
        model.train()

        # Initialize the total loss 
        total_loss = 0


        # Iterate over the training batches
        for inputs_embeddings, labels in tqdm(train_loader, desc='Train' + str(epoch+1 ), total=len(train_loader)):
            
            # Extract the inputs and the labels
            inputs_embeddings, labels = inputs_embeddings.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs_embeddings)

            # Compute the loss
            loss = criterion(outputs.squeeze(), labels.float())

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Add the loss to the total loss
            total_loss += loss.item() * inputs_embeddings.size(0)

        # Compute the average loss and append the loss to the list
        train_loss = total_loss / len(train_loader.dataset)
        train_loss_lst.append(train_loss)


        # Evaluate the model on the validation set and append the loss to the list
        valid_loss  , _ = evaluat_model(model , valid_loader , criterion , 'Validation' , str(epoch + 1))
        valid_loss_lst.append(valid_loss)

        # Evaluate the model on the test set and append the loss to the list
        test_loss, _  = evaluat_model(model , test_loader , criterion , 'Test' , str(epoch + 1))
        test_loss_lst.append(test_loss)

        # Print the losses
        print(f'Epoch {epoch + 1}/{num_of_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Test Loss: {test_loss:.4f}')

    return train_loss_lst , valid_loss_lst , test_loss_lst
    

def plot(train_loss_lst , valid_loss_lst , test_loss_lst) -> None:
    """
    This function should plot the training and validation loss for each epoch
    :param train_loss_lst: The list of training losses
    :param valid_loss_lst: The list of validation losses
    :return: None
    """
    Epochs = range(1 , 6)
    # Plot the training , validation and test loss
    plt.plot(Epochs , train_loss_lst, label='Train Loss' , color = 'red')
    plt.plot(Epochs , valid_loss_lst, label='Validation Loss' , color = 'blue')
    plt.plot(Epochs , test_loss_lst, label='Test Loss' , color = 'green')
    plt.title('Training , Validation and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def find_a_configuration_with_90_acc( train_loader, valid_loader, test_loader ):
    """
    This function should find a configuration that achieves at least 90% accuracy on the test set.
    :return: None
    """

    # Define the range of hyperparameters to search

    # num_of_blocks_values = [1, 2, 3]
    # num_of_epochs_values = [5, 10, 15]
    # learning_rates = [0.0001, 0.0005, 0.001]
    num_of_blocks_values = [4]
    num_of_epochs_values = [20]
    learning_rates = [0.0001]

    # Iterate over the hyperparameters
    for num_of_blocks in num_of_blocks_values:
        for num_of_epochs in num_of_epochs_values:
            for learning_rate in learning_rates:


                # Create the model and optimizer and criterion
                model = MyTransformer(vocab=vocab, max_len=MAX_SEQ_LEN, num_of_blocks=num_of_blocks).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                criterion = torch.nn.BCEWithLogitsLoss()

                # Train the Transformer
                train_loss_lst , valid_loss_lst , test_loss_lst = train_transformer(model, train_loader, valid_loader, test_loader , optimizer, criterion, num_of_epochs)

                # Evaluate the model on the test set
                test_loss , test_accuracy = evaluat_model(model , test_loader , criterion , 'Test' , str(num_of_epochs))

                # check if the accuracy is better than 90%
                if test_accuracy >= 0.9:
                    print(f'Configuration: batch_size={batch_size}, num_of_blocks={num_of_blocks}, num_of_epochs={num_of_epochs}, learning_rate={learning_rate} achieved an accuracy of {test_accuracy:.4f}')
                    return
                else:
                    print(f'Configuration: batch_size={batch_size}, num_of_blocks={num_of_blocks}, num_of_epochs={num_of_epochs}, learning_rate={learning_rate} achieved an accuracy of {test_accuracy:.4f}')




print('Loading the IMDB dataset...')
# Set the device    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VOCAB AND DATASET HYPERPARAMETERS, DO NOT CHANGE
MAX_VOCAB_SIZE = 25000 # Maximum number of words in the vocabulary
MIN_FREQ = 10 # We include only words which occur in the corpus with some minimal frequency
MAX_SEQ_LEN = 500 # We trim/pad each sentence to this number of words
SPLIT_RATIO = 0.8 # Split ratio between train and validation set
SEED = 0




# OUR HYPERPARAMETERS
# set the hyperparameters
batch_size = 32

# a single transformer block
num_of_blocks = 1 

# 5 epochs
num_of_epochs = 5

# using 0.0001 s the learning rate
learning_rate = 0.0001

# set the seed
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load the IMDB dataset
train_set, valid_set, test_set, train_loader, valid_loader, test_loader, vocab, pad_id = load_imdb_data(batch_size)

# Create the model and optimizer and criterion
model = MyTransformer(vocab=vocab, max_len=MAX_SEQ_LEN, num_of_blocks=num_of_blocks).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()

# Train the Transformer
train_loss_lst , valid_loss_lst , test_loss_lst = train_transformer(model, train_loader, valid_loader,test_loader ,optimizer, criterion, num_of_epochs)

# # Plot the training and validation loss
plot(train_loss_lst , valid_loss_lst , test_loss_lst)

# Evaluate the model on the test set
test_loss , test_accuracy = evaluat_model(model , test_loader , criterion , 'Test' , str(num_of_epochs))
print(f'Test Accuracy: {test_accuracy:.4f}')

# Find a configuration that achieves at least 90% accuracy on the test set
find_a_configuration_with_90_acc( train_loader, valid_loader, test_loader )

