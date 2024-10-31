#!/usr/bin/env python
import os
import string
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from collections import deque
from IPython.display import display
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import re

import rospy
from grounding.srv import CommandTasks, CommandTasksResponse

# Fix random seed for reproducibility
seed = 0
torch.manual_seed(0)
np.random.seed(0)

def load_embeddings(file_name, max_words=50000):

    """
    Read word vectors and create conversion dictionaries from a word embeddings file
    Optionally, you can change the number of maximum words to use
    """

    # Initialize the dictionaries with the unknown token
    word_to_ix = {"<UNK>": 0}
    ix_to_word = {0: "<UNK>"}
    vectors = [[]]

    # Go through all the words, get their embedding vectors, and add to the dictionaries

    with open(file_name,"r") as f:
        counter = 0
        for line in f.readlines():
            items = line.split()
            if counter < max_words:
                counter += 1
                word_to_ix[items[0]] = counter
                ix_to_word[counter] = items[0]
                vectors.append([float(x) for x in items[1:]])

        # Randomly set the weights of the first element mapping to "UNKNOWN"
        vector_len = len(vectors[-1])
        vectors[0] = [np.random.random() * 2.0 - 1.0
                        for _ in range(vector_len)]

    return np.array(vectors), word_to_ix, ix_to_word

def prepare_input_sequence(sentence, word_to_ix):

    """ Uses the word to index dictionary to create a sequence of vocabulary indices """

    # Convert sentence to lower case, remove punctuation, and split
    sentence = sentence.translate(str.maketrans("","", string.punctuation))
    seq = sentence.lower().split(" ")

    # Now convert to a word index
    indices = []
    for w in seq:
        try:
            idx = word_to_ix[w]
        except:
            idx = 0 # UNK token
        indices.append(idx)
    return indices

def prepare_target_sequence(seq, action_to_ix, room_to_ix, obj_to_ix):

    """
    Uses the output grounding target dictionaries to create a sequence of output indices
    """
    action_idxs = [action_to_ix[seq[0]]]
    room_idxs = [room_to_ix[seq[1]]]
    object_idxs = [obj_to_ix[seq[2]]]
    return action_idxs, room_idxs, object_idxs

glove_file = os.path.join("/home/mustar/test_ws/src/grounding/src", "glove.6B/glove.6B.300d.txt") # Modify this as needed
glove_vectors, w2i, i2w = load_embeddings(glove_file)

class GroundingDataset(Dataset):

    """ Language grounding dataset """


    def __init__(self, filename, word_to_ix, action_to_ix, room_to_ix, object_to_ix, transform=None):
        self.data = pd.read_table(filename, sep=",")
        self.word_to_ix = word_to_ix
        self.action_to_ix = action_to_ix
        self.room_to_ix = room_to_ix
        self.object_to_ix = object_to_ix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data.iloc[idx].to_dict()
        inp_seq = item["Sentence"]
        inp = prepare_input_sequence(inp_seq, self.word_to_ix)
        tgt_seq = (item["Action"], item["Room"], item["Object"])
        tgt = prepare_target_sequence(tgt_seq,
            self.action_to_ix, self.room_to_ix, self.object_to_ix)
        return torch.LongTensor(inp), torch.LongTensor(tgt)

    def get_sentence(self, idx):
        return self.data["Sentence"][idx]

    def print_dataset(self):
        # display(self.data) # If not in a Jupyter notebook, use print(self.data)
        print(self.data)
        print("")

def create_grounding_dict(grounding_list):

    """
    Creates a grounding dictionary from a list. For example:
        ["Find,"Go","Get","Store"]
    becomes
        {"Find":0, "Get":1, "Go":2, "Store":3}
    """

    idx = 0
    grounding_dict = {}
    for elem in grounding_list:
        grounding_dict[elem] = idx
        idx += 1

    return grounding_dict

def create_target_dictionaries(training_file):

    """
    Create dictionaries for all the outputs (targets) of the grounding network
    """

    # Read the training data and get sorted list of all the groundings
    data = pd.read_table(training_file,sep=",")
    action_list = sorted(set(data.Action))
    room_list = sorted(set(data.Room))
    object_list = sorted(set(data.Object))

    # Convert to dictionary that is compatible with the grounding model
    action_to_ix = create_grounding_dict(action_list)
    room_to_ix = create_grounding_dict(room_list)
    object_to_ix = create_grounding_dict(object_list)

    return action_to_ix, room_to_ix, object_to_ix


# # Create the training and test datasets
training_file = os.path.join("/home/mustar/test_ws/src/grounding/src", "rnn_training_data.txt")
test_file = os.path.join("/home/mustar/test_ws/src/grounding/src", "rnn_test_data.txt")
act2i, room2i, obj2i = create_target_dictionaries(training_file)
training_data = GroundingDataset(training_file, w2i, act2i, room2i, obj2i)
test_data = GroundingDataset(test_file, w2i, act2i, room2i, obj2i)


# Print the training dataset, which internally stores data as a Pandas table
# training_data.print_dataset()

# Get a random item from the training data and view its PyTorch-compatible representation
idx = np.random.randint(0, training_data.__len__())
# print(training_data.get_sentence(idx))

item = training_data.__getitem__(idx)
# print("\nWord indices:\n{}".format(item[0]))
# print("\nTarget indices:\n{}".format(item[1]))


class GroundingNetwork(nn.Module):

    """Neural network for language grounding"""

    def __init__(self, action_size, room_size, object_size,
                 unit_type="rnn", hidden_dim=32, num_layers=2,
                 emb_weights=glove_vectors, bidir=False, dropout=0.25):

        super().__init__()

        # Set dimension of hidden layer depending on LSTM directionality
        if bidir:
            self.hidden_dim = hidden_dim * 2
        else:
            self.hidden_dim = hidden_dim

        # Initialize word embeddings from the pretrained vectors and freeze the weights
        self.word_embeddings = nn.Embedding.from_pretrained(torch.Tensor(emb_weights), freeze=True)
        (_, embedding_dim) = emb_weights.shape

        # The RNN layers take word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.unit_type = unit_type
        if self.unit_type == "rnn":
            self.recurrent = nn.RNN(embedding_dim, hidden_dim,
                                num_layers=num_layers, bidirectional=bidir,
                                nonlinearity="tanh", dropout=dropout,
                                batch_first=True)
        elif self.unit_type == "gru":
            self.recurrent = nn.GRU(embedding_dim, hidden_dim,
                                num_layers=num_layers, bidirectional=bidir,
                                dropout=dropout, batch_first=True)
        elif self.unit_type == "lstm":
            self.recurrent = nn.LSTM(embedding_dim, hidden_dim,
                                num_layers=num_layers, bidirectional=bidir,
                                dropout=dropout, batch_first=True)

        # The linear layers that map from hidden state space to grounding space
        self.hidden2action = nn.Linear(self.hidden_dim, action_size)
        self.hidden2room = nn.Linear(self.hidden_dim, room_size)
        self.hidden2object = nn.Linear(self.hidden_dim, object_size)

    def forward(self, inp):

        # Embed the sentence
        embeds = self.word_embeddings(inp)

        # Pass through the LSTM Module
        rnn_out, _ = self.recurrent(embeds)

        # Grab the final RNN output for each batch element
        rnn_final = rnn_out[:,-1,:]

        # Final Fully Connected Networks
        action_scores = func.softmax(self.hidden2action(rnn_final), dim=1)
        room_scores = func.softmax(self.hidden2room(rnn_final), dim=1)
        object_scores = func.softmax(self.hidden2object(rnn_final), dim=1)

        return action_scores, room_scores, object_scores



################################################Added Section#########################################


def load_grounding_model(filepath):
  """
  Loads a grounding network model from a specified file path.

  Args:
      filepath (str): Path to the saved model file.

  Returns:
      GroundedNetwork: Loaded grounding network model.
  """
  # Create an instance of the GroundingNetwork class
  model = GroundingNetwork(action_size=len(act2i), room_size=len(room2i), object_size=len(obj2i),
                             hidden_dim=32, num_layers=2, unit_type="lstm", bidir=True, dropout=0.25)

  # Load the saved weights from the specified file path
  model.load_state_dict(torch.load(filepath))

  # Set the model to evaluation mode (important for models with dropout)
  model.eval()
  return model


model_path = '/home/mustar/test_ws/src/grounding/src/grounding_model.pth'
loaded_net = load_grounding_model(model_path)  # Path to your downloaded model
rospy.loginfo("model loaded")


def filter_description(description):
    # Define regular expressions for types, colors, and locations
    type_pattern = re.compile(r'(apple|banana|orange|pear|tuna|water|soda|chips|cookies|crackers|tuna|corn|tomato|vegetables|peas|carrots)')
    color_pattern = re.compile(r'(red|yellow|green|blue|grey|white|black)')
    location_pattern = re.compile(r'(left of|right of|front of|back|on)')

    # Find all matches for each pattern in the description
    types = [(match.group(), match.start()) for match in type_pattern.finditer(description)]
    colors = [(match.group(), match.start()) for match in color_pattern.finditer(description)]
    locations = [(match.group(), match.start()) for match in location_pattern.finditer(description)]

    # Sort the matches based on their starting positions
    types.sort(key=lambda x: x[1])
    colors.sort(key=lambda x: x[1])
    locations.sort(key=lambda x: x[1])

    # Construct the filtered description while maintaining the original order
    filtered_description = ""
    i = 0

    while(i < len(types)):
      if i < len(colors):
        filtered_description += colors[i][0] + " "
      if i < len(types):
        filtered_description += types[i][0] + " "
      if i < len(locations):
        filtered_description += locations[i][0] + " "
      i += 1

    return filtered_description.strip()

def separate_sentences(text):
    # Define the separators
    separators = ["then", "and", "after that", ","]

    # Initialize a list to store the separated sentences
    separated_sentences = []

    # Split the text using each separator and flatten the list
    for separator in separators:
        text = text.replace(separator, '\n')
    separated_sentences = [sentence.strip() for sentence in text.split('\n') if sentence.strip()]

    return separated_sentences

def get_tasks(request):
    output = ''
    text = request.command
    filtered_description = filter_description(text)


    rospy.loginfo("Prompt Text: {}".format(text))
    rospy.loginfo("Filtered Description: {}".format(filtered_description))

    sentences = separate_sentences(text)

    for query_sentence in sentences:
        # print("Query Prediction: {}".format(query_sentence))
        query_input = prepare_input_sequence(query_sentence, w2i)
        input_tensor = torch.LongTensor(query_input)

        with torch.no_grad():
            # Assuming `input_tensor` is a LongTensor containing your input sequence
            action_scores, room_scores, object_scores = loaded_net(input_tensor.view(1, -1))
            # print(action_scores, room_scores, object_scores)

        # Get predictions and indices for the query sentence
        query_act_ind = action_scores.argmax(dim=1).item()
        query_room_ind = room_scores.argmax(dim=1).item()
        query_obj_ind = object_scores.argmax(dim=1).item()

        # Convert indices to words using inverse dictionaries
        query_predicted_output = [
            list(act2i.keys())[list(act2i.values()).index(query_act_ind)],
            list(room2i.keys())[list(room2i.values()).index(query_room_ind)],
            list(obj2i.keys())[list(obj2i.values()).index(query_obj_ind)]
        ]

        if query_predicted_output[1] == 'living room':
            query_predicted_output[1] = 'livingroom'

        # Print the Predicted Output
        rospy.loginfo("Query Prediction: {}".format(query_predicted_output))
        output += ' '.join(query_predicted_output) + '.'

    output += ' Description ' + filtered_description

    return CommandTasksResponse(output)


if __name__=="__main__":

    rospy.init_node('grounding_server')
    rospy.loginfo('NLU Server Initiated')

    try:
        service = rospy.Service('command_tasks', CommandTasks, get_tasks)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass
