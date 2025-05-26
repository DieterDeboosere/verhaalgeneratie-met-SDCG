# This code is inspired by https://machinelearningmastery.com/text-generation-with-lstm-in-pytorch/

import sys
import janus_swi as janus
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
import numpy as np
import math
import csv

sys.path.append('../')
from helper_functions import prepare_input_text, top_k_top_p_filtering

best_model = torch.load("sentiment_writingprompts.pth", weights_only=False)
best_model = best_model['best_model']
glove_dict = torch.load("../glove_dict.pth", weights_only=False)
token_to_index = torch.load("token_to_index_sentiment_writingprompts.pth", weights_only=False)
n_vocab = len(token_to_index)
index_to_token = dict((i, c) for c, i in token_to_index.items())
embedding_size = 50
condition_size = 1
batch_size = 1

# Variables for decoding
temperature = 0.8  # the higher this is, the smoother the probability distribution will be
p = 0.9  # tokens with at least this cumulative probability are considered
k = 5  # minimum amount of tokens you want


# Reload the model
class SentimentWritingPromptsLSTM(nn.Module):
    def __init__(self, input_size=embedding_size+condition_size,
                 hidden_size=512, num_layers=3):

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        super(SentimentWritingPromptsLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(hidden_size, n_vocab)

    def forward(self, x, h, c):
        output, (h, c) = self.lstm(x, (h, c))
        # produce logits
        logits = self.linear(self.dropout(output))
        # divide by temperature
        logits = logits / temperature
        return logits, (h, c)


model = SentimentWritingPromptsLSTM()
model.load_state_dict(best_model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

sentiment_pipeline = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                              top_k=None,
                              device=device)

def main():

    # Add a prompt below
    prompt = "a girl goes on a walk in a magical forest."
    #prompt = "Someone took your stapler"
    #prompt = "You are sent over 1000 years into the past by accident. You must now learn to survive using the primitive technology of the year 2016..."
    #prompt = "On Purge Night, part-time employees are paid 50 times the hourly rate. It's that night of the year, and you're a pizza delivery worker looking to make some big money."
    #prompt = "Genres of music are embodied as deities. One genre is actually a cult that's planning an uprising."
    #prompt = "One day everyone notices the words ``Human Update 1.1 progress 1%'' in the corner of their eye."
    #prompt = "``No matter what you hear, no matter how badly you want to, do NOT open your eyes.''"
    #prompt = "A child was selected to ask the president a question on live television. Historians would cite the innocent question as the cause of the collapse of civilization."
    #prompt = "Every University has a graveyard for it's students' dead 4.0's. Each tombstone is different, and has a cause of death and date."
    #prompt = "All bullets suddenly leave trails, red for kill, blue for hit without kill, gray for miss. One very large red one is found at the bottom of Mariana 's trench."
    #prompt = "You're sitting in a therapist's room after years and years of being the Grim Reaper. You've got a lot to get off your chest."

    # Import the grammar
    #janus.consult('story_spine_only_condition')
    janus.consult('pos_to_neg')
    #janus.consult('neg_to_pos')

    # for perplexity data
    #with (open('perplexity_cond.csv', 'w', newline='', encoding='utf-8-sig') as perplexity_csv,
    #      open('verhalen_cond.csv', 'w', newline='', encoding='utf-8-sig') as verhalen_csv):
    #    perplexity_writer = csv.writer(perplexity_csv)
    #    perplexity_writer.writerow(['perplexity'])

    #    verhalen_writer = csv.writer(verhalen_csv)
    #    verhalen_writer.writerow(['verhaal'])

    for story_number in range(1):  # multiple times for multiple data points
        """
        csv_name = "probabilities_cond_" + str(story_number) + ".csv"
        with open(csv_name, 'w', newline='', encoding='utf-8-sig') as prob_csv:
            prob_writer = csv.writer(prob_csv)
            prob_writer.writerow(['probability'])
        """
        probabilities = []

        # Prepare inputs
        tokens = prepare_input_text(prompt, token_to_index)
        tokens.append("<BOS>")
        score_prompt = [0 for _ in range(len(tokens))]

        pattern = [np.append(glove_dict[tokens[i]], score_prompt[i]) for i in range(len(tokens))]
        h = torch.zeros(model.num_layers, model.hidden_size)
        c = torch.zeros(model.num_layers, model.hidden_size)

        with torch.no_grad():
            tensor = torch.tensor(np.array(pattern)).to(torch.float32)

            # Generate prediction, hidden and cell state based on prompt
            prediction, (h, c) = model(tensor.to(device), h.to(device), c.to(device))
            probs = F.softmax(prediction, -1)

            # Make sure <UNK>, <BOS> and <EOS> are not generated
            prediction[:, token_to_index['<UNK>']] = -np.inf
            prediction[:, token_to_index['<BOS>']] = -np.inf
            prediction[:, token_to_index['<EOS>']] = -np.inf

            index = top_k_top_p_filtering(prediction, k, p)
            current_prob = probs[-1][index].item()
            probabilities.append(current_prob)

            # Write into csv
            #prob_writer.writerow([current_prob])

            x = index_to_token[index]

        #story = janus.query_once("phrase(start_story(H0, C0, X0, H6, C6, X6, Probs), Generated)", {'H0':h, 'C0':c, 'X0':x})
        story = janus.query_once("phrase(start_pos_to_neg(H0, C0, X0, H5, C5, X5, Probs), Generated)", {'H0':h, 'C0':c, 'X0':x})
        #story = janus.query_once("phrase(start_neg_to_pos(H0, C0, X0, H5, C5, X5, Probs), Generated)", {'H0':h, 'C0':c, 'X0':x})

        """
        # Write into cs
        verhalen_writer.writerow([" ".join(story['Generated'])])

        for prob in story['Probs']:
            prob_writer.writerow([prob])
            probabilities.append(prob)

        perplexity = math.pow(2, -sum(np.log2(probabilities)) / len(probabilities))
        perplexity_writer.writerow([perplexity])
        """
        print(" ".join(story['Generated']))
        perplexity = math.pow(2, -sum(np.log2(story['Probs'])) / len(story['Probs']))
        print(perplexity)


def generate(h, c, x, cond):

    # Prepare inputs
    text = []
    pattern = [np.append(glove_dict[x], cond)]
    result = ""

    probabilities = []

    # append what grammar says to pattern
    with (((torch.no_grad()))):
        i = 0
        end_of_seq = False
        while i < 70 and not end_of_seq:
            if i > 35 and (result == "."  or result == "!" or result == "?"):
                end_of_seq = True
            else:
                tensor = torch.tensor(np.array(pattern)).to(torch.float32)

                # Generate probabilities as output from the model
                prediction, (h, c) = model(tensor.to(device), h.to(device), c.to(device))
                probs = F.softmax(prediction, -1)

                # Make sure <UNK>, <BOS> and <EOS> are not generated
                prediction[:, token_to_index['<UNK>']] = -np.inf
                prediction[:, token_to_index['<BOS>']] = -np.inf
                prediction[:, token_to_index['<EOS>']] = -np.inf

                index = top_k_top_p_filtering(prediction, k, p)
                current_prob = probs[-1][index].item()
                probabilities.append(current_prob)

                result = index_to_token[index]

                # Prepare the chosen character as input in the next iteration
                pattern = [np.append(glove_dict[result], cond)]
                text.append(result)

            i += 1

    text.append("END_OF_PART")
    return h, c, result, text, probabilities

if __name__ == '__main__':
    main()
