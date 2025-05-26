# This code is inspired by https://machinelearningmastery.com/text-generation-with-lstm-in-pytorch/

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
import numpy as np
import csv
import math

sys.path.append('../')
from helper_functions import prepare_input_text, get_sentiment_scores, top_k_top_p_filtering

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
class StateReviewLSTM(nn.Module):
    def __init__(self, input_size=embedding_size+condition_size,
                 hidden_size=512, num_layers=3):

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        super(StateReviewLSTM, self).__init__()
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


model = StateReviewLSTM()
model.load_state_dict(best_model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

sentiment_pipeline = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                              top_k=None,
                              device=device)

# Add a prompt below
prompt = "a girl goes on a walk in a magical forest."
#prompt = "Last Sentence: I saw my reflection blink at me."
#prompt = "Years ago a machine that records dreams was invented. Dreams have become the primary form of entertainment. Particularly talented dreamers have become stars. Tell me about tonight's big ``Oscar'' night"
#prompt = "Winter Lights"
#prompt = "You're a thief who breaks into homes, but try your best to stay undetected. You lubricate the hinges to prevent squeaky noises, you sweep the floor to get rid of footsteps, etc. Eventually, you fix more than you take, and rumors spread about a mysterious, helpful fairy in town."
#prompt = "A group of criminals discuss the urban legend of the Bat Man who supposedly stalks the night in Gotham city."
#prompt = "Love is a drug, literally."
#prompt = "You wake up in a strange room, only to find alternate universe versions of you there, each different in their own way (gender, race, background etc). You have no idea what brought you here."
#prompt = "The first true AI has been created by accident; it is malicious and wants world domination. Problem is that it is housed in a streetlamp."
#prompt = "You're the protagonist and for some reason you have multiple narrating voices covering your story. They are starting to dight over who gets to narrate you."
#prompt = "A teenager decides to skip class and explore the school, he finds something when he enters the basement level. What is it?"

# for perplexity data
with (open('perplexity_lstm.csv', 'w', newline='', encoding='utf-8-sig') as perplexity_csv,
      open('verhalen_lstm.csv', 'w', newline='', encoding='utf-8-sig') as verhalen_csv):
    perplexity_writer = csv.writer(perplexity_csv)
    perplexity_writer.writerow(['perplexity'])

    verhalen_writer = csv.writer(verhalen_csv)
    verhalen_writer.writerow(['verhaal'])

    for story_number in range(100): # multiple times for multiple data points
        story = ""
        csv_name = "probabilities_lstm_" + str(story_number) + ".csv"
        with open(csv_name, 'w', newline='', encoding='utf-8-sig') as prob_csv:
            prob_writer = csv.writer(prob_csv)
            prob_writer.writerow(['probability'])

            # Prepare inputs
            tokens = prepare_input_text(prompt, token_to_index)
            score_prompt = [get_sentiment_scores(prompt, sentiment_pipeline)[0] for i in range(len(tokens))]

            condition = score_prompt[-1]

            probabilities = []

            # Add <BOS> and the condition
            tokens.append('<BOS>')
            score_prompt.append(condition)

            pattern = [np.append(glove_dict[tokens[i]], score_prompt[i]) for i in range(len(tokens))]
            h = torch.zeros(model.num_layers, model.hidden_size).to(device)
            c = torch.zeros(model.num_layers, model.hidden_size).to(device)

            model.eval()
            with torch.no_grad():
                i = 0
                end_of_seq = False
                while i < 500 and not end_of_seq:
                    if i > 400 and (result == "." or result == "!"):
                        end_of_seq = True
                    else:
                        tensor = torch.tensor(np.array(pattern)).to(torch.float32)

                        # Generate probabilities as output from the model
                        prediction, (h, c) = model(tensor.to(device), h, c)

                        probs = F.softmax(prediction, -1)

                        # Make sure <UNK> and <BOS> are not generated
                        prediction[:, token_to_index['<UNK>']] = -np.inf
                        prediction[:, token_to_index['<BOS>']] = -np.inf

                        # Also ensure the proces doesn't stop too early
                        if i < 100:
                            prediction[:, token_to_index['<EOS>']] = -np.inf

                        index = top_k_top_p_filtering(prediction, k, p)
                        current_prob = probs[-1][index].item()
                        probabilities.append(current_prob)

                        # Write into csv
                        prob_writer.writerow([current_prob])

                        result = index_to_token[index]
                        story = story + result + " "

                        # Prepare the chosen character as input in the next iteration
                        pattern = np.array([np.append(glove_dict[result], condition)])

                        if result == '<EOS>':
                            end_of_seq = True
                    i += 1

        perplexity = math.pow(2, -sum(np.log2(probabilities)) / len(probabilities))

        # Write into csv
        perplexity_writer.writerow([perplexity])
        verhalen_writer.writerow([story])

print("-------------Done----------------")
