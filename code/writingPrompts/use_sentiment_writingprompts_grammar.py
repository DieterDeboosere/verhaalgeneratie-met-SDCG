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
    #prompt = "Mother always said, kill them with kindness."
    #prompt = "You are the only person on earth who bothers to read the Terms and Conditions. In Apple, Microsoft, and Facebook's T&C, you find a clause called ``Universal Private Defense Force and Mandatory Draft Protocols''. Shortly after posting your findings on Reddit, you hear a knock on your door."
    #prompt = "As he points to the vault, he says ``Sounds human, doesn't it? Don't be fooled. It's very much not.''"
    #prompt = "As you go about your normal morning routine a note falls out of the bathroom cabinet. It reads: ``If you don't remember, it happened again.''"
    #prompt = "In a world where people can only see in black and white, you are a drug dealer that sells drugs that allow people to see color.
    #prompt = "A supervillain monologue worth not killing the hero for."
    #prompt = "The world must convince the chosen one that he has to save the world. The only problem is he REALLY doesn't want to"
    #prompt = "You are the world, and humanity has decimated you in a nuclear holocaust. You are now left alone with only your thoughts."
    #prompt = "Extra-terrestrial beings have launched an assault on Earth. You have been chosen as a cultural representative to showcase humanity's achievements, in a final effort to convince the beings that humans are existentially valuable."
    #prompt = "The Avengers are playing a board game when things go horribly, hilariously wrong."

    prompt2 = "a ninja wants to become a baker."

    # Import the grammar
    #janus.consult('story_spine')
    janus.consult('aba')

    # for perplexity data
    with (open('perplexity_aba.csv', 'w', newline='', encoding='utf-8-sig') as perplexity_csv,
          open('verhalen_aba.csv', 'w', newline='', encoding='utf-8-sig') as verhalen_csv):
        perplexity_writer = csv.writer(perplexity_csv)
        perplexity_writer.writerow(['perplexity'])

        verhalen_writer = csv.writer(verhalen_csv)
        verhalen_writer.writerow(['verhaal'])

        for story_number in range(100):  # multiple times for multiple data points

            csv_name = "probabilities_aba_" + str(story_number) + ".csv"
            with open(csv_name, 'w', newline='', encoding='utf-8-sig') as prob_csv:
                prob_writer = csv.writer(prob_csv)
                prob_writer.writerow(['probability'])


                # Prepare inputs
                tokens = prepare_input_text(prompt, token_to_index)
                tokens.append("<BOS>")
                score_prompt = [0 for _ in range(len(tokens))]

                tokens2 = prepare_input_text(prompt2, token_to_index)
                tokens2.append("<BOS>")
                score_prompt2 = [0 for _ in range(len(tokens2))]

                pattern = [np.append(glove_dict[tokens[i]], score_prompt[i]) for i in range(len(tokens))]
                pattern2 = [np.append(glove_dict[tokens2[i]], score_prompt2[i]) for i in range(len(tokens2))]


                h = torch.zeros(model.num_layers, model.hidden_size)
                c = torch.zeros(model.num_layers, model.hidden_size)

                with torch.no_grad():
                    tensor = torch.tensor(np.array(pattern)).to(torch.float32)
                    tensor2 = torch.tensor(np.array(pattern2)).to(torch.float32)

                    # Generate hidden and cell state based on prompt
                    _, (h1, c1) = model(tensor.to(device), h.to(device), c.to(device))
                    _, (h2, c2) = model(tensor2.to(device), h.to(device), c.to(device))

                #story = janus.query_once("phrase(start_story(H0, C0, H6, C6, Probs), Generated)", {'H0':h1, 'C0':c1})
                story = janus.query_once("phrase(start_aba(HA0, CA0, HB0, CB0, HA2, CA2, HB1, CB1, Probs), Generated)", {'HA0':h1, 'CA0':c1, 'HB0':h2, 'CB0':c2})

                # Write into cs
                verhalen_writer.writerow([" ".join(story['Generated'])])

                for prob in story['Probs']:
                    prob_writer.writerow([prob])

                perplexity = math.pow(2, -sum(np.log2(story['Probs'])) / len(story['Probs']))
                perplexity_writer.writerow([perplexity])

                print(" ".join(story['Generated']))
                perplexity = math.pow(2, -sum(np.log2(story['Probs'])) / len(story['Probs']))
                print(perplexity)


def generate(h, c, text, cond):

    # Prepare inputs
    xin = " ".join(text)
    tokens = prepare_input_text(xin, token_to_index)
    pattern = [np.append(glove_dict[tokens[i]], cond) for i in range(len(tokens))]
    pattern = np.append(glove_dict['<BOS>'], cond) + pattern
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

                # Make sure <UNK> and <BOS> are not generated
                prediction[:, token_to_index['<UNK>']] = -np.inf
                prediction[:, token_to_index['<BOS>']] = -np.inf

                # Also ensure the proces doesn't stop too early for any specific part of the story spine
                if i < 10:
                    prediction[:, token_to_index['<EOS>']] = -np.inf

                index = top_k_top_p_filtering(prediction, k, p)
                current_prob = probs[-1][index].item()
                probabilities.append(current_prob)

                result = index_to_token[index]

                # Prepare the chosen character as input in the next iteration
                pattern = [np.append(glove_dict[result], cond)]
                if result == '<EOS>':
                    end_of_seq = True

                else:
                    text.append(result)

            i += 1

    text.append("END_OF_PART")
    return h, c, text, probabilities

if __name__ == '__main__':
    main()
