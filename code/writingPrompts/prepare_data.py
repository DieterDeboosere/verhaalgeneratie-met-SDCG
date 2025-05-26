import os

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import torch
import pandas as pd
import ast
import torch.utils.data as data
import numpy as np

sys.path.append('../')
from helper_functions import prepare_input_text


# load glove dictionary
glove_dict = torch.load("../glove_dict.pth", weights_only=False)

# Get scored data
scored_data = pd.read_csv("writingPrompts_scored.csv", on_bad_lines='warn')
all_prompt_sentences = [ast.literal_eval(prompt) for prompt in scored_data.prompt]
all_prompt_scores = [ast.literal_eval(prompt_scores) for prompt_scores in scored_data.score_per_prompt]
all_story_sentences = [ast.literal_eval(story) for story in scored_data.story]
all_story_scores = [ast.literal_eval(story_scores) for story_scores in scored_data.score_per_story]

token_counts = {"<BOS>": 0, "<EOS>": 0}
all_tokens_per_prompt_and_story = []
all_scores_per_prompt_and_story = []

for i in range(len(all_prompt_sentences)):

    # Check if the story isn't too long. If it is, skip it to ensure everything fits in memory
    if len(all_story_sentences[i]) <= 200:
        current_prompt_and_story_tokens = []
        current_prompt_and_story_scores = []

        for j in range(len(all_prompt_sentences[i])):
            tokenized_sentence = prepare_input_text(all_prompt_sentences[i][j], glove_dict)
            for token in tokenized_sentence:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts.update({token: 1})
            current_prompt_and_story_tokens += tokenized_sentence
            current_prompt_and_story_scores += [all_prompt_scores[i][j] for x in range(len(tokenized_sentence))]

        # Add <BOS> token
        current_prompt_and_story_tokens.append('<BOS>')
        token_counts["<BOS>"] += 1

        # current_review_scores[i] holds sentiment of current_review_tokens[i-1]
        # Remove the score from the very first token and add score of <BOS>
        current_prompt_and_story_scores = current_prompt_and_story_scores[1:] + [0.252]

        for j in range(len(all_story_sentences[i])):
            tokenized_sentence = prepare_input_text(all_story_sentences[i][j], glove_dict)
            for token in tokenized_sentence:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts.update({token: 1})
            current_prompt_and_story_tokens += tokenized_sentence
            current_prompt_and_story_scores += [all_story_scores[i][j] for x in range(len(tokenized_sentence))]

        # Add <EOS> token and score
        current_prompt_and_story_tokens.append('<EOS>')
        token_counts["<EOS>"] += 1
        current_prompt_and_story_scores.append(0.277)

        all_tokens_per_prompt_and_story.append(current_prompt_and_story_tokens)
        all_scores_per_prompt_and_story.append(current_prompt_and_story_scores)

# Remove unnecessary variables to prevent memory overload
del scored_data, all_prompt_sentences, all_prompt_scores, all_story_sentences, all_story_scores

# Remove all tokens that appear too little
different_tokens = {key for key in token_counts if token_counts[key] >= 25}

# Make a dictionary that maps every token to a unique index
token_to_index = {token: idx for idx, token in enumerate(different_tokens)}

# Save this dict
torch.save(token_to_index, "token_to_index_sentiment_writingprompts.pth")

# Map uncommon tokens to <UNK>
for story in all_tokens_per_prompt_and_story:
    for index in range(len(story)):
        if story[index] not in different_tokens:
            story[index] = "<UNK>"

# Make a custom dataset so we can handle batching later
class SentimentWritingPromptsDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.lengths = [len(review) for review in X]  # Store lengths for later use

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


print("Start making embeddings", flush=True)
fold_size = len(all_tokens_per_prompt_and_story) // 1000
for i in range(1000):
    if i == 999:
        tokens_per_prompt_and_story_fold = all_tokens_per_prompt_and_story[fold_size * i:]
        scores_per_prompt_and_story_fold = all_scores_per_prompt_and_story[fold_size * i:]
    else:
        tokens_per_prompt_and_story_fold = all_tokens_per_prompt_and_story[fold_size*i:fold_size*(i+1)]
        scores_per_prompt_and_story_fold = all_scores_per_prompt_and_story[fold_size*i:fold_size*(i+1)]

    # Make word embeddings and add the condition as an element to the vectors
    foldX = []
    foldY = []
    for j in range(len(tokens_per_prompt_and_story_fold)):
        text = tokens_per_prompt_and_story_fold[j]
        scores = scores_per_prompt_and_story_fold[j]
        tokens_per_prompt_and_story_fold[j] = []
        scores_per_prompt_and_story_fold[j] = []
        currentX = []
        currentY = []
        for k in range(len(text) - 1):  # <EOS> shouldn't ever be in X
            embedding = glove_dict[text[k]]
            currentX.append(np.append(embedding, scores[k]))
            currentY.append(token_to_index[text[k+1]])

        # Turn data into tensors of type float 32 as the LSTM requires this
        foldX.append(torch.tensor(currentX).to(torch.float32))
        foldY.append(torch.tensor(currentY))

    path = "folds/sentiment_writingprompts_fold" + str(i) + ".pth"
    torch.save(SentimentWritingPromptsDataset(foldX, foldY), path)
    print("Saved chunk " + str(i), flush=True)

print("Done!", flush=True)
