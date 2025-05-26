import sys
import re
from nltk import sent_tokenize
from transformers import pipeline
import torch
import csv

sys.path.append('../')
from helper_functions import split_sentences_with_pattern, get_sentiment_scores


# Enable gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sentiment_pipeline = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                              top_k=None,
                              device=device)

# load ascii text
prompts = open('source.txt', 'r', encoding='utf-8').readlines()
stories = open('target.txt', 'r', encoding='utf-8').readlines()

# Output file for dataset with scores
csv_out_path = "writingPrompts_scored.csv"

with open(csv_out_path, 'w', newline='', encoding='utf-8-sig') as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(['prompt', 'score_per_prompt', 'story', 'score_per_story'])

    for i in range(len(prompts)):
        prompt = prompts[i]
        story = stories[i]

        # Split into sentences
        sentences_prompt = sent_tokenize(prompt)
        sentences_story = sent_tokenize(story)

        if len(sentences_prompt) == 1:
            prompt = re.sub(r'(?<=\.)((?=\S))(?!(\.\.\.))', r' ', prompt)
            sentences_prompt = sent_tokenize(prompt)

        if len(sentences_story) == 1:
            story = re.sub(r'(?<=\.)((?=\S))(?!(\.\.\.))', r' ', story)
            sentences_story = sent_tokenize(story)

        # Also split sentences with specific patterns in them because these are sometimes too long
        sentences_prompt = split_sentences_with_pattern(sentences_prompt, '...', 0)
        sentences_prompt = split_sentences_with_pattern(sentences_prompt, ';', 0)
        sentences_prompt = split_sentences_with_pattern(sentences_prompt, '')
        sentences_prompt = split_sentences_with_pattern(sentences_prompt, '<newline>')

        sentences_story = split_sentences_with_pattern(sentences_story, '...', 0)
        sentences_story = split_sentences_with_pattern(sentences_story, ';', 0)
        sentences_story = split_sentences_with_pattern(sentences_story, '')
        sentences_story = split_sentences_with_pattern(sentences_story, '<newline>')

        # More than 4 commas => also split here
        sentences_prompt = split_sentences_with_pattern(sentences_prompt, ',', 4)
        sentences_story = split_sentences_with_pattern(sentences_story, ',', 4)

        sentences_short_enough = True

        try:
            sentiment_scores_sentences_prompt = get_sentiment_scores(sentences_prompt, sentiment_pipeline)
            sentiment_scores_sentences_story = get_sentiment_scores(sentences_story, sentiment_pipeline)
        except:
            sentences_short_enough = False
        finally:
            if sentences_short_enough:
                # Write into csv
                writer.writerow([str(sentences_prompt), str(sentiment_scores_sentences_prompt),
                                str(sentences_story), str(sentiment_scores_sentences_story)])
