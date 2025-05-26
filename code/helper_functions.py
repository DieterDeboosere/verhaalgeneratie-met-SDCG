from nltk import word_tokenize
import re
import torch
import torch.nn.functional as F

def prepare_input_text(raw_text, dictionary):
    raw_text = raw_text.lower()

    # split raw text into word tokens
    tokens = word_tokenize(raw_text)

    # split words like "pear-tree."
    token_idx = 0
    split_pattern = r'([a-z]+|[-.])'
    while token_idx < len(tokens):
        if tokens[token_idx] in dictionary:
            token_idx += 1
        else:
            long_token = tokens.pop(token_idx)
            split = re.findall(split_pattern, long_token)
            for new_token in split:
                tokens.insert(token_idx, new_token)
                token_idx += 1

    # Replace OOV words with <UNK>
    for token_idx in range(len(tokens)):
        if tokens[token_idx] not in dictionary.keys():
            tokens[token_idx] = '<UNK>'

    return tokens

def split_sentences_with_pattern(sentences, pattern, min_occurence=1):
    sentence_idx = 0
    while sentence_idx < len(sentences):
        if sentences[sentence_idx].count(pattern) > min_occurence:
            long_sentence = sentences.pop(sentence_idx)
            new_sentences = long_sentence.split(pattern)

            for new_sentence in new_sentences:
                if new_sentence != new_sentences[-1]:
                    new_sentence += pattern
                sentences.insert(sentence_idx, new_sentence)
                sentence_idx += 1
        else:
            sentence_idx += 1

    return sentences


def get_sentiment_scores(text_list, sentiment_pipeline):
    sentiments = sentiment_pipeline(text_list)
    sentiment_scores = []

    for sentiment in sentiments:
        score = 0
        for label in sentiment:
            if label["label"] == "positive":
                score += label["score"]
            elif label["label"] == "negative":
                score -= label["score"]
        sentiment_scores.append(score)

    return sentiment_scores


# The following function is based on https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep at least top k tokens with highest probability.
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

        Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    if top_k > 0:
        # If the number of tokens after top-p filtering is less than top_k, retain at least top_k tokens
        sorted_indices_to_remove[..., :top_k] = 0

    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

    pred_token = torch.multinomial(F.softmax(logits, -1), 1)  # [BATCH_SIZE, 1]
    return pred_token[0][-1].item()
