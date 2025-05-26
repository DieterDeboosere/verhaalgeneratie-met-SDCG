# with https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

import numpy as np
import torch


glove_dict = {}
with open('glove.6B/glove.6B.50d.txt', 'rb') as file:
    for l in file:
        line = l.decode().split()
        word = line[0]
        vector = np.array(line[1:]).astype(float)
        glove_dict[word] = vector

# Add an unknown token. OOV words will be mapped to this.
glove_dict['<UNK>'] = np.zeros(50, dtype=float)

# Add <BOS> and <EOS> tokens to mark beginning and end of sequences
glove_dict['<BOS>'] = np.random.uniform(-1, 1, size=50)
glove_dict['<EOS>'] = np.random.uniform(-1, 1, size=50)

# Add <PAD> token for padding
glove_dict['<PAD>'] = np.random.uniform(-1, 1, size=50)

torch.save(glove_dict, 'glove_dict.pth')
