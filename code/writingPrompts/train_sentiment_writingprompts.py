import os

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
import numpy as np
import copy


# Enable gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load token to index
token_to_index = torch.load("token_to_index_sentiment_writingprompts.pth", weights_only=False)

# Define some parameters
n_vocab = len(token_to_index)
embedding_size = 50  # GloVe embedding size
condition_size = 1  # condition is just one float
batch_size = 7  # 8 too much for hardware

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


# A dataloader needs a collate function to pad sequences
def collate_batch(batch):
    X_batch, y_batch, length_batch = zip(*batch)

    X_batch = nn.utils.rnn.pad_sequence(X_batch, batch_first=True, padding_value=2.0)
    y_batch = nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=-1)
    length_batch = torch.tensor(length_batch)

    return X_batch, y_batch, length_batch


# Define the model
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
        packed_output, _ = self.lstm(x, (h, c))
        # unpack output
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # produce logits
        logits = self.linear(self.dropout(output))
        return logits


model = SentimentWritingPromptsLSTM()
model.to(device)
path = "sentiment_writingprompts.pth"


clip_value = 1.0
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create the scheduler to reduce LR when the validation loss plateaus
smaller_step_size_after = 3
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=smaller_step_size_after-1,
                              factor=0.1, threshold=1, threshold_mode='abs')

loss_fn = nn.CrossEntropyLoss(reduction="sum")
best_model = None
best_loss = np.inf

epoch = 0
epochs_no_improvement = 0
epochs_no_improvement_step_size = 0
cooldown = 10  # Try at least 10 epochs with the initial learning rate

# Only run this when resuming training
checkpoint = torch.load(path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epoch = checkpoint['epoch']
epochs_no_improvement = checkpoint['epochs_no_improvement']
epochs_no_improvement_step_size = checkpoint['epochs_no_improvement_step_size']
cooldown = checkpoint['cooldown']
best_loss = checkpoint['best_loss']
best_model = checkpoint['best_model']
optimizer_best_model = checkpoint['optimizer_best_model']
# Only run above when resuming training

print("Start training", flush=True)
while epochs_no_improvement < 10:

    model.train()
    for i in range(900):
        current_path = "folds/sentiment_writingprompts_fold" + str(i) + ".pth"
        current_fold = torch.load(current_path)
        fold_loader = data.DataLoader(current_fold, shuffle=True, batch_size=batch_size, pin_memory=True,
                                  collate_fn=collate_batch, num_workers=8)

        for X_batch, y_batch, length_batch in fold_loader:
            h = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
            c = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)

            # Pack X_batch so it can be used as input for the LSTM
            X_batch = nn.utils.rnn.pack_padded_sequence(X_batch, length_batch, batch_first=True, enforce_sorted=False)
            y_pred = model(X_batch.to(device), h, c)

            # Make a mask so padding doesn't influence the loss function
            mask = y_batch != -1

            loss = loss_fn(y_pred[mask], y_batch[mask].to(device))

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()


    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for i in range(900, 1000):
            current_path = "folds/sentiment_writingprompts_fold" + str(i) + ".pth"
            current_fold = torch.load(current_path)
            fold_loader = data.DataLoader(current_fold, shuffle=True, batch_size=batch_size, pin_memory=True,
                                          collate_fn=collate_batch, num_workers=8)

            for X_batch, y_batch, length_batch in fold_loader:
                h = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                c = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)

                # Pack X_batch so it can be used as input for the LSTM
                X_batch = nn.utils.rnn.pack_padded_sequence(X_batch, length_batch, batch_first=True, enforce_sorted=False)
                y_pred = model(X_batch.to(device), h, c)

                # Make a mask so padding doesn't influence the loss function
                mask = y_batch != -1

                loss += loss_fn(y_pred[mask], y_batch[mask].to(device))


        print("Epoch %d: Cross Entropy Loss: %.4f" % (epoch, loss), flush=True)

        # Update learning rate
        if cooldown <= 0:
            scheduler.step(loss)

        if loss < best_loss - 1:
            best_loss = loss
            epochs_no_improvement = 0
            epochs_no_improvement_step_size = 0
            best_model = copy.deepcopy(model.state_dict())
            optimizer_best_model = copy.deepcopy(optimizer.state_dict())

        else:
            if cooldown <= 0:
                epochs_no_improvement += 1
                epochs_no_improvement_step_size += 1

            # Original model could be overtrained, go back to the previous best model/optimizer
            if epochs_no_improvement_step_size >= smaller_step_size_after:
                print("No more improvement, revert to previous model with lower learning rate", flush=True)
                model.load_state_dict(best_model)
                optimizer.load_state_dict(optimizer_best_model)
                epochs_no_improvement_step_size = 0
                cooldown = 3  # wiil be 2 after this, ensure you try at least 5 epochs with each learning rate

    epoch += 1  # This needs to happen here, otherwise the current epoch will be used for training twice
    cooldown -= 1
    torch.save({
        'epoch': epoch,
        'epochs_no_improvement': epochs_no_improvement,
        'epochs_no_improvement_step_size': epochs_no_improvement_step_size,
        'cooldown': cooldown,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'best_model': best_model,
        'optimizer_best_model': optimizer_best_model
    }, path)

torch.save(best_model, path)
print("-------------Done----------------")
