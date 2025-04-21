import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

# create dataset for training and testing
def create_dataset(data, lookback):
    x = []
    y = []
    for i in range(len(data)-lookback):
        feature = data[i:i+lookback,:]
        target = data[i+lookback, :]
        x.append(feature)
        y.append(target)
    return torch.tensor(x), torch.tensor(y)

# define LSTM class
class LSTMModel(nn.Module):
    def __init__(self, input_size = 9, hidden_size = 256, num_layers = 2, output_size = 9):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional= True, dropout=0.3)

        # dropout layer
        # fully connected layer
        # and output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # save output from lstm layer (output, hn, cn)
        output, (_,_) = self.lstm(x)

        # shape of output = [batch_size, seq_len / time_step, hidden_size]
        out = self.fc(output[:,-1,:])
        return out
    
# define EarlyStopping class
class EarlyStopping:
    def __init__(self, patience = 20, delta = 0, verbose = False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
        self.best_y_pred = None

    def __call__(self, val_loss, model, X):
        score = -val_loss

        # if the func is called for the first time, save score, model state, predicted result
        # else, count 1 cycle until the no of cycle reach patience, stop training
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
            with torch.no_grad():
                self.best_y_pred = model(X)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # if verbose == true, print score
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}, score: {self.best_score}')

            # set early_stop condition
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # record better score, model state and predicted result
            self.best_score = score
            self.best_state = model.state_dict()
            with torch.no_grad():
                self.best_y_pred = model(X)
            # reset counter
            self.counter = 0

# multi-variate time-series prediction
def predict_future(model, initial_sequence, num_steps):
    model.eval()
    current_sequence = initial_sequence.copy()
    predictions = []

    # generate prediction for the next num_steps
    for _ in range(num_steps):
        input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor).numpy()[0]

        # create new predicted data
        for i in range(1, num_steps):
            current_sequence[i-1,:] = current_sequence[i,:]
        current_sequence[num_steps-1, :] = output
        new_step = output
        predictions.append(new_step)

    return np.array(predictions)

# plot graph for prediction result
def plot_prediction(previous_data, predictions, num_steps, title):
    time_steps = [i for i in range(-previous_data.shape[0],0)]

    plt.figure(dpi=128,figsize=(10,6))
    plt.title(title,fontsize=20)
    L1, = plt.plot(time_steps, previous_data[:,-1],label = "History Temperature")
    L2, = plt.plot([i for i in range(num_steps)], predictions,marker="o", markersize=10,label="Predicted Temperature")
    plt.legend(handles=[L1, L2],labels=['History Temperature', 'Predicted Temperature'], loc='best')
    plt.ylabel('Temperature', fontsize=10)
    plt.xlabel('Time-steps', fontsize=10)
    plt.tight_layout()
    plt.tick_params(axis='both',which='major',labelsize=10)
    return plt