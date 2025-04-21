import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from bilstm_model import predict_future, LSTMModel, EarlyStopping
import numpy as np
import os

# variable list
place = "HKA"
path = ""  # path you store the data
lookback = 7

# load trained model from depository
model = LSTMModel()

# load best scored
best_model_weights_path = os.path.join(path, 'best_model_weights.pth')
model.load_state_dict(torch.load(best_model_weights_path))
# load the whole model
# best_model_path = os.path.join(path, 'best_model.pth')
# model = torch.load(best_model_path)

# evaluation
model.eval()

# load other parameters
params_path = os.path.join(path, 'training_params.pth')
params = torch.load(params_path)
best_score = params['best_score']
best_train_preds = params['best_train_preds']
best_test_preds = params['best_test_preds']
epochs = params['epochs']

# data processing for prediction
train_ = pd.read_csv(path+"TRAIN_"+place+"_DATASET_.csv")
test_ = pd.read_csv(path+"TEST_"+place+"_DATASET_.csv")
train_ = train_.values.astype("float32")
test_ = test_.values.astype("float32")
x_train = train_[-(lookback):,1:]
y_train = x_train[:,-1]
x_test = test_[-(lookback*2):-lookback,1:]
y_test = x_test[:,-1]
scaler = MinMaxScaler()
x_test = scaler.fit_transform(x_test)

# make prediction
future_steps = 7
predictions = predict_future(model, x_test, future_steps)

# undo scaling
predicted_temps = scaler.inverse_transform(np.array(predictions))
print(f"Temperature predictions of coming {future_steps} days:", predicted_temps[:,-1])