import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from bilstm_model import create_dataset, LSTMModel, EarlyStopping
import numpy as np
import os

# variable list
place = "HKA"
path = ""  # path you store the data
lookback = 7

# load training data and testing data
train = pd.read_csv(path+"TRAIN_"+place+"_DATASET_.csv")
test = pd.read_csv(path+"TEST_"+place+"_DATASET_.csv")

# convert dataframe to numpy array
train = train.values.astype("float32")
test = test.values.astype("float32")

# preprocessing -- scaling using MinMax
scaler = MinMaxScaler()
train = scaler.fit_transform(train) # curve fitting and transform to a specific range(default: 0-1)
test = scaler.transform(test) # transform to a specific range(default: 0-1)

# convert data into feature and target set
train_x, train_y = create_dataset(train, lookback)
test_x, test_y = create_dataset(test, lookback)

# load the training data with DataLoader and shuffle the data
train_loader = DataLoader(TensorDataset(train_x, train_y),batch_size = 32, shuffle = True)

# call model
model = LSTMModel()

# optimizer: Adam
optimizer = optim.Adam(model.parameters(), lr = 2e-4, weight_decay = 1e-5)

# loss func: huberloss
loss_fn = nn.HuberLoss(delta=0.5)

# scheduler (tune learning rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# define early stopping condition
Early_stop = EarlyStopping(patience = 15, verbose = True)

# save best score, weightings, training set prediction and testing set prediction
best_score = None
best_weights = None
best_train_preds = None
best_test_preds = None

# set epochs
epochs = 100
tr_rmse = []
te_rmse = []

for e in range(epochs):
    # training
    model.train()

    for batch_x, batch_y in train_loader:
        pred_y = model(batch_x)
        # print(pred_y.shape, batch_x.shape, batch_y.shape)
        loss = loss_fn(pred_y, batch_y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # evaluation
    model.eval()

    with torch.no_grad():
        # save train error and result
        pred_y = model(train_x)
        # print(pred_y.shape, train_y.shape)
        train_rmse = np.sqrt(loss_fn(pred_y, train_y).item())
        tr_rmse.append(train_rmse)
        train_preds = pred_y.clone().detach().cpu().numpy()

        # save test error and result
        pred_y = model(test_x)
        # print(pred_y.shape, test_y.shape)
        test_rmse = np.sqrt(loss_fn(pred_y, test_y).item())
        te_rmse.append(test_rmse)
        test_preds = pred_y.clone().detach().cpu().numpy()

        # update scheduler
        scheduler.step(test_rmse)

        # save better score, weights, train prediction and test prediction
        if best_score is None or test_rmse < best_score:
            best_score = test_rmse
            best_weights = model.state_dict()
            best_train_preds = train_preds
            best_test_pred = test_preds

        # call EarlyStopping func
        Early_stop(test_rmse, model, test_x)

        # check if early stopping condition reached
        # if so, stop training
        if Early_stop.early_stop:
            print("Early Stopping")
            break

        # print error every ecpoch
        print('Epoch: ', e+1, '\t train RMSE: ', train_rmse, '\t test RMSE', test_rmse, '\t Loss:', loss.item())

# save model, weights, prediction result
if best_weights is not None:
    model.load_state_dict(best_weights)

    # save best weight
    best_model_weights_path = os.path.join(path, 'best_model_weights.pth')
    torch.save(best_weights, best_model_weights_path)

    # save model
    best_model_path = os.path.join(path, 'best_model.pth')
    torch.save(model.state_dict(), best_model_path)

    # save other parameters
    params = {
    'best_score': best_score,
    'best_train_preds': best_train_preds,
    'best_test_preds': best_test_preds,
    'epochs': epochs
    }
    params_path = os.path.join(path, 'training_params.pth')
    torch.save(params, params_path)

    with torch.no_grad():
        pred_y_train = model(train_x).clone().detach().cpu().numpy()
        pred_y_test = model(test_x).clone().detach().cpu().numpy()