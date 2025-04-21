from weather_forecast.bilstm_model import LSTMModel, predict_future
import os
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from HVAC_Temperature_control.ac_model import AC_output
from HVAC_Temperature_control.hvac_basic import mock_indoor_temp, indoor_temp_change, energy_consume

# variable list
place = "HKA"
path = "" # path you store the data

if __name__ == "__main__":
    # load trained model from depository
    lstm_model = LSTMModel() # from cell 33

    # load best scored
    best_model_weights_path = os.path.join(path, 'best_model_weights.pth')
    lstm_model.load_state_dict(torch.load(best_model_weights_path))

    # evaluation
    lstm_model.eval()

    # process test data
    future_steps = 10
    test_ = pd.read_csv(path+"TEST_"+place+"_DATASET_.csv")  # from cell 25
    test_ = test_.values.astype("float32")
    x_test = test_[-future_steps:,1:]
    y_test = x_test[:,-1]
    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(x_test)

    # make prediction
    pred_temp = predict_future(lstm_model, x_test, future_steps)  # from cell 40

    # undo scaling
    predicted_temps = scaler.inverse_transform(np.array(pred_temp))
    print(f"Temperature predictions of coming {future_steps} days:", predicted_temps[:,-1])

    # since indoor temperature data is not provided, we cannot train the BiLSTM model to
    # predict future indoor temperature, the required indoor temperature for Air Conditioner
    # temperature setting is generated randomly
    mock_ind_temp = mock_indoor_temp(predicted_temps[:,-1])
    pred_temps = np.concatenate((mock_ind_temp[:,None], predicted_temps[:,-1][:,None]), axis=1)

    # predict action of Air Conditioner
    ACMPC = AC_output(model_path=path+"ac_model_test.pth")
    print(f"Current Indoor Temperature: {pred_temps[0][0]:.3f}°C\tOutdoor Temperature: {pred_temps[0][1]:.3f}°C")
    print(f"Recommend AC Temperature: {ACMPC.predict(pred_temps)}°C")
    print(f"New Indoor Temperature: {indoor_temp_change(ACMPC.predict(pred_temps), pred_temps[0][0], pred_temps[0][0]):.3f}°C")
    print(f"Energy Consumption: {energy_consume(ACMPC.predict(pred_temps), pred_temps[0][0], pred_temps[0][0]):.3f}kWh")
