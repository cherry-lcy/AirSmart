from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import numpy as np
import torch
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from AirSmart.HVAC_Temperature_control import hvac_basic
from AirSmart.HVAC_Temperature_control.ac_model import AC_output
from AirSmart.weather_forecast import bilstm_model

path = ""
place = ""

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# mock user data
users = {
    'admin01': '123456'
}

@app.route('/')
def home():
    if 'username' in session:
        return render_template('monitor.html')
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        print(username, password)
        
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('monitor'))
        else:
            return "Invalid username or password"
    return render_template('login.html')

@app.route('/monitor', methods=['GET', 'POST'])
def monitor():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        building = request.form.get('building')
        print(building)

        # uncomment this part if you have access to the HKT Limited sensor data
        # this part demonstrate how we suggest a AC temperature based on the data received from the IOT sensors
        """
        # load trained model from depository
        lstm_model = bilstm_model.LSTMModel()

        # load best scored
        best_model_weights_path = os.path.join(path, 'best_model_weights.pth')
        lstm_model.load_state_dict(torch.load(best_model_weights_path))

        # evaluation
        lstm_model.eval()

        # load other parameters
        params_path = os.path.join(path, 'training_params.pth')
        try:
            params = torch.load(params_path, weights_only=False)
        except Exception as e:
            print(f"Error loading params: {e}")
            return "Error loading model parameters", 500

        best_score = params['best_score']
        best_train_preds = params['best_train_preds']
        best_test_preds = params['best_test_preds']
        epochs = params['epochs']

        # process test data
        future_steps = 10
        test_ = pd.read_csv(path+"TEST_"+place+"_DATASET_.csv")
        test_ = test_.values.astype("float32")
        x_test = test_[-future_steps:,1:]
        scaler = MinMaxScaler()
        x_test = scaler.fit_transform(x_test)

        # make prediction
        pred_temp = bilstm_model.predict_future(lstm_model, x_test, future_steps)

        # undo scaling
        predicted_temps = scaler.inverse_transform(np.array(pred_temp))
        print(f"Temperature predictions of coming {future_steps} days:", predicted_temps[:,-1])

        hkt02 = pd.read_excel("C:\\test\\dataset_hkt\\hkt\\a002-1f-sr-tp01_timeseries.xlsx")
        hkt02temps = np.array(hkt02["temperature"].astype(float))
        hkt02temps = hkt02temps[5000:5010]
        hkt02temps = hkt02temps[:,None]
        ptemp = predicted_temps[:,-1]
        ptemp = ptemp[:,None]
        predicted_temps = np.concatenate((hkt02temps, ptemp), axis=1)

        # predict action of Air Conditioner
        ACMPC = AC_output(model_path=path+"ac_model_test_hkt.pth")
        energy = hvac_basic.energy_consume(ACMPC.predict(predicted_temps), predicted_temps[-1][0], predicted_temps[-1][0])
        new_indoor = hvac_basic.indoor_temp_change(ACMPC.predict(predicted_temps), predicted_temps[-1][0], predicted_temps[-1][0])
        re_ac_temp = ACMPC.predict(predicted_temps)

        print(f"Current Indoor Temperature: {predicted_temps[-1][0]:.3f}°C\tOutdoor Temperature: {predicted_temps[-1][1]:.3f}°C")
        print(f"Recommend AC Temperature: {re_ac_temp}°C")
        print(f"New Indoor Temperature: {new_indoor:.3f}°C")
        print(f"Energy Consumption: {energy:.3f}kWh")

        # clean cache
        app.jinja_env.cache.clear()
        del lstm_model
        del ACMPC
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # send data to the server
        return jsonify({"currIn":predicted_temps[-1][0], "currOut":round(predicted_temps[-1][1],3),"newIndoor":round(new_indoor,3),"ACtemp":re_ac_temp,"energy":round(energy,3)}), 200
        """
        # return the data produced by the above codes
        return jsonify({"currIn":22.800, "currOut":21.827,"newIndoor":22.803,"ACtemp":23,"energy":4.650}), 200
    else:
        return render_template('monitor.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)