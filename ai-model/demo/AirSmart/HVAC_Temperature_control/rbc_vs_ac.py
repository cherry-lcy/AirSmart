from datetime import datetime, timedelta
import numpy as np
from rbc import AC_Temp_Control
from hvac_basic import indoor_temp_change, energy_consume
from ac_model import AC_output

# variable list
place = "HKA"
path = "" # path you store the data

# create mock sensor data
historical_data = []
for i in range(10):
    day = datetime(2023,7,20) - timedelta(days=10-i)
    # obtain indoor and outdoor temperature
    indoor = 30 + np.random.uniform(-2,2)
    outdoor = 28.3 + i*0.5  # simulate temperature increase
    historical_data.append([indoor, outdoor])

raw_data = np.array(historical_data)

# obtain recommended ac temperature with RBC AC
AC = AC_Temp_Control()
print(f"Current Indoor Temperature: {raw_data[-1][0]:.3f}°C\tOutdoor Temperature: {raw_data[-1][1]}°C")
print(f"Recommend AC Temperature: {AC.temp_set(raw_data[-1][1])}°C")
print(f"New Indoor Temperature: {indoor_temp_change(AC.temp_set(raw_data[-1][1]), raw_data[-1][0], raw_data[-1][0]):.3f}°C")
print(f"Energy Consumption: {energy_consume(AC.temp_set(raw_data[-1][1]), raw_data[-1][0], raw_data[-1][0]):.3f}kW")

# obtain recommended ac temperature with AC-MPC
ACMPC = AC_output(model_path=path+"ac_model_test.pth")
print(f"Current Indoor Temperature: {raw_data[-1][0]:.3f}°C\tOutdoor Temperature: {raw_data[-1][1]}°C")
print(f"Recommend AC Temperature: {ACMPC.predict(raw_data)}°C")
print(f"New Indoor Temperature: {indoor_temp_change(ACMPC.predict(raw_data), raw_data[-1][0], raw_data[-1][0]):.3f}°C")
print(f"Energy Consumption: {energy_consume(ACMPC.predict(raw_data), raw_data[-1][0], raw_data[-1][0]):.3f}kW")