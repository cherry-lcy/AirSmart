import numpy as np
import random

# create indoor temperature data
def mock_indoor_temp(data):
    indoor_temp = []
    for i in range(len(data)):
        temp = data[i] + random.uniform(-5,5)
        indoor_temp.append(temp)
    return np.array(indoor_temp)

# calculate the new indoor temperature after setting AC temperature
def indoor_temp_change(ac_temp, indoor_temp, outdoor_temp, room_heat_capacity=1000,
                       ac_capacity=10, outdoor_heat_coeff=0.5,
                       time_step=1.0):
    # heat exchange
    q_ac = ac_capacity * (ac_temp - indoor_temp) * time_step
    q_outdoor = outdoor_heat_coeff * (outdoor_temp - indoor_temp) * time_step
    delta_temp = (q_ac + q_outdoor) / room_heat_capacity
    indoor_temp += delta_temp
    indoor_temp = np.clip(indoor_temp, 10.0, 40.0)

    return indoor_temp

# calculate energy consumption (kW)
def energy_consume(ac_temp, indoor_temp, ac_capacity=10, time_step=1):
    ac_temp = np.array(ac_temp)
    indoor_temp = np.array(indoor_temp)
    energy = ac_capacity * abs(ac_temp - indoor_temp) * time_step
    return energy