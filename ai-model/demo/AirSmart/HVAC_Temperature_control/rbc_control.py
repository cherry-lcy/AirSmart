import pandas as pd
import numpy as np
from hvac_basic import mock_indoor_temp, energy_consume, indoor_temp_change
from rbc import AC_Temp_Control

# variable list
place = "HKA"
path = ""  # path you store the data

# prepare training data
clean_df = pd.read_csv(path+"ALL_DATA_"+place+"_.csv")  # processed data from cell 12
train_size = int(len(clean_df["Mean Temp"]) * 0.8)
test_size = len(clean_df["Mean Temp"]) - train_size
train_df, test_df = clean_df.iloc[:train_size], clean_df.iloc[train_size:]
train_df = train_df[["Date", "Year", "Month", "Day", "Mean Temp"]]
test_df = test_df[["Date", "Year", "Month", "Day", "Mean Temp"]]

# add mock indoor temperature to the data
indtemp = mock_indoor_temp(list(test_df["Mean Temp"]))
test_df["Indoor Temp"] = indtemp

# Set the temperature of AC
AC_Temp_Simple = AC_Temp_Control(buffer=5)
temps = AC_Temp_Simple.auto_ac_control(test_df, "Mean Temp")
ac_temp = test_df
ac_temp["AC Temp"] = temps

# temperature control and energy consumption based test data
ac_temp["Final Indoor Temp"] = np.nan
for index, row in ac_temp.iterrows():
    ac_temp.loc[index, "Final Indoor Temp"] = indoor_temp_change(
        row["AC Temp"], row["Indoor Temp"], row["Mean Temp"], time_step=1/24
    )
energy = energy_consume(ac_temp["AC Temp"], ac_temp["Indoor Temp"])
ac_temp["Energy Consumption"] = energy

# calculate temperature error
temp_errors = np.abs(np.array(ac_temp["Final Indoor Temp"])-22)
avg_energy = np.mean(energy)

# print test results
print("\n" + "="*40)
print(f"{' Test report ':=^40}")
print(f"Average temperature error: {np.mean(temp_errors):.2f} ± {np.std(temp_errors):.2f} °C")
print(f"Average energy consumption: {avg_energy:.2f} ± {np.std(energy):.2f} kW")
print("="*40)