# construct a simple class to set the ac temp with
class AC_Temp_Control:
  def __init__(self, average_temp = 22, buffer = 2):
    self.average_temp = average_temp
    self.buffer = buffer

  def auto_ac_control(self, data, column_name):
    ac_temp = []
    for i in range(len(data)):
      ac_temp.append(self.temp_set(data[column_name].iloc[i]))

    return ac_temp

  def temp_set(self, temp):
    if temp <= 20:
      return self.average_temp + self.buffer
    elif temp > 20 and temp <= 25:
      return self.average_temp
    elif temp > 25:
      return self.average_temp - self.buffer