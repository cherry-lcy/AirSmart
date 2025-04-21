import pandas as pd

# variable list
place = "HKA"
path = "" # path you store the data
invalid = "***"

# open the csv file
data = pd.read_csv(path+"CLMTEMP_"+place+"_.csv", on_bad_lines='skip')
maxt = pd.read_csv(path+"CLMMAXT_"+place+"_.csv", on_bad_lines='skip')
mint = pd.read_csv(path+"CLMMINT_"+place+"_.csv", on_bad_lines='skip')
wind = pd.read_csv(path+"daily_"+place+"_WSPD_ALL.csv", on_bad_lines='skip')
hum = pd.read_csv(path+"daily_"+place+"_RH_ALL.csv", on_bad_lines='skip')
pr = pd.read_csv(path+"daily_"+place+"_MSLP_ALL.csv", on_bad_lines='skip')

# create dataframe
meandf = pd.DataFrame(data)
maxdf = pd.DataFrame(maxt)
mindf = pd.DataFrame(mint)
winddf = pd.DataFrame(wind)
humdf = pd.DataFrame(hum)
prdf = pd.DataFrame(pr)

# combine all csv files into 1
maxdf = maxdf.drop(columns=["data Completeness"])
mindf = mindf.drop(columns=["data Completeness"])
meandf = meandf.drop(columns=["data Completeness"])
# convert year, month and day columns into int
meandf["Year"] = meandf["Year"].astype(int)
meandf["Month"] = meandf["Month"].astype(int)
meandf["Day"] = meandf["Day"].astype(int)
# create a dataframe to collect all data
df = pd.DataFrame()
df["Date"] = meandf.apply(lambda row: f"{int(row['Year'])}-{int(row['Month'])}-{int(row['Day'])}", axis=1)
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = meandf["Year"]
df["Month"] = meandf["Month"]
df["Day"] = meandf["Day"]
df["Max Temp"] = maxdf["Value"]
df["Mean Temp"] = meandf["Value"]
df["Min Temp"] = mindf["Value"]
df["Humidity"] = humdf["Value"]
df["Wind Speed"] = winddf["Value"]
df["Mean Pressure"] = prdf["Value"]

# delete invalid data
clean_df = df[(df["Max Temp"] != invalid) & (df["Mean Temp"]!=invalid) & (df["Min Temp"]!=invalid) & (df["Humidity"]!=invalid) & (df["Wind Speed"]!=invalid) & (df["Mean Pressure"]!=invalid)]

# convert "Mean Temperature" column to number
clean_df['Mean Temp'] = pd.to_numeric(df['Mean Temp'], errors='coerce')
clean_df['Max Temp'] = pd.to_numeric(df['Max Temp'], errors='coerce')
clean_df['Min Temp'] = pd.to_numeric(df['Min Temp'], errors='coerce')
clean_df['Humidity'] = pd.to_numeric(df['Humidity'], errors='coerce')
clean_df['Wind Speed'] = pd.to_numeric(df['Wind Speed'], errors='coerce')
clean_df['Mean Pressure'] = pd.to_numeric(df['Mean Pressure'], errors='coerce')

# add Humidity Pressure Ratio column
clean_df["Humidity Pressure Ratio"] =clean_df["Humidity"] /clean_df["Mean Pressure"]

# export the file for record
clean_df.to_csv(path+"ALL_DATA_"+place+"_.csv", encoding="UTF-8")

# create train data and test data for BiLSTM model training
train_size = int(len(clean_df["Mean Temp"]) * 0.8)
test_size = len(clean_df["Mean Temp"]) - train_size
train, test = clean_df.iloc[:train_size], clean_df.iloc[train_size:]
train = train[["Month", "Day", "Max Temp", "Min Temp", "Humidity", "Wind Speed", "Mean Pressure", "Humidity Pressure Ratio", "Mean Temp"]]
test = test[["Month", "Day", "Max Temp", "Min Temp", "Humidity", "Wind Speed", "Mean Pressure", "Humidity Pressure Ratio", "Mean Temp"]]

# save dataframe
train.to_csv(path+"TRAIN_"+place+"_DATASET_.csv", encoding = "UTF-8")
test.to_csv(path+"TEST_"+place+"_DATASET_.csv", encoding = "UTF-8")