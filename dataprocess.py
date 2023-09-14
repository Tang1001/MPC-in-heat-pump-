import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def load_and_preprocess_data(filepath, sample_time):
    # Read the data
    df = pd.read_csv(filepath, delimiter=';')
    # print(df)

    # Renaming for ease
    df.columns.values[0] = "timestamp"
    df.columns.values[1] = "value"

    # Convert Dutch-style numbers to standard floating point numbers
    df['value'] = df['value'].str.replace(',', '.').astype(float)

    # Convert 'timestamp' to datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Exclude weekend data
    df = df[df['timestamp'].dt.weekday < 5]
    # print(df)

    # Extract hour and day from DateTime
    df['minute'] = df['timestamp'].dt.minute
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    # print(df)

    # Filter data to only include hours from 7am to 7pm
    df = df[((df['hour'] >= 7) & (df['hour'] < 19)) | ((df['hour'] == 19) & (df['minute'] == 0))]
    # df.set_index('timestamp', inplace=True)

    # Set the timestamp as index
    df.set_index('timestamp', inplace=True)

    if len(df) == 90:
        # New data to add to the beginning and end
        date = df.index[0].date()
        beginning_time = pd.Timestamp(str(date) + ' 07:00:00')
        end_time = pd.Timestamp(str(date) + ' 19:00:00')

        # New data to add to the beginning and end
        beginning = pd.DataFrame(
            {'value': [df['value'][0]], 'hour': [beginning_time.hour], 'day': [beginning_time.day]},
            index=[beginning_time])
        end = pd.DataFrame({'value': [df['value'][-1]], 'hour': [end_time.hour], 'day': [end_time.day]},
                           index=[end_time])

        # Concatenate the new data with the original DataFrame
        df = pd.concat([beginning, df, end])

    # Set the desired interval as a variable
    # sample_time = 8
    sample_time_string = f"{sample_time}T"
    # print(sample_time_string)

    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=sample_time_string)
    # date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=sample_time_string)
    # date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='1T')
    df_interp = pd.DataFrame(index=date_range)
    # df_interp['value'] = None

    # print()
    if len(df) <= 13:
        # print(df_interp)
        # print(df)
        peak_time = 30
        df_interp = df_interp.join(df[:], how='outer')
        # print(df_interp)
        df_interp = df_interp.shift(periods=-round((60 - peak_time)/sample_time))
        # print(df_interp)

        # df_interp['value'] = df_interp['value'].interpolate(method='spline', order=4)
        df_interp['value'][0] = df['value'][0]
        df_interp['value'][-1] = df['value'][-1]
        df_interp['value'] = df_interp['value'].interpolate(method='polynomial', order=3)
        df_interp = df_interp[((df_interp.index.hour-7)*60+df_interp.index.minute) % sample_time == 0]
        # print(df_interp['value'])


        # Adjust the interpolated values to ensure the sum matches the original data
        total_hourly = df['value'][1:].sum()
        total_minute = df_interp['value'][1:].sum() * (sample_time / 60)  # Converting to hours

        difference = total_hourly - total_minute
        # print(difference)
        df_interp['value'][1:] = df_interp['value'][1:] + (difference / (len(df_interp) - 1))

        # print(df_interp)

        # # Plotting
        # plt.figure(figsize=(14, 6))
        # df['value'].plot(drawstyle='steps-pre',label='Original Data', marker='o')
        # # df['value'].plot(drawstyle='steps-post', label='Original Data', marker='o')
        # df_interp['value'].plot(label='Interpolated & Adjusted Data', linestyle='--', alpha=0.7)
        # # plt.title('Water Consumption Data')
        # plt.xlabel('Timestamp')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()





    elif len(df) <= 92:  # Temp of upper tank2
        # print(df)
        # print(df_interp)
        df_interp = df_interp.join(df[:], how='outer')
        df_interp['value'] = df_interp['value'].interpolate(method='polynomial', order=3)
        df_interp = df_interp[((df_interp.index.hour-7)*60+df_interp.index.minute) % sample_time == 0]
        # print(df_interp['value'])

        # # Plotting
        # plt.figure(figsize=(14, 6))
        # df['value'].plot(label='Original Data')
        # # df['value'].plot(drawstyle='steps-pre',label='Original Data', marker='o')
        # # df['value'].plot(drawstyle='steps-post', label='Original Data', marker='o')
        # df_interp['value'].plot(label='Interpolated & Adjusted Data', linestyle='--', alpha=0.7)
        # # plt.title('Water Consumption Data')
        # plt.xlabel('Timestamp')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

    else:
        # print(df)
        # print(df_interp)
        df_interp = df_interp.join(df, how='outer')
        df_interp = df_interp[((df_interp.index.hour-7)*60+df_interp.index.minute) % sample_time == 0]
        # print(df_interp['value'])
        # df_interp['value'] = df_interp['value'].interpolate(method='polynomial', order=3)
        # print(df_interp['value'])

    return df, df_interp
