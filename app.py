import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, flash
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge
from scipy.stats import loguniform
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from pmdarima import auto_arima  # library for finding p, q, d
from statsmodels.tsa.stattools import adfuller  # library for finding d
from statsmodels.tsa.arima_model import ARIMA  # library for ARIMA model
from flask import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r"Dataset"
app.config['SECRET_KEY'] = 'b0b4fbefdc48be27a6123605f02b6b86'

global df, X_train, X_test, y_train, y_test


def preprocess():
    df = pd.read_csv('weather_history.csv')

    df['Date'] = df["Formatted Date"].str[:10]
    df['Summary'] = df['Summary'].replace({'Partly Cloudy': 'Cloudly',
                                           'Mostly Cloudy': 'Cloudly',
                                           'Overcast': 'Overcast',
                                           'Clear': 'Sunny',
                                           'Foggy': 'Foggy',
                                           'Breezy and Overcast': 'Overcast',
                                           'Breezy and Mostly Cloudy': 'Overcast',
                                           'Breezy and Partly Cloudy': 'Cloudy',
                                           'Dry and Partly Cloudy': 'Cloudy',
                                           'Windy and Partly Cloudy': 'Cloudy',
                                           'Light Rain': 'Rain',
                                           'Breezy': 'Rain',
                                           'Windy and Overcast': 'Overcast',
                                           'Humid and Mostly Cloudy': 'Cloudy',
                                           'Drizzle': 'Rain',
                                           'Windy and Mostly Cloudy': 'Cloudy',
                                           'Breezy and Foggy': 'Foggy',
                                           'Dry': 'Sunny',
                                           'Humid and Partly Cloudy': 'Cloudy',
                                           'Dry and Mostly Cloudy': 'Cloudy',
                                           'Rain': 'Rain',
                                           'Windy': 'Rain',
                                           'Humid and Overcast': 'Overcast',
                                           'Windy and Foggy': 'Foggy',
                                           'Breezy and Dry': 'Rain',
                                           'Windy and Dry': 'Rain',
                                           'Dangerously Windy and Partly Cloudy': 'Cloudy'})

    # There is only one unique value so no need of Loud Cover
    df.drop(['Loud Cover'], axis=1, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])

    df.drop(['Formatted Date'], axis=1, inplace=True)
    df.set_index(['Date'], inplace=True)

    df['Precip Type'].fillna(method='ffill', inplace=True)

    # As the correlation between Temperature (C) and Apparent Temperature (C) is almost 1 so we are dropping Apparent Temperature (C)
    df.drop(['Apparent Temperature (C)'], axis=1, inplace=True)

    df.select_dtypes(exclude='O')
    mm = MinMaxScaler()
    df['Temperature (C)'] = mm.fit_transform(pd.DataFrame(df['Temperature (C)']))
    df['Humidity'] = mm.fit_transform(pd.DataFrame(df['Humidity']))
    df['Wind Speed (km/h)'] = mm.fit_transform(pd.DataFrame(df['Wind Speed (km/h)']))
    df['Wind Bearing (degrees)'] = mm.fit_transform(pd.DataFrame(df['Wind Bearing (degrees)']))
    df['Visibility (km)'] = mm.fit_transform(pd.DataFrame(df['Visibility (km)']))
    df['Pressure (millibars)'] = mm.fit_transform(pd.DataFrame(df['Pressure (millibars)']))

    le = LabelEncoder()
    df['Summary'] = le.fit_transform(pd.DataFrame(df['Summary']))
    df['Precip Type'] = le.fit_transform(pd.DataFrame(df['Precip Type']))
    df['Daily Summary'] = le.fit_transform(pd.DataFrame(df['Daily Summary']))

    cat_names = ['Cloudly', 'Overcast', 'Sunny', 'Foggy', 'Cloudy', 'Rain']
    cat_names = pd.DataFrame(cat_names, columns=['cat_name'])
    cat_names['Summary'] = le.fit_transform(cat_names['cat_name'])

    df = df.groupby('Date').mean()
    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)
    return df


def imputation(data):
    dummy = []
    r = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    dummy = data.reindex(r).fillna(' ').rename_axis('Date').reset_index()
    dummy = dummy.replace(' ', np.nan)
    dummy = dummy.ffill()
    dummy.set_index('Date', inplace=True)
    return dummy


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Forecast for test data
def forecast_test(fitted, train, test):
    fc, se, conf = fitted.forecast(len(test), alpha=0.05)  # 95% confidence
    fc_series_test = pd.Series(fc, index = test.index)
    lower_series_test = pd.Series(conf[:, 0], index=test.index)
    upper_series_test = pd.Series(conf[:, 1], index=test.index)
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train, label='Training Data')
    plt.plot(test, color = 'blue', label='Actual Summary')
    plt.plot(fc_series_test, color = 'orange',label='Predicted Summary')
    plt.fill_between(lower_series_test.index, lower_series_test, upper_series_test, color='k', alpha=.10)
    plt.title('Weather Prediction')
    plt.xlabel('Time')
    plt.ylabel('Summary')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    #Errors - ARIMA model
    mse = mean_squared_error(test, fc_series_test)
    print('MSE: '+str(mse))
    mae = mean_absolute_error(test, fc_series_test)
    print('MAE: '+str(mae))
    rmse = mean_squared_error(test, fc_series_test, squared=False)
    print('RMSE: '+str(rmse))



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/load', methods=["POST", "GET"])
def load():
    if request.method == "POST":
        file = request.files['weather_history']
        ext1 = os.path.splitext(file.filename)[1]
        if ext1.lower() == ".csv":
            try:
                shutil.rmtree(app.config['UPLOAD_FOLDER'])
            except:
                pass
            os.mkdir(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'weather_history.csv'))
            flash('The data is loaded successfully', 'success')
            return render_template('load.html')
        else:
            flash('Please upload a csv type documents only', 'warning')
            return render_template('load.html')
    return render_template('load.html')


@app.route('/view', methods=['POST', 'GET'])
def view():
    df = preprocess()
    df = imputation(df)
    X = df.drop(['Summary'], axis=1)
    y = df['Summary']
    if request.method == 'POST':
        filedata = request.form['df']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=42)
        if filedata == '0':
            flash(r"Please select an option", 'warning')
        elif filedata == '1':
            return render_template('view.html', col=X_train.columns.values, df=list(X_train.values.tolist()))
        else:
            return render_template('view.html', col=X_test.columns.values, df=list(X_test.values.tolist()))

            # return render_template('view.html')
        # temp_df = pd.read_csv('Dataset/weather_history.csv')
        # print(temp_df)
        # temp_df =load(os.path.join(app.config["UPLOAD_FOLDER"]))

    return render_template('view.html')


x_train = None;
y_train = None;
x_test = None;
y_test = None


@app.route('/training', methods=['GET', 'POST'])
def training():
    df = preprocess()
    df = imputation(df)
    X = df.drop(['Summary'], axis=1)
    y = df['Summary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if request.method == 'POST':
        model_no = int(request.form['algo'])

        if model_no == 0:
            flash(r"You have not selected any model", "info")

        elif model_no == 1:
            model = SVR()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)

            msg1 = "MEAN ABSOLUTE ERROR OF SUPPORT VECTOR REGRESSOR IS :" + str(mae)
            msg2 = "MEAN SQUARED ERROR OF SUPPORT VECTOR REGRESSOR IS :" + str(mse)
            msg3 = "R2 SCORE OF SUPPORT VECTOR REGRESSOR IS :" + str(r2)
            return render_template('training.html', mag1=msg1, mag2=msg2, mag3=msg3)



        elif model_no == 2:
            model = DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None, ccp_alpha=0.0)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            msg1 = "MEAN ABSOLUTE ERROR OF DECISION TREE REGRESSOR IS :" + str(mae)
            msg2 = "MEAN SQUARED ERROR OF  DECISION TREE REGRESSOR IS :" + str(mse)
            msg3 = "R2 SCORE OF DECISION TREE REGRESSOR IS :" + str(r2)
            return render_template('training.html', mag1=msg1, mag2=msg2, mag3=msg3)




        elif model_no == 3:
            model = RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=0, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            msg1 = "MEAN ABSOLUTE ERROR OF RANDOM FOREST REGRESSOR IS :" + str(mae)
            msg2 = "MEAN SQUARED ERROR OF RANDOM FOREST REGRESSOR IS :" + str(mse)
            msg3 = "R2 SCORE OF RANDOM FOREST REGRESSOR IS  :" + str(r2)
            return render_template('training.html', mag1=msg1, mag2=msg2, mag3=msg3)



        elif model_no == 4:
            model =LinearRegression()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            msg1 = "MEAN ABSOLUTE ERROR OF LINEAR REGRESSOR IS :" + str(mae)
            msg2 = "MEAN SQUARED ERROR OF LINEAR REGRESSOR IS :" + str(mse)
            msg3 = "R2 SCORE OF RANDOM LINEAR REGRESSOR IS  :" + str(r2)
            return render_template('training.html', mag1=msg1, mag2=msg2, mag3=msg3)



        elif model_no == 5:
            model =Ridge()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            msg1 = "MEAN ABSOLUTE ERROR OF RIDGE REGRESSOR IS :" + str(mae)
            msg2 = "MEAN SQUARED ERROR OF RIDGE REGRESSOR IS :" + str(mse)
            msg3 = "R2 SCORE OF RIDGE REGRESSOR IS  :" + str(r2)
            return render_template('training.html', mag1=msg1, mag2=msg2, mag3=msg3)


    return render_template('training.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    df = preprocess()
    df = imputation(df)
    X = df.drop(['Summary'], axis=1)
    y = df['Summary']
    if request.method == "POST":
        Precip_Type = request.form['Precip Type']
        print(Precip_Type)
        Temperature = request.form['Temperature (C)']
        print(Temperature)
        Humidity = request.form['Humidity']
        print(Humidity)
        Wind_Speed = request.form['Wind Speed (km/h)']
        print(Wind_Speed)
        Wind_Bearing = request.form['Wind Bearing (degrees)']
        print(Wind_Bearing)
        Visibility = request.form['Visibility (km)']
        print(Visibility)
        Pressure = request.form['Pressure (millibars)']
        print(Pressure)
        Daily_Summary = request.form['Daily Summary']
        print(Daily_Summary)

        di = {'Precip Type': [Precip_Type], 'Temperature (C)': [Temperature], 'Humidity': [Humidity],
              'Wind Speed (km/h)': [Wind_Speed],
              'Wind Bearing (degrees)': [Wind_Bearing], 'Visibility (km)': [Visibility],
              'Pressure (millibars)': [Pressure],
              'Daily Summary': [Daily_Summary]}

        test = pd.DataFrame.from_dict(di)
        print(test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        cfr = RandomForestRegressor()
        model = cfr.fit(X_train, y_train)
        output = model.predict(test)
        print(output)

        if output<1:
            msg = 'The Weather is <span style = color:grey;> Cloudy </span></b>'

        elif output<2:
            msg = 'The Weather is <span style = color:red;> Foggy </span></b>'

        elif output<3:
            msg = 'The Weather is <span style = color:blue;> Overcast </span></b>'

        elif output<4:
            msg = 'The Weather is <span style = color:red;> Rain </span></b>'

        else:
            msg = 'The Weather is <span style = color:dark yellow;>Sunny</span></b>'

        return render_template('prediction.html', mag=msg)
    return render_template('prediction.html')


@app.route('/Graph')
def Graph():

    return render_template('Graph.html')



if __name__ == '__main__':
    app.run(debug=True)
