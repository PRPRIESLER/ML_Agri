
#!/usr/bin/python
# -*- coding: utf-8 -*-

# Code source: Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score



from flask import Flask, request, render_template
import sklearn
print (sklearn.__version__)

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('yieldprediction.html')


@app.route('/', methods=['POST'])
def my_form_post():
    Selectstate = request.form['Selectstate']
    SelectCrop = request.form['SelectCrop']
    AreatoHarvest = request.form['AreatoHarvest']



    # Load the diabetes dataset
    importdataset = pd.read_csv('Cropyieldprediction.csv')
    # State, District, year, Season, Crop, Area
    predict_data = [Selectstate, 'SHIMOGA', 1997, SelectCrop, 'Maize', AreatoHarvest]
    # df = pd.DataFrame(futureprediction_data,columns=['State_Name','District','year','crop','area'])

    State = predict_data[0]
    crop = predict_data[4]
    mydata = pd.DataFrame([predict_data[5]])

    mydataset = importdataset[(importdataset.State_Name == State) & (importdataset.Crop == crop)]
    mydataset = mydataset[~np.isnan(mydataset["Production"])]

    # Use only one feature
    x = mydataset.iloc[:, [5]]
    y = mydataset.iloc[:, [6]]



    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x, y)

    # Split the data into training/testing sets
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.7)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)
    my_pred = regr.predict(mydata)
    Output=int(my_pred)
    print('Actual predicted Production :', str(int(my_pred)))
    # The coefficients
    #print('Coefficients: ', regr.coef_)
    # The mean squared error
    #print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    print()
    # Plot outputs
    plt.scatter(x_test, y_test, color='black')
    # plt.fill("1000000","1000000",color='yellow')
    plt.plot(x_test, y_pred, color='blue', linewidth=2)
    plt.title("Crop yield predicton")
    plt.xlabel("Area in hectors")
    plt.ylabel("Production in tons")
    plt.show()
    processed_text = Output
    if processed_text<0:
        processed_text*=-1
    return render_template('yieldprediction.html',value=processed_text)

if __name__ == '__main__':
   app.run(debug = True)