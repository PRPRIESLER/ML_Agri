# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:45:39 2018

@author: Abdul
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score,r2_score
from flask import Flask, request, render_template
app = Flask(__name__)
@app.route('/')
def my_form():
    return render_template('findmsp.html')


@app.route('/', methods=['POST'])
def my_form_post():
    selectcrop = request.form['selectcrop']
    year = request.form['year']

    # Load the msp dataset
    importmspdataset = pd.read_csv('MSP.csv')
    # crop, year
    predict_data = [selectcrop, year]
    # df = pd.DataFrame(futureprediction_data,columns=['State_Name','District','year','crop','area'])

    crop = predict_data[0]

    predyear = pd.DataFrame([predict_data[1]])
    mspdataset = importmspdataset[importmspdataset.Crop == crop]
    # mspdataset = mspdataset[~np.isnan(mspdataset["Crop"])]

    # Use only one feature
    # x=pd.DataFrame([[2014],[2015],[2016],[2017],[2018]],index=[2014,2015,2016,2017,2018])
    x = mspdataset.iloc[:, [1]]
    y = mspdataset.iloc[:, [2]]
    # y=y.T


    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(x, y)

    # Split the data into training/testing sets
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.7)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)
    my_pred = regr.predict(predyear)
    Output=int(my_pred)

    print('Predicted MSP is :', int(my_pred))
    print('Accuracy:',r2_score(y_pred ,y_test)*100)

    # The coefficients
    #print('Coefficients: ', regr.coef_)
    # The mean squared error
    #print("Mean squared error: %.2f" %mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    #print('Variance score: %.2f' % r2_score(y_test, y_pred))
    # Plot outputs
    plt.scatter(x_test, y_test, color='black')
    # plt.fill("1000000","1000000",color='yellow')
    plt.plot(x_test, y_pred, color='blue', linewidth=2)
    plt.title("Minimum support price")
    plt.xlabel("years")
    plt.ylabel("m s p")
    plt.show()
    processed_text = Output
    if processed_text<0:
        processed_text*=-1
    return render_template('findmsp.html', value=processed_text)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5018)