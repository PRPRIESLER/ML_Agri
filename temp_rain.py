# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 13:37:41 2018


"""

import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template

app = Flask(__name__)
@app.route('/')
def my_form():
    return render_template('weather.html')


@app.route('/', methods=['GET','POST'])
def my_form_post():
    year = request.form['year']

    dataset = pd.read_csv('temp_rain.csv')
    year = year


    temp_g = dataset.iloc[:, [0]].values
    rain_g = dataset.iloc[:, [1]].values
    month_g = dataset.iloc[:, [2]].values

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf1 = SVR(kernel='rbf', C=1e3, gamma=0.1)

    temperature = []
    rain = []
    result=[]

    for i in range(1, 13):
        mydataset = dataset[dataset.month == i]
        x = mydataset.iloc[:, [3]].values
        y_temp = mydataset.iloc[:, [0]].values
        y_rain = mydataset.iloc[:, [1]].values

        xtrain, xtest, ytrain, ytest = train_test_split(x, y_temp, test_size=.3)
        x_train, x_test, y_train, y_test = train_test_split(x, y_rain, test_size=.3)

        svr_rbf.fit(xtrain, ytrain.ravel())
        y_rbf = svr_rbf.predict(xtest)
        act_pred = pd.DataFrame([year])
        act_pred = svr_rbf.predict(act_pred)

        svr_rbf1.fit(x_train, y_train.ravel())
        y_rbf1 = svr_rbf1.predict(x_test)
        act_pred1 = pd.DataFrame([year])
        act_pred1 = svr_rbf1.predict(act_pred1)

        temperature.append(act_pred)
        rain.append(act_pred1)


        print("%.i month %.i Temperature: %.2f Accuracy: %.2f" % (int(year), i, act_pred, svr_rbf.score(ytest, y_rbf)))

        print("%.i month %.i Rain: %.2f Accuracy: %.2f \n" % (int(year), i, act_pred1, svr_rbf1.score(y_test, y_rbf1)))
        result.append(["%.i month %.i Temperature: %.2f " % (int(year), i, act_pred)])
        result.append(["%.i month %.i Rain: %.2f " % (int(year), i, act_pred1)])


    plt.scatter(month_g, temp_g, color='darkorange', label='data')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], temperature, color='navy', lw=2, label='RBF model')
    plt.xlabel('month')
    plt.ylabel('temperature')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    plt.scatter(month_g, rain_g, color='darkorange', label='data')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], rain, color='navy', lw=2, label='RBF model')
    plt.xlabel('month')
    plt.ylabel('rain')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    processed_text = result
    return render_template('weather.html',your_list=processed_text)



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5018)
