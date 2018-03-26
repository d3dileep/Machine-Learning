"""
This allows to predict required value and the prediction constants "C and gamma" has been found by hit & trail.
"""
import csv
import numpy as np 
from sklearn.svm import SVR 
import matplotlib.pyplot as plt 
print("type corresponding integer \n","1 : opening price \n","2 : daily high \n", "3 : daily low \n" \
      "4 : closing price \n","5 : volume \n","6 : turnover")
k=input("choose what you want to predict-")
print("error means sample points are not available for following date")
month=[]
dates=[]
prices=[]

#s=input("enter month to predict-")
def get_data(result):
    with open(result,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
               
            dates.append(int(row[0].split('-')[2]))                
            prices.append(float(row[int(k)]))
                    
    return                                      


def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates),1))
    svr_rbf = SVR(kernel='rbf', C=1e5, gamma=0.5)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black',label='Data')
    plt.plot(dates,svr_rbf.predict(dates), color='red', label='rbf_model')
    plt.xlabel('Dates')
    plt.ylabel('Prices')
    plt.title('support vector regression')
    #plt.legend()
    #plt.show()

    return svr_rbf.predict(x)[0]

get_data('result.csv')

p=input("year to predict-")
predicted_price = predict_prices(dates, prices,int(p))
print(predicted_price)

