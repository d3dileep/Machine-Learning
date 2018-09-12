"""
datapoints are taken and plotted from all previous dates of particular month and year

@author: DILEEP
"""
import csv
import numpy as np 
from sklearn.svm import SVR 
import matplotlib.pyplot as plt 

month=[]
dates=[]
prices=[]
print("type corresponding integer \n","1 : opening price \n","2 : daily high \n", "3 : daily low \n" \
      "4 : closing price \n","5 : volume \n","6 : turnover")
k=input("choose what you want to predict-")
print("error means sample points are not available for following date")
s=input("enter date to predict-")
w=input("enter month to predict-")
def get_data(result,s):
    with open(result,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            u = int(row[0].split('-')[0]) #month
            v = int(row[0].split('-')[1]) #date
           
            
            if v == int(s):
               if u== int(w):
                    dates.append(int(row[0].split('-')[2]))                
                    prices.append(float(row[int(k)]))
                
    return                                      


def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates),1))
    svr_rbf = SVR(kernel='rbf', C=1e5, gamma=0.5) #changing parameter will modify result
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black',label='Data')
    plt.plot(dates,svr_rbf.predict(dates), color='red', label='rbf_model')
    plt.xlabel('Dates')
    plt.ylabel('Prices')
    plt.title('support vector regression')
    #plt.legend()
    #plt.show()

    return svr_rbf.predict(x)[0]

get_data('result.csv',s)

p=input("year to predict-")
predicted_price = predict_prices(dates, prices,int(p))
print(predicted_price)
print(s,"-",w,"-",p)
