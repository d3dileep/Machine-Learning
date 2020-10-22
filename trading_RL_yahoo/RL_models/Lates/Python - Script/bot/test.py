from flask import Flask,request,redirect,render_template
from main import *
import sys


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/result',methods=['POST','GET'])
def result():
	with open("profit.txt","r") as G:
		profit = G.read()
	return render_template("result.html",profit = profit)

@app.route('/',methods=['POST','GET'])
def index():
	if request.method == 'POST':
		symbol = request.form['ticker']
		_id = request.form['train_bool']
		run_method(symbol,_id)
		return redirect("/result")
	content = ""
	return render_template("index.html")	
if __name__ == '__main__':
	global value;
	app.run(host='127.0.0.1', port=8080, debug=True) 