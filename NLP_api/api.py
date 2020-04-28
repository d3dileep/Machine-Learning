from flask import Flask, jsonify
from flask_restful import Resource, Api
import json

app = Flask(__name__)
api = Api(app)

class Hello(Resource):
    def get(self):
        return {'Message': 'Hello!'}

class Increase_Status(Resource):
    def get(self):
        with open('Increase_alert.json') as f:
            data = json.load(f)
        return(jsonify(data))

class Decrease_Status(Resource):
    def get(self):
        with open('Decrease_alert.json') as f:
            data = json.load(f)
        return(jsonify(data))

class Main_news(Resource):
    def get(self):
        with open('Main_news.json') as f:
            data = json.load(f)
        return(jsonify(data))

class Sub_news(Resource):
    def get(self):
        with open('Sub_news.json') as f:
            data = json.load(f)
        return(jsonify(data))

class Main_search(Resource):
    def get(self):
        with open('Main_search.json') as f:
            data = json.load(f)
        return(jsonify(data))

class Sub_search(Resource):
    def get(self):
        with open('Sub_search.json') as f:
            data = json.load(f)
        return(jsonify(data))


api.add_resource(Hello,'/')
api.add_resource(Increase_Status,'/increase')
api.add_resource(Decrease_Status,'/decrease')
api.add_resource(Main_news,'/main_news')
api.add_resource(Sub_news,'/sub_news')
api.add_resource(Main_search,'/main_search')
api.add_resource(Sub_search,'/sub_search')

if __name__ == "__main__":
    app.run()
    #host = '0.0.0.0' , port = 80