from flask import Flask, render_template, request
from weather import get_weather_data
from waitress import serve

app = Flask(__name__)
@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"

@app.route('/weather')
def get_weather():
    city = request.args.get('city', 'Punggol Coast')



if __name__ == '__main__':
    app.run(app, host = "0.0.0.0", port = 5000)