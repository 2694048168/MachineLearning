"""
Minimal example using Flask (Flask's Hello World application) with external access and showing route() decorator
"""

# Import required packages:
from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/user")
def hello_user():
    return "User: Hello World!"


if __name__ == "__main__":
    # Add parameter host='0.0.0.0' to run on your machines IP address:
    app.run(host='0.0.0.0')
