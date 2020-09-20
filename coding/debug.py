""" This file should have break points (spefically a hitcount and a logpoint)
"""
import re
from datetime import datetime

from flask import Flask


class Classy():
    def __init__(self, name):
        self.id = name

    def get_id(self):
        return self.id

    def method(self):
        pass
        # print("Hi from " + str(self))

    def __str__(self):
        return f"==={self.id}==="


app = Flask(__name__)


@app.route("/")
def home():
    users = [Classy(i) for i in range(100)]
    for user in users:
        user.method()
    return "Hello, Flask!"

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")

    # Filter the name argument to letters only using regular expressions. URL arguments
    # can contain arbitrary text, so we restrict to safe characters only.
    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! It's " + formatted_now
    return content
