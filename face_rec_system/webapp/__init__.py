# -*- coding=utf-8 -*-
# python37
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
db = SQLAlchemy(app)

from webapp import models, views
