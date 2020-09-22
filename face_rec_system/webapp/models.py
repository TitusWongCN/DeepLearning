# -*- coding=utf-8 -*-
# python37
from webapp import db

class User(db.Model):
    id = db.Column(db.Integer(), primary_key=True, index=True, nullable=False)
    username = db.Column(db.String(40), index=True, nullable=False)
    feature = db.Column(db.String(200), nullable=True)

    def __repr__(self):
        return '<User %r>' % (self.username)
