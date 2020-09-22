# -*- coding=utf-8 -*-
# python37
from flask import render_template, session, redirect, url_for
import face_recognition
from webapp.forms import *
import os
import numpy as np
from webapp.models import User
from webapp.face_handler import get_face_feature
from webapp import app
import json

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.photo.data
        save_as = os.path.join('webapp/temp', 'upload.jpg')
        f.save(save_as)
        session['file'] = save_as
        return redirect(url_for('recog'))
    return render_template('upload.html', form=form)

@app.route('/recog')
def recog():
    to_be_recoged = get_face_feature(session['file'])
    # os.remove(session['file'])
    users = User.query.all()
    distances = []
    for user in users:
        user_feature = np.array(json.loads(user.feature))
        distances.append(face_recognition.face_distance([user_feature, ], to_be_recoged)[0])
    target_user = users[distances.index(min(distances))]
    return target_user.username
