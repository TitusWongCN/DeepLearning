# -*- coding=utf-8 -*-
# python37
import os
from webapp.models import User
from webapp.face_handler import get_face_feature
import json
import time
from webapp import app, db

db_path = 'webapp/users.db'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = 'you-will-never-guess'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, db_path)

if not os.path.isfile(db_path):
    db.create_all()
    face_cache = 'FacePhotos'
    if len(os.listdir(face_cache)) > 0:
        for index, face_image_file in enumerate(os.listdir(face_cache)):
            username = face_image_file.split('_')[0]
            face_image = os.path.join(face_cache, face_image_file)
            feature = json.dumps(list(get_face_feature(face_image)))
            user = User(
                id=int(1000 * time.time()),
                username=username,
                feature=feature
            )
            db.session.add(user)
            db.session.commit()
app.run(debug=True)
