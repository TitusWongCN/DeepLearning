# -*- coding=utf-8 -*-
# python37
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from flask_wtf import FlaskForm

class UploadForm(FlaskForm):
    photo = FileField('Upload Image', validators=[FileRequired(), FileAllowed(['jpg','jpeg','png','gif'])])
    submit = SubmitField()
