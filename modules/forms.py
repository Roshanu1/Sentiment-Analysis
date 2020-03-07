from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import Length, DataRequired

class KeywordSearch(FlaskForm):
	keyword = StringField(validators = [DataRequired(),Length(min=2, max=20)])
	submit = SubmitField('Search Tweets')

class ProfileSearch(FlaskForm):
	searchTerm = StringField(validators=[DataRequired(), Length(min=2, max=20)])
	submit = SubmitField('Go')

class TextClassification(FlaskForm):
	text = TextAreaField(validators = [DataRequired()])
	submit = SubmitField("Go")