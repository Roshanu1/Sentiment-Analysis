from gevent.monkey import patch_all
patch_all()

from flask import Flask, render_template, request, url_for, redirect
from flask_socketio import SocketIO, emit
from os import environ

import modules.forms as forms
import modules.analyze as tw
from modules.load_model import LoadModel

from threading import Thread
from json import dumps, loads
from math import floor
from random import shuffle

thread, tag, run = None, None, True

app = Flask(__name__)

# Key against CSRF
app.config["SECRET_KEY"] = '1f5aad569eef9cc369c32ec14f3d2cab'

socketio = SocketIO(app)

port = int(environ.get("PORT", 5000))

model = LoadModel("NaiveBayes")

def weigh(topics):
	'''
	Assigns weights to topics for tag cloud
	'''
	maximum = topics[0][1]
	shuffle(topics)
	topics = dict(topics)

	for key in topics:
		topics[key] = floor(topics[key]/maximum*5)
	return topics

def changeState(P):
	'''
	Function to stop the twitter stream
	'''
	global thread
	global run

	if P == 0:
		run = False
		thread = False
	elif P == 1:
		run = True
		thread = True


@app.route('/')
def index():
    return render_template("home.html", title="Sentiment Analysis")

@app.route('/analyze/text', methods=["GET", "POST"])
def text():
	text = request.form.get("text")
	topics, polarities, sentences, f = None, None, None, None
	if text:
		f = 1
		topics = tw.getTopics([text], 10)
		topics = weigh(topics) if topics else None
		sentences = tw.textTokenize(text)
		text_polarity = model.label(text)

		sentences = tw.group_by_polarity([(model.label(sentence), sentence) for sentence in sentences])

		polarities = {}
		for polarity, sentence in sentences.items():
			polarities[polarity] = len(sentence)
	return render_template('text.html',title="Text Analysis", form=forms.TextClassification(data={"text":text}), topics=topics, polarities=polarities, sentences=sentences, f=f)

@app.route('/about')
def about():
    return render_template("about.html", title="About Sentiment Analysis", name="Sentiment Analysis")



if __name__ == "__main__":
	socketio.run(app, host='0.0.0.0', port=port, debug=True)