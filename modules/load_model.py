import pickle
from nltk import TweetTokenizer

tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

def word_features(tweet):
	'''
	Function to extract word features (unigrams in this case) and start preparing the data for training
	'''
	features = tknzr.tokenize(tweet)
	return dict([(word), True] for word in features)

class LoadModel():
	'''
	Argument for LoadModel() determines which model to use.
	"NaiveBayes" => Naive Bayes model
	"SVM" => Support Vector Machine model
	'''
	def __init__(self, model_name):
		self.model = model_name
		self.model_file = "NB"
		self.classifier = False

		if self.model == "NaiveBayes":
			pass
		elif self.model == "SVM":
			self.model_file = "SVM"

		with open("models/" + self.model_file + ".pickle", "rb") as m:
			self.classifier = pickle.load(m)

	def polarity(self, unseen):
		prob_dist = self.classifier.prob_classify(word_features(unseen))
		pred_label = prob_dist.max()

		if pred_label == "pos":
			p = 1
		elif pred_label == "neg":
			p = -1
		else:
			p = 0

		return p*round(prob_dist.prob(pred_label),4)

	def label(self, unseen):
		return self.classifier.classify(word_features(unseen))

if __name__ == '__main__':
	model = LoadModel("NaiveBayes")

	tweet = "This is bad !!!"
	print(tweet, model.polarity(tweet))