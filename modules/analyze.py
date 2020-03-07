from json import loads
from tweepy import Stream, OAuthHandler, API
from tweepy.streaming import StreamListener
from os import getenv

from nltk.corpus import stopwords
from nltk import TweetTokenizer, pos_tag, sent_tokenize
from collections import Counter
from re import match
from collections import defaultdict

# Get the follow keys from your twitter account
TWIT_API = getenv('TWIT_API', '')
TWIT_SECRET = getenv('TWIT_SECRET', '')
TWIT_ACCESS_TOKEN = getenv('TWIT_ACCESS_TOKEN', '')
TWIT_ACCESS_SECRET = getenv('TWIT_ACCESS_SECRET', '')

class listener(StreamListener):
	# Currently reads only the tweet text. Also see if metadata can improve analysis.
	def __init__(self):
		self.count = 0
		
	def on_data(self, data):
		tweet = loads(data)["text"]
		# print(tweet)
		return True

	def on_error(self, status):
		print("Error! ", (status))

def stream(tag):
	'''
	A stream is probably better with single keywords. Multiple keywords ("apple", "android") will shift live sentiment badly.
	'''
	stream = Stream(auth, listener())
	stream.filter(languages=["en"], track=[tag])

def readTweepyObj(TwObj):
	'''
	Returns a list of dictionaries
	'''
	data = []

	for element in TwObj:
		data.append(element._json)

	return data

def unpackTwDict(packed, toUnpack):
	'''
	Unpacks the dictionary that tweepy API outputs
	'''
	unpacked = []
	for dic in packed:
		unpacked.append({key: dic[key] for key in toUnpack})
	return unpacked

def profilesearch(q, count=20):
	'''
	Search for Twitter profiles
	'''
	keys = ["id", "name", "screen_name", "description", "verified", "profile_image_url_https", "followers_count"]
	return unpackTwDict(readTweepyObj(api.search_users(q, count)), keys)

def getTweets(user, count=50):
	return unpackTwDict(readTweepyObj(api.user_timeline(screen_name=user, count=count, tweet_mode="extended")), ["created_at", "id", "full_text"])

def getTopics(tweets, count=10):
	stop_words = set(stopwords.words("english"))
	stop_words.update(["rt","anybody","anyone","anything","everybody","everyone","everything","nobody","noone","nothing","somebody","someone","something","thing","things"]) 

	tknzr = TweetTokenizer()

	trimmed_tweets = [[word for (word, pos) in pos_tag(tknzr.tokenize(tweet)) if len(word) > 1 and word.casefold() not in stop_words and pos[0] == 'N'] for tweet in tweets]
	
	t = trimmed_tweets
	t[:] = [[word.lower() if not match(r"\b[A-Z]{2,}\b", word) else word for word in wordlist ] for wordlist in trimmed_tweets]


	trimmed_tweets_counts = [Counter(wordlist) for wordlist in t]
	
	topics = Counter()
	for c in trimmed_tweets_counts:
		topics.update(c)

	# Counter dict `topics` can be very important. We can put preferences on twitter handles
	# they are complete nouns as opposed to parts of broken-down noun phrases like "graphic"
	# and "novel" which individually do not give the idea of the original phrase.
	# A large number of handles might mean they are connected to their followers better, interactive, etc.
	
	return topics.most_common(count)

def textTokenize(text):
	return sent_tokenize(text)

def group_by_polarity(sentences_and_polarity):
	d1 = defaultdict(list)

	for k, v in sentences_and_polarity:
	    d1[k].append(v)

	return dict((k, list(v)) for (k, v) in d1.items())

auth = OAuthHandler(TWIT_API,TWIT_SECRET)
auth.set_access_token(TWIT_ACCESS_TOKEN,TWIT_ACCESS_SECRET)

api = API(auth)

if __name__ == "__main__":
	# Stream example
	# stream("apple")
	# tweets = getTweets("KUnepal", 60)
	# tw = []
	# for tweet in tweets:
	# 	tw.append(tweet["full_text"])
	# print(tw)
	# print(getTopics(tw,20))
	pass
