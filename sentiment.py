from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores("I love Ryerson")
print "pos score: ",ss['pos']
print "neg score: ",ss['neg']
print "neutral score: ",ss['neu']
