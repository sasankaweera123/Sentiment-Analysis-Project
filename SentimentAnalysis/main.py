from textblob import TextBlob

sentence = "This is a good product"

blob = TextBlob(sentence)
print(blob.sentiment.polarity)
