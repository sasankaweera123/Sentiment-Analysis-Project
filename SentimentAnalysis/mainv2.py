import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

pd.options.mode.chained_assignment = None

sentence = "This is a good product"

blob = TextBlob(sentence)
# print(blob.sentiment)
# Upload data set
reviewDataSet = pd.read_csv(r'data/redmi6.csv', sep=',', encoding='cp1252')

# print(reviewDataSet.dtypes)
# Crete a query
subData = reviewDataSet[["Review Title", "Rating", "Comments", "Useful"]]

# print(subData.shape)

# Useful Column Clean
subData.Useful = subData.Useful.astype(str).str[0]
subData.Useful = subData.Useful.replace(" ", 0)
subData.Useful = subData.Useful.replace("n", 0)
subData.Useful = subData.Useful.replace("O", 1)
subData.Useful = subData.Useful.astype('int64')

# Rating Column Clean
subData.Rating = subData.Rating.astype(str).str[0] + ".0"
subData.Rating = subData.Rating.astype('float')

# using review title get the sentiment
tPolarity = []

for i in range(0, subData.shape[0]):
    score = TextBlob(subData.iloc[i][0])
    scoreUp = score.sentiment[0]
    tPolarity.append(scoreUp)

# Add Review Title Sentiment to table
subData = pd.concat([subData, pd.Series(tPolarity)], axis=1)
subData.rename(columns={subData.columns[4]: "T_Sentiment"}, inplace=True)
# subData.to_csv("data/edited.csv")

# using Comments get the sentiment
cPolarity = []

for i in range(0, subData.shape[0]):
    cScore = TextBlob(subData.iloc[i][2])
    cScoreUp = cScore.sentiment[0]
    cPolarity.append(cScoreUp)

# Add Comments Sentiment to table
subData = pd.concat([subData, pd.Series(cPolarity)], axis=1)
subData.rename(columns={subData.columns[5]: "C_Sentiment"}, inplace=True)

avgRating = (subData['Rating'].sum()) / subData.shape[0]

# Printout the Necessary details
print("Average Rating in Data set", avgRating)
print("Comment Sentiment Value = ", subData["C_Sentiment"].sum() / subData.shape[0])
print("Review Title Sentiment Value =  ", subData["T_Sentiment"].sum() / subData.shape[0])
avg_Sentiment = (((subData["C_Sentiment"].sum() + subData["T_Sentiment"].sum() )/ subData.shape[
    0])) / 2
print("Average Sentiment =", avg_Sentiment)

# Convert Sentiment Value to Rating Value
predict_Rating = 0
if avg_Sentiment > 0:
    predict_Rating = (3 + (avg_Sentiment * 2))
elif avg_Sentiment < 0:
    predict_Rating = (3 - (2 + (avg_Sentiment * 3)))
elif avg_Sentiment == 0:
    predict_Rating = 3

# Printout the Target Values
print("predict_Rating = ", predict_Rating)
print("Accuracy = ", (predict_Rating/avgRating) * 100 , "%")


