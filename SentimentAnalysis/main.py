import pandas as pd
from textblob import TextBlob

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
subData.Useful = subData.Useful.astype(str).str[:2]
subData.Useful = subData.Useful.replace(" ", 0)
subData.Useful = subData.Useful.replace("na", 0)
subData.Useful = subData.Useful.replace("On", 1)
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

# using Comments get the sentiment
cPolarity = []

for i in range(0, subData.shape[0]):
    cScore = TextBlob(subData.iloc[i][2])
    cScoreUp = cScore.sentiment[0]
    cPolarity.append(cScoreUp)

# Add Comments Sentiment to table
subData = pd.concat([subData, pd.Series(cPolarity)], axis=1)
subData.rename(columns={subData.columns[5]: "C_Sentiment"}, inplace=True)

# subData.to_csv("data/edited.csv")

# Find True Sum of Sentiment & Rating Sum (Using Useful Column also)
tSum = 0  # Review Title Sentiment Sum
cSum = 0  # Comments Sentiment Sum
rSum = 0  # Rating Sum
for i in range(0, subData.shape[0]):
    mulValue = subData.iloc[i][3] + 1
    tSum = tSum + (subData.iloc[i][4] * mulValue)
    cSum = cSum + (subData.iloc[i][5] * mulValue)
    rSum = rSum + (subData.iloc[i][1] * mulValue)

# Calculate Average of Sums
divideFactor = subData["Useful"].sum() + subData.shape[0]  # To Find the Average this is the dividing Factor
avgTSum = tSum / divideFactor
avgCSum = cSum / divideFactor
avgTC = (avgTSum + avgCSum) / 2  # Average of TSum & CSum
avgRating = rSum / divideFactor  # Average of the Rating

# Convert Sentiment Value to Rating Value
predict_Rating = 0
if avgTC > 0:
    predict_Rating = (3 + (avgTC * 2))
elif avgTC < 0:
    predict_Rating = (3 - (2 + (avgTC * 3)))
elif avgTC == 0:
    predict_Rating = 3

# Printout the Target Values
print("Average_Rating = ", avgRating)
print("predict_Rating = ", predict_Rating)
print("Accuracy = ", (predict_Rating / avgRating) * 100, "%")

# subData.to_csv("data/edited.csv")
