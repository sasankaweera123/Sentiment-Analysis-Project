import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords as st
from textblob import TextBlob
from wordcloud import WordCloud

pd.options.mode.chained_assignment = None


def check_bob(text):
    return TextBlob(text).sentiment.polarity


# print(blob.sentiment)
# Upload data set
def create_data_set():
    review_data_set = pd.read_csv(r'data/redmi6.csv', sep=',', encoding='cp1252')
    return review_data_set


# Crete a query

sub_data = create_data_set()[["Review Title", "Rating", "Comments", "Useful"]]


# print(subData.shape)
def data_cleaning(subData):
    # Useful Column Clean
    subData.Useful = subData.Useful.astype(str).str[:2]
    subData.Useful = subData.Useful.replace(" ", 0)
    subData.Useful = subData.Useful.replace("na", 0)
    subData.Useful = subData.Useful.replace("On", 1)
    subData.Useful = subData.Useful.astype('int64')

    # Rating Column Clean
    subData.Rating = subData.Rating.astype(str).str[0] + ".0"
    subData.Rating = subData.Rating.astype('float')

    return subData


subData = data_cleaning(sub_data)

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

divideFactor = subData["Useful"].sum() + subData.shape[0]  # To Find the Average this is the dividing Factor


# Calculate Average of Sums
def calculate_average(t_sum, c_sum, r_sum, divide_factor):
    avgTSum = t_sum / divide_factor
    avgCSum = c_sum / divide_factor
    avgTC = (avgTSum + avgCSum) / 2  # Average of TSum & CSum
    avgRating = r_sum / divide_factor  # Average of the Rating

    return avgTC, avgRating


avgTC, avgRating = calculate_average(tSum, cSum, rSum, divideFactor)

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

subData.to_csv("data/edited.csv")

# Create Positive cloud image Using Review Title
pos_subData = subData[subData.T_Sentiment > 0]
cloud = WordCloud(max_words=250, stopwords=st.words("english")).generate(str(pos_subData["Review Title"]))
plt.Figure(figsize=(10, 10))
plt.imshow(cloud)

image = cloud.to_image()
image.show()
