import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

pd.options.mode.chained_assignment = None

sentence = "This is a good product"

blob = TextBlob(sentence)
# print(blob.sentiment)

reviewDataSet = pd.read_csv(r'data/redmi6.csv', sep=',', encoding='cp1252')

# print(reviewDataSet.dtypes)

subData = reviewDataSet[["Review Title", "Rating", "Category", "Useful"]]

print(subData.shape)

subData.Useful = subData.Useful.astype(str).str[0]
subData.Useful = subData.Useful.replace(" ", 0)
subData.Useful = subData.Useful.replace("n", 0)
subData.Useful = subData.Useful.replace("O", 1)

subData.Useful = subData.Useful.astype('int64')
print(subData.dtypes)
