import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

sentence = "This is a good product"

blob = TextBlob(sentence)
print(blob.sentiment)

reviewDataSet = pd.read_csv(r'data/redmi6.csv', sep='\t')



