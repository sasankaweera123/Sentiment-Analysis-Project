import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords as st
from textblob import TextBlob
from wordcloud import WordCloud

pd.options.mode.chained_assignment = None


# check textblob sentiment
def check_bob(text):
    return TextBlob(text).sentiment.polarity


# Upload data set to data frame
def create_data_set():
    review_data_set = pd.read_csv(r'data/redmi6.csv', sep=',', encoding='cp1252')
    return review_data_set


# print(subData.shape)
def data_cleaning(data_frame):
    # Useful Column Clean
    data_frame.Useful = data_frame.Useful.astype(str).str[:2]
    data_frame.Useful = data_frame.Useful.replace(" ", 0)
    data_frame.Useful = data_frame.Useful.replace("na", 0)
    data_frame.Useful = data_frame.Useful.replace("On", 1)
    data_frame.Useful = data_frame.Useful.astype('int64')

    # Rating Column Clean
    data_frame.Rating = data_frame.Rating.astype(str).str[0] + ".0"
    data_frame.Rating = data_frame.Rating.astype('float')

    return data_frame


# Add new column to the data frame
def add_new_column(data_frame, column_name, column_data, column_number):
    data_frame = pd.concat([data_frame, pd.Series(column_data)], axis=1)
    data_frame.rename(columns={data_frame.columns[column_number]: column_name}, inplace=True)

    return data_frame


# Calculate Polarity of the given column
def calculate_polarity(data_format, raw_number):
    polarity = []
    for i in range(0, data_format.shape[0]):
        polarity.append(TextBlob(data_format.iloc[i][raw_number]).sentiment[0])
    return polarity


# Find True Sum of Sentiment & Rating Sum (Using Useful Column also)
def calculate_sum(data_format):
    rt_sum = 0  # Review Title Sentiment Sum
    c_sum = 0  # Comments Sentiment Sum
    r_sum = 0  # Rating Sum
    for i in range(0, data_format.shape[0]):
        multiply_value = data_format.iloc[i][3] + 1
        rt_sum += (data_format.iloc[i][4] * multiply_value)
        c_sum += (data_format.iloc[i][5] * multiply_value)
        r_sum += (data_format.iloc[i][1] * multiply_value)

    return rt_sum, c_sum, r_sum


# Calculate Average of Sums
def calculate_average(sum_data, divide_factor):
    t_sum, c_sum, r_sum = sum_data
    avg_rt_sum = t_sum / divide_factor
    avg_c_sum = c_sum / divide_factor
    avg_rt_c = (avg_rt_sum + avg_c_sum) / 2  # Average of TSum & CSum
    avg_rating = r_sum / divide_factor  # Average of the Rating

    return avg_rt_c, avg_rating


# Convert Sentiment Value to Rating Value
def convert_sentiment_to_rating(avg_rt_c):
    predict_rating = 0
    if avg_rt_c > 0:
        predict_rating = (3 + (avg_rt_c * 2))
    elif avg_rt_c < 0:
        predict_rating = (3 - (2 + (avg_rt_c * 3)))
    elif avg_rt_c == 0:
        predict_rating = 3
    return predict_rating


# Printout the Target Values
def print_data(avg_rating, predict_rating):
    print("Average Rating: ", avg_rating)
    print("Predicted Rating: ", predict_rating)
    print("Accuracy = ", (predict_rating / avg_rating) * 100, "%")


# Create Word Cloud from the given data
def create_word_cloud(data_frame):
    # Create Positive cloud image Using Review Title
    positive_data_frame = data_frame[data_frame.T_Sentiment > 0]
    cloud = WordCloud(max_words=250, stopwords=st.words("english")).generate(str(positive_data_frame["Review Title"]))
    plt.Figure(figsize=(10, 10))
    plt.imshow(cloud)

    image = cloud.to_image()
    image.show()


# Main Function
def main():

    # Create Data Frame
    data_frame = create_data_set()[["Review Title", "Rating", "Comments", "Useful"]]
    data_frame = data_cleaning(data_frame)

    # using review title get the sentiment
    # Add Review Title Sentiment to table
    data_frame = add_new_column(data_frame, "T_Sentiment", calculate_polarity(data_frame, 0), 4)

    # using Comments get the sentiment
    # Add Comments Sentiment to table
    data_frame = add_new_column(data_frame, "C_Sentiment", calculate_polarity(data_frame, 2), 5)

    # To Find the Average this is the dividing Factor
    divide_factor = data_frame["Useful"].sum() + data_frame.shape[0]

    # Calculate Sum of Sentiment & Rating Sum
    avg_title_comment, avg_rating = calculate_average(calculate_sum(data_frame), divide_factor)

    # Convert Sentiment Value to Rating Value and Print
    print_data(avg_rating, convert_sentiment_to_rating(avg_title_comment))

    # Create csv file from the data frame
    data_frame.to_csv("data/edited.csv")

    # Create Word Cloud
    create_word_cloud(data_frame)


# Call Main Function
if __name__ == '__main__':
    main()
