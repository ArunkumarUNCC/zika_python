# README
# IN THIS PYTHON FILE, I PLAYED AROUND SOME BASIC STUFF LIKE
#       1. CLEANING ALL HASHTAGS - CONVERTING HASTAGS TO LOWERCASES AND CONVERTING TIMESTAMPS TO YYYY-MM-DD hh:mm:ss FORMAT - MORE CLEANING CAN BE INCORPORATED
#       2. CREATING WORDCLOUD FOR ALL HASHTAGS
#       3. HISTOGRAM FOR NUMBER OF TWEETS PER HOUR/DAY
#       4. GETTING TIME SERIES DATA FOR EACH HASHTAG LIKE
#               i) FIRST ALL DAYS ARE FIRST EXTRACTED AND CREATED COLUMNS
#               ii) EXTRACTED ALL HASTAGS
#               iii) ITERATED OVER THE DATA TO CHECK IN WHICH TIME, A HASHTAG IS PRESENT. IF A HASHTAG 'h(i)' IS PRESENT AT A PARTICULAR TIME 't(J)', VALUE OF 'h(i)' AT 't(j)' IS INCREMENTED ONCE
#               iv) ALL TIME SERIES ARE OUTPUTTED AS A .csv FILE
# APART FROM THESE, DAILY FREQUENCY COUNT OF EACH HASHTAG IS ALSO TAKEN - SINCE THE DATA IS OVER THE SPAN OF 6 MONTHS, K-SC CLUSTERING IS VERY SLOW FOR HOURLY DATA
# README END


from collections import Counter
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import collections
from itertools import groupby
import json


# Function to clean tweets
# Currently it converts hashtags to lowercases and timestamps to the desired format
def clean_tweets(uncleaned_tweets_filepath,cleaned_hashtags_filepath):
    tweet_df = pd.read_csv(uncleaned_tweets_filepath)

    # Converting all hastags to lowercase
    tweet_df["Hashtags"] = tweet_df["hashtags"].apply(lambda string: str(string)).apply(lambda string: string.lower())
    # Parsing the posted time of tweets to adesired format
    tweet_df["Created_At"] = tweet_df["postedTime"].apply(
        lambda x: "" if isinstance(x, float) else "" if len(x) < 10 else time.strftime('%Y-%m-%d',time.strptime(x, '%Y-%m-%dT%H:%M:%S.000Z'))
    )

    cleaned_hashtags_df = tweet_df.loc[:,["Hashtags","Created_At"]]

    # Saving the dataframe as a .csv file
    cleaned_hashtags_df.to_csv(cleaned_hashtags_filepath, index=False, header=True)

# Function to print total number of unique hashtags available in the given dataset
def get_unique_hashtags(cleaned_hashtags_filepath):
    tweet_df = pd.read_csv(cleaned_hashtags_filepath,encoding="ISO-8859-1")

    # Getting all hashtags - duplicates included
    hashtags = get_hashtags(tweet_df["Hashtags"].tolist())

    # Getting frequency of each hashtags occurring in all time series
    hashtag_counts = dict(Counter(hashtags))

    print("Total Hashtags: " + str(len(hashtag_counts.keys())))

# Function to retreive all hashtags from all tweets
# Not filtering out the duplicates
def get_hashtags(df_hashtags):
    hashtags = []
    for all_hashtags in df_hashtags:
        if not isinstance(all_hashtags,float):
            single_tweet_hashtags = all_hashtags.split(',')
            if single_tweet_hashtags[0] != 'nan':
                for hashtag in single_tweet_hashtags:
                    hashtags.append(hashtag)

    return hashtags

# Create and display the Wordcloud for hashtags obtained from tweets
def create_wordcloud(hashtags):
    hashtag_counts = dict(Counter(hashtags))
    print(type(hashtag_counts))
    wc = WordCloud(width=1500, height=800, max_words=2000).generate_from_frequencies(hashtag_counts)

    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Function that gets all hours from the available timestamp
def get_hours(created_at_df):
    times_for_plot = []

    # Getting only the date and hours
    for str_time in created_at_df:
        times_for_plot.append(str_time.split(" ")[0])

    # Sorting the dictionary by dates
    times_counter_per_hour = collections.OrderedDict(sorted(Counter(times_for_plot).items(), key=lambda t: t[0]))

    return times_counter_per_hour


def plot_timeseries_plot(cleaned_hashtags_file):
    tweet_df = pd.read_csv(cleaned_hashtags_file,encoding="ISO-8859-1")

    # Getting all hours from all available dates
    filtered_times = tweet_df[pd.notnull(tweet_df["Created_At"])]
    hours = get_hours(filtered_times["Created_At"])

    # UNCOMMENT THE FOLLOWING TO PLOT BAR CHART OF TWEETS COUNT / HOUR
    # Plotting a bar chart for tweets count for every hour
    tweet_count_plot(hours)

# Creating a time series plot for number of tweets for every 30 minutes
def tweet_count_plot(times_counter_per_hour):
    time_count_tuples = times_counter_per_hour.items()
    x,y = zip(*time_count_tuples)
    plt.plot(range(len(x)),y)
    plt.xlabel("Time ID")
    plt.ylabel("Tweets Count")
    plt.show()

# Function to parse the given file in a desired format to get frequencies for each hashtag
def get_frequencies_for_hashtags(cleaned_hashtags_file,hourly_frequency_file):
    hashtags_timestamps_df = pd.read_csv(cleaned_hashtags_file,encoding="ISO-8859-1")

    # Getting date with hours alone from the Created_At column
    # hashtags_timestamps_df["Created_At"] = hashtags_timestamps_df["Created_At"].apply(
    #     lambda timestamp : "" if isinstance(timestamp, float) else timestamp.split(":")[0]
    # )

    # Getting unique timestamps from the dataframe
    unique_timestamps = hashtags_timestamps_df["Created_At"].unique()
    unique_timestamps = unique_timestamps[~pd.isnull(unique_timestamps)]

    timestamp_dict = {}
    # Creating index for each timestamp
    for index, timestamp in enumerate(unique_timestamps):
        timestamp_dict[timestamp] = index

    timestamp_indices = sorted(list(timestamp_dict.values()))
    timestamp_indices.insert(0,"Words")

    # Replacing timestamps with assigned index
    hashtags_timestamps_df["Created_At"] = hashtags_timestamps_df["Created_At"].apply(
        lambda timestamp: "" if isinstance(timestamp, float) else timestamp_dict[timestamp]
    )

    hashtags_with_hourly_counts_df = get_perhour_hashtag_count(hashtags_timestamps_df,timestamp_indices)

    hashtags_with_hourly_counts_df.to_csv(hourly_frequency_file,index=False,header=True,encoding='utf-8')

# Function to return hashtag counts/day
def get_perhour_hashtag_count(hashtags_df,timestamp_indices):
    hashtags_timestamps = []
    timestamps_for_hashtag = {}
    hashtags_with_hourly_counts = {}

    hashtag_frequency_df = pd.DataFrame(columns=timestamp_indices)
    hashtag_frequency_df.fillna(0.0)

    # Iterating over the provided dataframe
    # Inserting tuples of type (hashtag,timestamp) into a list
    for index,row in hashtags_df.iterrows():
        if isinstance(row["Hashtags"],float) or isinstance(row["Created_At"],float) or row["Created_At"]=="":
            continue
        else:
             row_hashtags_list = row["Hashtags"].split(",")

             for hashtag in row_hashtags_list:
                 hashtags_timestamps.append((hashtag,row["Created_At"]))

    # Sorting all tuples to use them in groupby function
    # Sorting by the key element
    sorted_tuples = sorted(hashtags_timestamps,key = lambda tuple : tuple[0])

    # Using group by key method to collect all timestamps related to a particular hashtag
    # Collecting all hastags and their corresponding timestamps in a dictionary
    for key,values in groupby(sorted_tuples,lambda x:x[0]):
        timestamps = []

        # Here, the value is again an array of tuples of type: (key,timestamp)
        for value in values:
            timestamps.append(value[1])
        timestamps_for_hashtag[key] = timestamps

    row_list = []

    # Counting each timestamps and storing as sorted dictionary
    # Filtering out hashtags that occurs only once
    for key in timestamps_for_hashtag:
        counts = collections.OrderedDict(sorted(Counter(timestamps_for_hashtag[key]).items(), key=lambda t: t[0]))

        if len(counts.keys()) > 1:
            counts["Words"] = key
            row_list.append(counts)

    hashtag_frequency_df = pd.DataFrame(row_list,columns=timestamp_indices)
    # Used the following code line to remove all columns in the end that contains only empty values
    hashtag_frequency_df = hashtag_frequency_df.dropna(axis=1,how='all')
    hashtag_frequency_df = hashtag_frequency_df.fillna(0)
    hashtag_frequency_df = hashtag_frequency_df.astype(float,raise_on_error=False)


    return hashtag_frequency_df


# Function that call all functions
if __name__ == "__main__":
    # FILEPATHS
    uncleaned_tweets_file = "zika_jj_cols/zika_jj_cols.csv"
    cleaned_hashtags_file = "zika_jj_cols/zika_jj_cols_Cleaned.csv"
    hastags_hourly_frequency_filepath = "zika_jj_cols/zika_jj_cols_hourly_frequencies.csv"
    hastags_daily_frequency_filepath = "zika_jj_cols/zika_jj_cols_daily_frequencies.csv"

    # UNCOMMENT THE FOLLOWING TO CLEAN THE .CSV FILE AND SAVE AGAIN
    clean_tweets(uncleaned_tweets_file,cleaned_hashtags_file)

    # UNCOMMENT THE FOLLOWING LINE TO PLOT A HISTOGRAM OF HOURLY FREQUENCY OF TWEETS
    plot_timeseries_plot(cleaned_hashtags_file)

    # UNCOMMENT THE FOLLOWING TO GET TOTAL NUMBER OF UNIQUE HASHTAGS AVAILABLE IN THE GIVEN DATA
    # get_unique_hashtags(cleaned_hashtags_file)

    # UNCOMMENT THE FOLLOWING LINES TO GET HOURLY FREQUENCY OF EACH HASHTAGS
    # TODO: NUMBER OF HASHTAGS CAN BE REDUCED BY SETTING THE FREQUENCY LIMITS
    get_frequencies_for_hashtags(cleaned_hashtags_file,hastags_daily_frequency_filepath)

    # final_df = pd.read_csv(hastags_daily_frequency_filepath,encoding="ISO-8859-1")
    # print(len(final_df.columns))



    # Setting a frequency to filter out all hashtags that occur less than this frequency
    # user_frequency = 1000
    # List to store hashtags that support user_frequency
    # valid_hashtags = []


    # UNCOMMENT THE FOLLOWING CODE TO GET VALID HASHTAGS WHOSE FREQUENCY IS GREATER THAN THE GIVEN FREQUENCY
    # # Getting all hashtags - duplicates included
    # hashtags = get_hashtags(tweet_df["Hashtags"].tolist())
    #
    # # Getting frequency of each hashtags occurring in all time series
    # hashtag_counts = dict(Counter(hashtags))
    # for count in hashtag_counts:
    #     if hashtag_counts[count] > 1000:
    #         valid_hashtags.append(count)

    # UNCOMMENT THE FOLLOWING TO GENERATE WORDCLOUD FOR ALL HASHTAGS IN THE DATASET
    # # Generating WordCloud for the retrieved hastags
    # # Input hashtags from the previous step
    # create_wordcloud(hashtag_counts)
