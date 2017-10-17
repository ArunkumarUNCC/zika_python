# README
'''
THIS SCRIPT HAS TO BE RUN ON THE OUTPUT OF K-SC CLUSTERING
K-SC CLUSTERING PROGRAM OUTPUTS .csv FILE WITH COLUMNS=[WORDS,CLUSTER_ID]

1. THIS PROGRAM TAKES CLUSTERED WORDS AND TIMESERIES DATA INPUTS
2. FOR EACH WORD 'w' FROM THE CLUSTER, SCAN THE TIMESERIES DATA FOR 'w' AND GET FREQUENCY OF 'w' BY SUMMING UP ALL VALUES FROM ALL TIMESTAMPS
3. ONCE GETTING FREQUENCIES OF ALL WORDS, THE RESULT IS WRITTEN INTO A .csv FILE WITH COLUMNS OF [WORDS,CLUSTER_ID,SUM,]
4. NEXT THE PROGRAM GENERATES WORDCLOUDS FOR EACH CLUSTER BASED ON FREQUENCIES
'''
# README END

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def get_sum_of_hashtags(hashtags_daily_frequencies_filepath,hashtags_clusters_filepath,hashtags_cluster_sum_filepath):
    clusters_df = pd.read_csv(hashtags_clusters_filepath, encoding="ISO-8859-1")
    daily_frequencies_df = pd.read_csv(hashtags_daily_frequencies_filepath, encoding="ISO-8859-1")

    # Getting unique cluster ids
    unique_cluster_ids = clusters_df["Cluster"].unique().tolist()

    # Getting unique column names from the hashtag frequencies file
    # Getting first time label and last time label
    unique_time_labels = list(daily_frequencies_df.columns.difference(["Words"]))
    unique_time_labels = [int(i) for i in unique_time_labels]

    first_time_label = str(min(unique_time_labels))
    last_time_label = str(max(unique_time_labels))

    total_hashtags_sum = dict.fromkeys(unique_time_labels)
    wordcloud_frequencies_df = pd.DataFrame(columns=["Words", "Cluster", "Sum"])

    # Iterating hashtags from each cluster and finding sum of each hashtag
    for id in unique_cluster_ids:
        current_cluster_df = clusters_df.loc[clusters_df["Cluster"] == id]

        for hashtag in current_cluster_df["Words"]:
            time_series_df = daily_frequencies_df.loc[daily_frequencies_df["Words"] == hashtag]
            sum = time_series_df.loc[:, first_time_label:last_time_label].sum(axis=1)
            # time_series_df['Sum'] = time_series_df[unique_time_labels].sum(axis=1)

            if len(sum.values) == 1:
                row = [hashtag, id, sum.values[0]]

                wordcloud_frequencies_df.loc[len(wordcloud_frequencies_df)] = row

    wordcloud_frequencies_df.to_csv(hashtags_cluster_sum_filepath, index=False, header=True, encoding='utf-8')


def get_wordclouds(hashtags_cluster_sum_filepath):
    df = pd.read_csv(hashtags_cluster_sum_filepath,encoding="ISO-8859-1")
    unique_cluster_ids = df["Cluster"].unique().tolist()
    print(df.head(10))

    for id in unique_cluster_ids:
        current_frequencies_df = df.loc[df["Cluster"] == id]
        wordcloud_dict = {}

        for index, row in current_frequencies_df.iterrows():
            wordcloud_dict[row["Words"]] = row["Sum"]
            # print(type(row["Sum"]))

        # for i in wordcloud_dict:
        wc = WordCloud(width=1500, height=800).generate_from_frequencies(wordcloud_dict)

        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()

# Function that call all functions
if __name__ == "__main__":
    hashtags_daily_frequencies_filepath = "zika_jj_cols/zika_jj_cols_daily_frequencies_new.csv" # Columns: Hashtags,1,2,3,4,5,.....
    hashtags_clusters_filepath = "zika_jj_cols/new_daily_frequency_clusters/Words_Daily_Frequencies_2Clusters.csv" # Columns: Hashtags, Cluster
    hashtags_cluster_sum_filepath = "zika_jj_cols/new_daily_frequency_clusters/WordCloud_2Clusters.csv"

    get_sum_of_hashtags(hashtags_daily_frequencies_filepath,hashtags_clusters_filepath,hashtags_cluster_sum_filepath)
    get_wordclouds(hashtags_cluster_sum_filepath)
