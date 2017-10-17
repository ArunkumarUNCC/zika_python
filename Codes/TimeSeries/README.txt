The TimeSeries folder contains python scripts to generate time series data based on hashtags and words from tweets body
Execute time series clustering algorithm on time series data
Generate WordCloud for each time series cluster

Some INFO about python files inside TimeSeries directory:
1. Hashtags.py - Creates timeseries data for all hashtags in the data
2. HashtasgTweets.py - Creates timeseries data for all hashtags and all unique words from a tweet body in the data
3. get_clusters.py - Reads time series data and generates time series clusters
4. parse_clusters.py - Reads the generated clusters and generates WordCloud for each cluster

Some INFO about directories inside TimeSeries directory:
1. Results: 
  a. Contains WordCloud results of timeseries clusters which are produced from the data given by Hashtags.py and HashtagTweets.py programs
