# README
'''
THIS SCRIPT NEEDS THE pyksc PACKAGE INSTALLED.
FOLLOW INSTRUCTIONS FROM 'Steps to install pyksc package.txt' FILE OR https://github.com/flaviovdf/pyksc TO INSTALL pyksc PACKAGE

1. THIS SCRIPT READS THE TIME SERIES DATA AND CREATES 'k' CLUSTERS
2. CLUSTERED WORDS ARE SAVED IN A .csv FILE WITH COLUMNS=[WORDS,CLUSTER]
'''
# README END

import pandas as pd
import numpy as np
from pyksc import ksc
from pyksc import dist

def alter_inputs(input_df):
    # Eliminating the first column - Hashtags
    ksc_input_df = input_df.loc[:,input_df.columns != "Words"]

    # Converting the dataframe into a numpy array
    ksc_input = ksc_input_df.as_matrix()
    return ksc_input

if __name__ == "__main__":
    df = pd.read_csv("new_daily_frequencies/zika_jj_cols_daily_frequencies_new.csv", encoding="iso-8859-1")
    # df = pd.read_csv("Frequencies.csv")

    ksc_input = alter_inputs(df)
    ksc_input = ksc_input.copy(order='C')

    # ksc_input_values = ksc_input.values
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(ksc_input)
    # ksc_input_normalized = pd.DataFrame(x_scaled)

    k = 3 # Number of clusters
    
    centers,assign,series_shifts,dists = ksc.inc_ksc(ksc_input,k)
    
    # Creating a Dataframe to store Hashtags and Cluster
    result_df = pd.DataFrame(columns=['Words','Cluster'])
    result_df['Words'] = df['Words']
    result_df['Cluster'] = assign
    
    result_df.to_csv("new_daily_frequencies/Words_Daily_Frequencies_3Clusters.csv",index=False,encoding='utf-8')
