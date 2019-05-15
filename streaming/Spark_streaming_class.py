
# coding: utf-8

# In[ ]:

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row,SQLContext,SparkSession
import sys
import requests
from pyspark.sql.types import *
from pyspark.sql.functions import *
import pandas as pd
import time
import numpy as np 
from sklearn.externals import joblib
from datetime import datetime
import random

import pickle
import re
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from collections import deque

import csv

import tensorflow as tf

ALPHA = 0.5

tags = ['#AvengersEndgame','#LaLlorona','#CaptainMarvel','#Breakthrough','#gameofthrones']

tags_nohash = ['AvengersEndgame','CaptainMarvel','gameofthrones']

columns = ['movie','time','score']
# score_df = pd.DataFrame({'hastag':[],'time':[],'score':[]})
columns_elo = ['movie', 'rank']
# load model

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stop_words = stopwords.words('english')
stemmer = SnowballStemmer("english")

with open('model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    # recorver model
    #lstm = LSTMs("embedding_matrix.npy").model
lstm = load_model('model/model.h5')
graph = tf.get_default_graph()

def aggregate_tags_count(new_values, total_sum):
    return sum(new_values) + (total_sum or 0)

def updateFunction(newValues, runningCount):
    if runningCount is None:
        runningCount = 0
    return sum(newValues, runningCount)

def get_sql_context_instance(spark_context):
    if ('sqlContextSingletonInstance' not in globals()):
        globals()['sqlContextSingletonInstance'] = SQLContext(spark_context)
    return globals()['sqlContextSingletonInstance']

def process_rdd(time, rdd):
    print("----------- %s -----------" % str(time))
    if not rdd.isEmpty():
        try:
            # Get spark sql singleton context from the current context
            sql_context = get_sql_context_instance(rdd.context)
            rdd = rdd.reduceByKey(lambda a,b: a+b)
            row_rdd = rdd.map(lambda w: Row(hashtag=w[0], hashtag_count=w[1]))
            hashtags_df = sql_context.createDataFrame(row_rdd)
            # Register the dataframe as table
            hashtags_df.registerTempTable("hashtags")
            # get the top 10 hashtags from the table using SQL and print them
            hashtag_counts_df = sql_context.sql("select hashtag, hashtag_count from hashtags order by hashtag_count desc limit 10")
            print(hashtag_counts_df.show())
            pandas_df = hashtag_counts_df.toPandas()
            pandas_df.to_csv('hashtag_count.csv', header=False, index = False)
        except Exception as e:
            print(e)
        finally:
            try:
                hashtag_counts_df.show()
            except:
                pass
    else:
        print("-----this rdd is empty-------")

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    # convert all @username to "AT_USER"
    text = re.sub('@[^\s]+','AT_USER', text)
    # correct all multiple white spaces to a single white space
    text = re.sub('[\s]+', ' ', text)
    # convert "#topic" to just "topic"
    text = re.sub(r'#([^\s]+)', r'\1', text)
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

def sentiment_analysis(input_data):
    """
    :model_name : list
    :input_data : list[string]
    """
    clean_text = [preprocess(text) for text in input_data]
    x_text = pad_sequences(tokenizer.texts_to_sequences(clean_text), maxlen=300)
    # print("--------xtext is-----\n{}".format(x_text))
    with graph.as_default():
        score = lstm.predict(x_text)
    return np.mean(score)

# In[ ]:
def save_file(file, data):
    with open(file, 'a+') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(data)

def sentiment_Rdd(i, tag):
    """
    Do sentiment analysis, moving average and elo score
    """
    def _sentiment_Rdd(time_rdd, rdd):
        try:
            if not rdd.isEmpty():
                    real_data = rdd.collect() # data that pass to the model
                    if len(real_data) != 0:
                        score_now = sentiment_analysis(real_data)
                        save_score_now(score_now, tag)
                        cal_moving_average(score_now, tag)
                        cal_elo(score_now, i, tag)
                    else:
                        print('----empty dataframe-----')
            else:
                print("empty rdd")
        except Exception as e:
            print(e)
    return _sentiment_Rdd

def save_score_now(score_now, tag):
    try:
        score_df = pd.read_csv("{}.csv".format(tag))
    except Exception as e:
        score_df = pd.DataFrame(columns = columns)
    time_now = datetime.now()
    temp_df = pd.DataFrame([[tag,str(time_now),score_now]],columns = columns)
    score_df = score_df.append(temp_df, ignore_index=True)
    # print("-----score_now-----\n{}".format(score_df.head()))
    score_df.to_csv('{}.csv'.format(tag), sep=',',index=False)
    # print('success save-------')

def write_data_to_file(data, tag):
    """data: rdd"""
    sql_context = get_sql_context_instance(data.context)
    row_rdd = data.map(lambda x: Row(text = x))
    df = sql_context.createDataFrame(row_rdd)
    pandas_df = df.toPandas()
    pandas_df.to_csv('{}_text.csv'.format(tag),header=False, index = False)

def cal_moving_average(score_now, tag):
    try:
        score_df = pd.read_csv("{}_avg.csv".format(tag))
        # print("read old success")
    except Exception as e:
        # print("error {}".format(e))
        # print("new csv")
        score_df = pd.DataFrame(columns = columns)
    # print("score_now={}".format(score_now))
    try:
        score_before = score_df['score'].iloc[-1]
        # score_now = random.randint(0,5)
        score = ALPHA*score_before + (1-ALPHA)*score_now
    except Exception as e:
        print(e)
        score = score_now
    # time_now = time.time()
    time_now = datetime.now()
    temp_df = pd.DataFrame([[tag,str(time_now),score]],columns = columns)
    score_df = score_df.append(temp_df, ignore_index=True)
    score_df.to_csv('{}_avg.csv'.format(tag), sep=',',index=False)
    # print('success save-------')

def cal_elo(scoreA, i, tag):
    # print('--------start cal elo----------')
    try:
        eloDF = pd.read_csv("elo.csv")
        # print("read old success")
    except Exception as e:
        print("error {}".format(e))
        # print("new csv")
        eloDF = pd.DataFrame(columns = columns_elo)
    # print('----eloDF----{}'.format(eloDF))
    try:
        RA = eloDF.loc[eloDF['movie'] == tag][1]
    except Exception as e:
        # print("RA error {}".format(e))
        RA = 1400
    try:
        # movie_list = eloDF['movie'].to_list()
        movie_before_loc = int(eloDF[eloDF['movie'] == tag].index[0]) - 1
        # print('---movie before loc--{}'.format(movie_before_loc))
        movie_before = eloDF.iloc[movie_before_loc]
        # print(movie_before)
        if movie_before[0] != tag:
            movieB_name = movie_before[0]
            RB = movie_before[1]
            sbdf = pd.read_csv("{}.csv".format(movie_before[0]))
            scoreB = sbdf['score'].iloc[-1]
        else:
            RB = 1400
            scoreB = 0.5
            movieB_name = None
    except Exception as e:
        # print('rb error {}'.format(e))
        RB = 1400
        scoreB = 0.5
        movieB_name = None
    RA_new = cal_elo_a_b(RA, RB, scoreA, scoreB)
    temp_df = pd.DataFrame([[tag, RA_new]],columns = columns_elo)
    eloDF = eloDF.append(temp_df, ignore_index=True)
    if movieB_name != None:
        RB_new = cal_elo_a_b(RB, RA, scoreB, scoreA)
        temp_df = pd.DataFrame([[movieB_name, RB_new]],columns = columns_elo)
        eloDF = eloDF.append(temp_df, ignore_index=True)
    eloDF.drop_duplicates(subset = 'movie', keep = 'last', inplace = True)
    eloDF = eloDF.sort_values(by = ['rank'], ascending = False)
    print(eloDF)
    eloDF.to_csv('elo.csv', sep=',',index=False)
    # print ('save success')


def cal_elo_a_b(RA, RB, scoreA, scoreB):
    """return the ranking of A based on B"""
    K = 10
    if scoreA > scoreB:
        S = 1
    elif scoreA == scoreB:
        S = 0.5
    else:
        S = 0
    E = 1/(1 + 10**((RA - RB)/400))
    RA_new = int(RA + 10*(S - E))
    return RA_new

def count_throughput(time, rdd):
    rdd = rdd.map(lambda x: (time, 1))
    rdd = rdd.reduceByKey(lambda x,y: x+y)
    lines = rdd.collect()
    print(lines)
    with open('throughput_2.csv', 'a+') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)


class SparkStreaming:
    def __init__(self):
        # create spark configuration
        conf = SparkConf()
        conf.setMaster('local[2]')
        conf.setAppName("TwitterStreamApp")
        # create spark context with the above configuration
        self.sc = SparkContext(conf=conf)
        self.sc.setLogLevel("ERROR")
        # sc.setSystemProperty('spark.executor.memory','8g')
        # spark = SparkSession(sc)
        # create the Streaming Context from the above spark context with interval size 2 seconds
        self.ssc = StreamingContext(self.sc, 10)
        # setting a checkpoint to allow RDD recovery
        self.ssc.checkpoint("checkpoint_TwitterApp")

    def run_stream(self):
        dataStream = self.ssc.socketTextStream("localhost",9001)
        dataStream.pprint()
        dataStream.foreachRDD(count_throughput)
        # split each tweet into words
        words = dataStream.flatMap(lambda line: line.split(" "))
        hashtags = words.filter(lambda w: '#' in w).map(lambda x: (x.lower(), 1))
        hashtags.foreachRDD(process_rdd)
        for i, tag in enumerate(tags_nohash):
        # for tag in tags_nohash:
            print("-------{}------{}------".format(i, tag))
            filteredStream = dataStream.filter(lambda x: tag in x)
            filteredStream.foreachRDD(sentiment_Rdd(i, tag))

        self.ssc.start()
        # wait for the streaming to finish
        self.ssc.awaitTermination()


if __name__ == "__main__":
    sparkstreaming = SparkStreaming()
    sparkstreaming.run_stream()

