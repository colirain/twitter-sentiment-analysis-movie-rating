#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tingyu Li (tl2861)
# ---------------------------

import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import json

ACCESS_TOKEN = '1048005627816169473-5tJdLYafXOfCUaHWDFtcqjCvBRdU90'
ACCESS_SECRET = 'iIQRVbZTN625Zn1xRPwDKkOH8Mk4xqrAEn8ICb39CTIeY'
CONSUMER_KEY = 'iTrUDoKffZNLfSYvc9ctDXGJZ'
CONSUMER_SECRET = 'XEuXkw8AesuyxYaiPG7CSI1r0IhuXhcN8vvTYyimrCoAvMShU4'

tags = ['#AvengersEndgame','#CaptainMarvel','#gameofthrones']

class TweetsListener(StreamListener):
  def __init__(self, csocket):
      self.client_socket = csocket
  def on_data(self, data):
      try:
          msg = json.loads( data )
          print('TEXT:{}\n'.format(msg['text']))
          self.client_socket.send( msg['text'].encode('utf-8') )
          return True
      except BaseException as e:
          print("Error on_data: %s" % str(e))
          return False
      # return True
  def on_error(self, status):
      print(status)
      return False

def sendData(c_socket, tags):
  auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
  auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
  twitter_stream = Stream(auth, TweetsListener(c_socket))
  twitter_stream.filter(track=tags,languages=['en'])
  # for line in twitter_stream.iter_lines():


TCP_IP = "localhost"
TCP_PORT = 9001
conn = None
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((TCP_IP, TCP_PORT))


# In[ ]:

class twitter_client:
  def __init__(self):
    self.s = s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.s.bind((TCP_IP, TCP_PORT))

  def run_client(self, tags):
    try: 
      self.s.listen(1)
      # print("Waiting for TCP connection...")
      while True:
        print("Waiting for TCP connection...")
        conn, addr = self.s.accept()
        print("Connected... Starting getting tweets.")
        sendData(conn,tags)
        # http_resp = resp
        # tcp_connection = conn
        # for line in http_resp.iter_lines():
        #     try:
        #         full_tweet = json.loads(line)
        #         tweet_text = full_tweet['text']
        #         print("Tweet Text: " + tweet_text)
        #         print ("------------------------------------------")
        #         tweet_text += '\n'
        #         tcp_connection.send(tweet_text.encode())
        #     except:
        #         e = sys.exc_info()
        #         print("Raise an Error:")
        #         print(e)
        # self.s.close()
        conn.close()
    except KeyboardInterrupt:
      exit 

# try: 
#     s.listen(1)
#     print("Waiting for TCP connection...")
#     conn, addr = s.accept()
#     print("Connected... Starting getting tweets.")
#     sendData(conn)
#     # http_resp = resp
#     # tcp_connection = conn
#     # for line in http_resp.iter_lines():
#     #     try:
#     #         full_tweet = json.loads(line)
#     #         tweet_text = full_tweet['text']
#     #         print("Tweet Text: " + tweet_text)
#     #         print ("------------------------------------------")
#     #         tweet_text += '\n'
#     #         tcp_connection.send(tweet_text.encode())
#     #     except:
#     #         e = sys.exc_info()
#     #         print("Raise an Error:")
#     #         print(e)
# except KeyboardInterrupt:
#     s.close()
#     exit 

if __name__ == '__main__':
  client = twitter_client()
  client.run_client(tags)
