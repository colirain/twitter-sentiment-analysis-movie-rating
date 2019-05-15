#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tingyu Li (tl2861)
# ---------------------------

import Spark_streaming_class
from multiprocessing import Process

def spark_streaming():
    spark_streaming = Spark_streaming_class.SparkStreaming()
    spark_streaming.run_stream()

if __name__ == '__main__':
    spark_streaming()
