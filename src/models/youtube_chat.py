import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load API key from .env file and set it as the API key
import os

# from dotenv import load_dotenv, find_dotenv

# dotenv_path = find_dotenv()
# load_dotenv(dotenv_path)
openai_key = os.getenv("OPENAI_KEY")
# openai.api_key = openai_key
os.environ["OPENAI_API_KEY"] = openai_key



form langchain import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi my name is John")
print(len(embed))
print(embed[0:5])

import langchain as lc

import tensorflow as tf
import keras as k
import openai as oai
import langchain as lc