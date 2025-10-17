# utils.py
from pythainlp.tokenize import word_tokenize

def custom_tokenizer(text):
    return word_tokenize(text, engine='newmm')