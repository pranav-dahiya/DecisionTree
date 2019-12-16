import glob
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from multiprocessing import Pool


with open("vocabulary.pickle", "rb") as f:
    vocabulary = pickle.load(f)
words = list(vocabulary.keys())
words.sort()


def create_freq_dict(text):
    """
    creates frequency distribution from input text
    """
    lemmatizer = WordNetLemmatizer()
    freq_dict = {key:0 for key in vocabulary.keys()}
    for line in text:
        words = word_tokenize(line)
        for word in words:
            word = lemmatizer.lemmatize(word.lower())
            if word in vocabulary:
                freq_dict[word] += 1
    return freq_dict


def compute_tf(freq_dict):
    """
    computes tf of input file using input frequency distribution
    """
    count = sum(freq_dict.values()) + (10**(-5))
    tf = {key:freq_dict[key]/count for key in vocabulary.keys()}
    return tf


def compute_idf(filenames):
    """
    computer idf of text in all the files in the input list
    """
    lemmatizer = WordNetLemmatizer()
    num_documents = {key:1 for key in vocabulary.keys()}
    for filename in filenames:
        with open(filename, "r") as f:
            text = f.readlines()
        encountered = []
        for line in text:
            words = word_tokenize(line)
            for word in words:
                word = lemmatizer.lemmatize(word.lower())
                if word in vocabulary and not(word in encountered):
                    num_documents[word] += 1
                    encountered.append(word)
    idf = {}
    for word in vocabulary:
        idf[word] = np.log(len(filenames)/num_documents[word])
    return idf


def tfidf_filename(filename):
    """
    takes in original filename and returns the filename where the tfidf version
    of the email needs to be saved
    """
    temp = filename.split("/")
    temp[-2] += "_tfidf"
    temp[-1] = temp[-1].replace(".txt", ".pickle")
    new_filename = ""
    for dir in temp[:-1]:
        new_filename += dir + "/"
    new_filename += temp[-1]
    return new_filename


def dict_to_array(dict):
    """
    sorts dictionary lexicographically and converts the values to a numpy array
    """
    arr= np.ones(len(dict))
    for i, word in enumerate(words):
        arr[i] = dict[word]
    return arr


def compute_tfidf(filename):
    """
    computes tfidf vector of the text in the given file and saves it
    """
    with open(filename, "r") as f:
        text = f.readlines()
    freq_dict = create_freq_dict(text)
    tf = compute_tf(freq_dict)
    tfidf = {key:tf[key]/idf[key] for key in vocabulary.keys()}
    tfidf = dict_to_array(tfidf)
    with open(tfidf_filename(filename), "wb") as f:
        pickle.dump(tfidf, f)


if __name__ == "__main__":
    filenames = []
    for i in range(1,11):
        filenames.extend(glob.glob("lingspam/part"+str(i)+"/*.txt"))
    global idf
    idf = compute_idf(filenames)
    pool = Pool()
    pool.map(compute_tfidf, filenames)
