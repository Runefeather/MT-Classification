# SVM classifier for detecting Machine and Human translated text
# Author: Dhanya
# Date: 6 Feb 2021

# IMPORTS
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import spacy
nlp = spacy.load('en_core_web_sm')

import numpy as np

import io
import re

import jieba

# HELPER FUNCTIONS
# convert label to -1 and 1
def convert_label(x):
    if(x == 'H'):
        return 1
    return -1 

# Determinants
def count_DT(x):
    doc = nlp(x)
    return round(len([w for w in doc if w.pos_ == "DET"])/len_sentence(x), 3)

# Pronouns
def count_PRON(x):
    doc = nlp(x)
    return round(len([w for w in doc if w.pos_ == "PRON"])/len_sentence(x), 3)

# Punctuation marks
def count_PUNC(x):
    doc = nlp(x)
    return round(len([w for w in doc if w.pos_ == "PUNCT"])/len_sentence(x), 3)

# Auxilliary verbs
def count_AUX(x):
    doc = nlp(x)
    return round(len([w for w in doc if w.pos_ == "AUX"])/len_sentence(x), 3)

# Prepositions
def count_PREP(x):
    doc = nlp(x)
    return round(len([w for w in doc if w.pos_ == "ADP"])/len_sentence(x), 3)

# Predeterminers
def count_PDT(x):
    doc = nlp(x)
    return round(len([w for w in doc if w.tag_ == "PDT"])/len_sentence(x), 3)

# Noun phrases
def count_NP(x):
    doc = nlp(x)
    return round(len([n for n in doc.noun_chunks])/len_sentence(x), 3)

# Indirect objects
def count_IOBJ(x):
    doc = nlp(x)
    return round(len([w for w in doc if w.dep_ == "iobj"])/len_sentence(x), 3)

# Nominal subjects
def count_NSUB(x):
    doc = nlp(x)
    return round(len([w for w in doc if w.dep_ == "nsubj"])/len_sentence(x), 3)

# Coordinating Conjunctions
def count_CONJ(x):
    doc = nlp(x)
    return round(len([w for w in doc if w.pos_ == "CCONJ"])/len_sentence(x), 3)

# Length of sentence
def len_sentence(x):
    doc = nlp(x)
    return len([token for token in doc if not token.is_punct | token.is_space])

# Create labels for data using the helper functions defined above
def create_new_labels(df):
    # Reference
    df['Ref_DT'] = df['Reference'].apply(count_DT)
    df['Ref_NP'] = df['Reference'].apply(count_NP)
    df['Ref_pron'] = df['Reference'].apply(count_PRON)
    df['Ref_punc'] = df['Reference'].apply(count_PUNC)
    df['Ref_aux'] = df['Reference'].apply(count_AUX)
    df['Ref_prep'] = df['Reference'].apply(count_PREP)
    df['Ref_predet'] = df['Reference'].apply(count_PDT)
    df['Ref_iobj'] = df['Reference'].apply(count_IOBJ)
    df['Ref_nsub'] = df['Reference'].apply(count_NSUB)
    df['Ref_cconj'] = df['Reference'].apply(count_CONJ)
    df['Ref_len'] = df['Reference'].apply(len_sentence)

    # # Candidate
    df['Cand_DT'] = df['Candidate'].apply(count_DT)
    df['Cand_NP'] = df['Candidate'].apply(count_NP)
    df['Cand_pron'] = df['Candidate'].apply(count_PRON)
    df['Cand_punc'] = df['Candidate'].apply(count_PUNC)
    df['Cand_aux'] = df['Candidate'].apply(count_AUX)
    df['Cand_prep'] = df['Candidate'].apply(count_PREP)
    df['Cand_predet'] = df['Candidate'].apply(count_PDT)
    df['Cand_iobj'] = df['Candidate'].apply(count_IOBJ)
    df['Cand_nsub'] = df['Candidate'].apply(count_NSUB)
    df['Cand_cconj'] = df['Candidate'].apply(count_CONJ)
    df['Cand_len'] = df['Candidate'].apply(len_sentence)

    return df

# Function to extract data from txt file
def extract_data(f):
    out_dict = {}
    i = 0
    counter = 0
    headers = ['Original', 'Reference', 'Candidate', 'BLEU', 'Label']
    # Read file into a nested dictionary
    with io.open(f, 'r', encoding='utf-8') as file:
        temp_dict = {}
        for line in file:
            if line.strip():
                temp_dict[headers[counter]] = line.strip()
                counter += 1
            else:
                out_dict[i] = temp_dict
                i += 1
                temp_dict = {}
                counter = 0
    df = pd.DataFrame(data=out_dict).T
    return df

# Use jieba to tokenize chinese characters
def tokenize_zh(text):
    words = jieba.lcut(text)
    return words

# Function to vectorize chinese text
def convert_chinese(x, vectorizer, fit=False):
    if(fit):
        matrix = vectorizer.fit_transform(x)
    else:
        matrix = vectorizer.transform(x)
    converted = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names())
    return converted, vectorizer.get_feature_names()

# Function to vectorize english text
def convert_english(x, count_vect, fit=False):
    x = [a.lower() for a in x] 
    x = [re.sub(r'\d+', '', a) for a in x]
    if(fit):
        eng_matrix = count_vect.fit_transform(x)
    else:
        eng_matrix = count_vect.transform(x)
    eng_converted = pd.DataFrame(eng_matrix.toarray(), columns=count_vect.get_feature_names())
    return eng_converted, count_vect.get_feature_names()


if __name__ == "__main__":

    # Extract data from train and text files
    train = extract_data('train.txt')
    train_labels = train.pop('Label').apply(convert_label)

    # Performs the creation of labels based on POS
    train = create_new_labels(train)

    # Declaring the vectorizers
    # We need two because we have to vectorize both the chinese and the english text
    stop_words = ['。', '，']
    chi_vect = CountVectorizer(tokenizer=tokenize_zh, stop_words=stop_words)
    eng_vect = CountVectorizer()

    # Converts words in Chinese and English to one hot vectors in TRAIN
    orig_df, orig_cols = convert_chinese(list(train.pop('Original')), chi_vect, fit=True)
    ref_df, ref_cols = convert_english(list(train.pop('Reference')), eng_vect, fit=True)
    cand_df, cand_cols = convert_english(list(train.pop('Candidate')), eng_vect)

    # Add the new columns back to train 
    train[orig_cols] = orig_df
    train[ref_cols] = ref_df
    train[cand_cols] = cand_df

    # Use the transformer to get TF-IDF
    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(train)
    train_size,num_features = train_tfidf.shape

    # Converts words in Chinese and English to one hot vectors in TEST
    test = extract_data('test.txt')
    test_labels = test.pop('Label').apply(convert_label)
    test = create_new_labels(test)

    # Perform the pre-processing for the test inputs
    test_orig_df, test_orig_cols = test_convert_chinese(list(test.pop('Original')), chi_vect)
    test_ref_df, test_ref_cols = test_convert_english(list(test.pop('Reference')), eng_vect)
    test_cand_df, test_cand_cols = test_convert_english(list(test.pop('Candidate')), eng_vect)
    test[test_orig_cols] = test_orig_df
    test[test_ref_cols] = test_ref_df
    test[test_cand_cols] = test_cand_df

    test_tfidf = tfidf_transformer.transform(test)

    # Finally, fit the SVM using a poly kernel because it has the best F1 score
    SVM = svm.SVC(C=1.0, kernel='poly', gamma=2)
    SVM.fit(train_tfidf,train_labels)

    # Predict on the test labels and calculate the F-1 score
    pred_labels = SVM.predict(test_tfidf)
    print("Average F1-score of the two classes:", sum(sklearn.metrics.f1_score(test_labels, pred_labels, average=None)/2))
