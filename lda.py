
# coding: utf-8

# In[14]:


import pandas as pd
import nltk
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[8]:


df = pd.read_table("./careerData.txt")
df = df [['Title','Description']]
df.columns=['title','desc']


# In[9]:


#display(df)


# In[10]:


#strip any proper names from a text...unfortunately right now this is yanking the first word from a sentence too.
import string
def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


# In[11]:


#strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text
from nltk.tag import pos_tag

def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns


# In[16]:


# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# In[20]:


from gensim import corpora, models, similarities 
import re
from nltk.stem.snowball import SnowballStemmer
# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")


# In[21]:


#remove proper names
preprocess = [strip_proppers(doc) for doc in df['desc']]

#tokenize
tokenized_text = [tokenize_and_stem(text) for text in preprocess]

#remove stop words
texts = [[word for word in text if word not in stopwords] for text in tokenized_text]


# In[22]:


#create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(texts)

#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.8)

#convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(text) for text in texts]


# In[25]:


lda = models.LdaMulticore(corpus, num_topics=100, id2word=dictionary, chunksize=10000, passes=100)



# In[26]:


lda.save("ldaResult")

#import numpy as np

# In[ ]:


#topics_matrix = lda.show_topics(formatted=False, num_words=20)
#topics_matrix = np.array(topics_matrix)
#numpy.savetxt("careerTopics.csv", topics_matrix, delimiter=",")
#topic_words = topics_matrix[:,:,1]
#for i in topic_words:
#    print([str(word) for word in i])
#    print()

