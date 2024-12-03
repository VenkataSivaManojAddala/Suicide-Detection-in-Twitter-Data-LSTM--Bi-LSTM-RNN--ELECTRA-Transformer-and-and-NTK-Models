
# Importing Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import spacy
import unidecode
import contractions as contract
import re
import wordninja
import collections
import pkg_resources
from spellchecker import SpellChecker 
from symspellpy import SymSpell, Verbosity

# --------------------------------------------------------------------------------

# Load dataset
df = pd.read_csv('Data/twitter_data.csv', index_col=0)
df.reset_index(drop=True, inplace=True)
df.head()

# --------------------------------------------------------------------------------

# Defining methods

nlp = spacy.load("en_core_web_sm") 
vocab = collections.Counter()
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename(
"symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

# Spell Check using Symspell
def fix_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    correctedtext = suggestions[0].term # get the first suggestion, otherwise returns original text if nothing is corrected 
    return correctedtext 

# Remove some important words from stopwords list 
deselect_stop_words = ['no', 'not']
    
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False

# Remove extra whitespaces from text
def remove_whitespace(text):
    text = text.strip()
    return " ".join(text.split())

# Remove accented characters from text, e.g. café
def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text

# Remove URL 
def remove_url(text):
    return re.sub(r'http\S+', '', text)

# Removing symbols and digits
def remove_symbols_digits(text):
    return re.sub('[^a-zA-Z\s]', ' ', text)

# Removing special characters
def remove_special(text):
    return text.replace("\r", " ").replace("\n", " ").replace("    ", " ").replace('"', '')

# Fix word lengthening (characters are wrongly repeated)
def fix_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def text_preprocessing(text, accented_chars=True, contractions=True, convert_num=True, 
                       extra_whitespace=True, lemmatization=True, lowercase=True, 
                       url=True, symbols_digits=True, special_chars=True, 
                       stop_words=True, lengthening=True, spelling=True):
    """preprocess text with default option set to true for all steps"""
    if accented_chars == True: # remove accented characters
        text = remove_accented_chars(text)
    if contractions == True: # expand contractions
        text = contract.fix(text)
    if lowercase == True: # convert all characters to lowercase
        text = text.lower()
    if url == True: # remove URLs before removing symbols 
        text = remove_url(text)
    if symbols_digits == True: # remove symbols and digits
        text = remove_symbols_digits(text)
    if special_chars == True: # remove special characters
        text = remove_special(text)
    if extra_whitespace == True: # remove extra whitespaces
        text = remove_whitespace(text)
    if lengthening == True: # fix word lengthening
        text = fix_lengthening(text)
    if spelling == True: # fix spelling
        text = fix_spelling(text)

    doc = nlp(text) # tokenise text

    clean_text = []

    # return text
    
    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
        # exclude number words
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            flag = False
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list 
        if edit != "" and flag == True:
            clean_text.append(edit)        
    return " ".join(clean_text)

# --------------------------------------------------------------------------------

# Test functions on a subset of 20 rows
df['cleaned_text'] = df['text'][:20].apply(lambda row: text_preprocessing(row))
df[:20]

# --------------------------------------------------------------------------------

# Apply preprocessing to all data
df['cleaned_text'] = df['text'].apply(lambda row: text_preprocessing(row))

# --------------------------------------------------------------------------------

# Export cleaned dataset
df.to_csv('Data/twitter_data_full_cleaned.csv', index=False)

# --------------------------------------------------------------------------------

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# --------------------------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt 
from keras.preprocessing.text import Tokenizer
import seaborn as sns

# --------------------------------------------------------------------------------

# Load dataset
cleaned_df = pd.read_csv('Data/twitter_data_full_cleaned.csv') 
cleaned_df.head()

# --------------------------------------------------------------------------------

# Obtain word frequency 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(cleaned_df['cleaned_text'])
word_freq = pd.DataFrame(tokenizer.word_counts.items(), columns=['word','count']).sort_values(by='count', ascending=False)

# --------------------------------------------------------------------------------

# Plot bar graph for word frequency 
plt.figure(figsize=(16, 8))
sns.barplot(x='count',y='word',data=word_freq.iloc[:30])
plt.title('Most Frequent Words')
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()

# --------------------------------------------------------------------------------

# Removed anomalous "filler" word 
cleaned_df['cleaned_text'] = cleaned_df['cleaned_text'].str.replace('filler', '')

# --------------------------------------------------------------------------------

# Remove rows with text length 0
cleaned_df = cleaned_df[cleaned_df['cleaned_text'].apply(lambda x: len(x.split())!=0)]
cleaned_df.reset_index(drop=True, inplace=True)
cleaned_df.head()

# --------------------------------------------------------------------------------

# Get word count of posts 
posts_len = [len(x.split()) for x in cleaned_df['cleaned_text']]
pd.Series(posts_len).hist(bins=60)
plt.show()
print(pd.Series(posts_len).describe())

# --------------------------------------------------------------------------------

# Subset dataset to obtain rows with less than or equal to 62 words
cleaned_df = cleaned_df[cleaned_df['cleaned_text'].apply(lambda x: len(x.split())<=62)]
cleaned_df.reset_index(drop=True, inplace=True)

# --------------------------------------------------------------------------------

# Check dataset 
cleaned_df.head()

# --------------------------------------------------------------------------------

# Export cleaned dataset 
cleaned_df.to_csv('/Data/twitter_data_final_cleaned.csv', index=False)

# --------------------------------------------------------------------------------

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# --------------------------------------------------------------------------------

!pip install -qqq emoji fasttext

# --------------------------------------------------------------------------------

# Import packages
import os
import pandas as pd
import numpy as np
import itertools
import collections
import networkx as nx
import six
import matplotlib.pyplot as plt
import seaborn as sns
import emoji as em
import fasttext

from keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split

import nltk
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk.corpus import stopwords
nltk.download('punkt')

# --------------------------------------------------------------------------------

# Set constants
SEED = 4222

# --------------------------------------------------------------------------------

# Load dataset
df = pd.read_csv('Data/twitter_data.csv', index_col=0)
df.reset_index(drop=True, inplace=True)
df.head()

# --------------------------------------------------------------------------------

# Check for null values
df.isnull().sum()

# --------------------------------------------------------------------------------

# Check class distribution
print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True))

sns.countplot(x=df['class'])
plt.title('Original Dataset Class Distribution')
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# --------------------------------------------------------------------------------

# Check lowercase
lowercase = df['text'].str.islower()
print(lowercase.value_counts())

# df['text'][lowercase == True]

# --------------------------------------------------------------------------------

# Check extra whitespace
extra_whitespace = df['text'].str.match('\s\s+')
print(extra_whitespace.value_counts())

# df['text'][extra_whitespace == True]

# --------------------------------------------------------------------------------

# Check URL
url = df['text'].str.contains("http")
print(url.value_counts())

# df['text'][url == True]

# --------------------------------------------------------------------------------

# Check mentions
mention = df['text'].str.match('@(\w+)')
print(mention.value_counts())

# df['text'][mention == True]

# --------------------------------------------------------------------------------

# Check hashtags
hashtag = df['text'].str.match('#(\w+)')
print(hashtag.value_counts())

# df['text'][hashtag == True]

# --------------------------------------------------------------------------------

# Check subreddit tag
subreddit = df['text'].str.match('r/(\w+)')
print(subreddit.value_counts())

# df['text'][subreddit == True]

# --------------------------------------------------------------------------------

# Check users tag
users = df['text'].str.match('u/(\w+)')
print(users.value_counts())

# df['text'][users == True]

# --------------------------------------------------------------------------------

# Check special characters
special_characters = df['text'].str.match('[^0-9a-zA-Z]+')
print(special_characters.value_counts())

# df['text'][special_characters == True]

# --------------------------------------------------------------------------------

! pip install emoji
import emoji

# Check emojis
def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return True
    return False

emoji = df['text'].apply(text_has_emoji)
print(emoji.value_counts())

# df['text'][emoji == True]

# --------------------------------------------------------------------------------

def count_emojis(text):
    return len([c for c in text if c in em.UNICODE_EMOJI['en'].keys()])

emoji = df['text'].apply(count_emojis)
print(emoji.value_counts())

# df['text'][emoji > 0]

# --------------------------------------------------------------------------------

# Check language
!wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

PRETRAINED_FASTTEXT_PATH = '/tmp/lid.176.bin'
model = fasttext.load_model(PRETRAINED_FASTTEXT_PATH)

def check_language(text):
  sentences = text.split("\n")
  predictions = model.predict(sentences)
  language = predictions[0][0][0].split("__label__")[1]
  confidence = predictions[1][0][0]

  return language

language = df['text'].apply(check_language)
print(language.value_counts())

# df['text'][language != 'en']

# --------------------------------------------------------------------------------

# Load dataset
clean_df = pd.read_csv('Data/twitter_data_final_cleaned.csv', header=0)
clean_df

# --------------------------------------------------------------------------------

# Check for null values
clean_df.isnull().sum()

# --------------------------------------------------------------------------------

# Split dataset into train and test sets 
train_data, test_data = train_test_split(clean_df,
                                         test_size=0.2, 
                                         random_state=SEED,
                                         stratify=clean_df['class'])

# --------------------------------------------------------------------------------

# Split train set into classes
train_data_suicidal = train_data[train_data['class'] == "suicide"]
train_data_nonsuicidal = train_data[train_data['class'] == "non-suicide"]

# --------------------------------------------------------------------------------

# Check class distribution
print(clean_df['class'].value_counts())
print(clean_df['class'].value_counts(normalize=True))

sns.countplot(x=clean_df['class'])
plt.title('Cleaned Dataset Class Distribution')
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# --------------------------------------------------------------------------------

# Check class distribution
print(train_data['class'].value_counts())
print(train_data['class'].value_counts(normalize=True))

sns.countplot(x=train_data['class'])
plt.title('Cleaned Train Class Distribution')
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

# --------------------------------------------------------------------------------

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data_suicidal['cleaned_text'])

# --------------------------------------------------------------------------------

tokenizer.word_counts.items()

# --------------------------------------------------------------------------------

word_freq_suicidal = pd.DataFrame(tokenizer.word_counts.items(), columns=['word','count']).sort_values(by='count', ascending=False)

# --------------------------------------------------------------------------------

# Word Frequency Bar Graph
plt.figure(figsize=(16, 8))
sns.barplot(x='count',y='word',data=word_freq_suicidal.iloc[:20])
plt.title('Most Frequent Words - Suicidal Text')
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()

# --------------------------------------------------------------------------------

# Wordcloud 
feature_names=word_freq_suicidal['word'].values
wc=WordCloud(max_words=100, background_color="white", width=2000, height=1000)
wc.generate(' '.join(word for word in feature_names))
plt.figure(figsize=(20,15))
plt.axis('off')
plt.imshow(wc)
plt.show()

# --------------------------------------------------------------------------------

# Get average text length
train_data_suicidal['cleaned_text'] = train_data_suicidal['cleaned_text'].astype('str')
train_data_suicidal['length'] = train_data_suicidal['cleaned_text'].apply(lambda x: len(x.split()))

# --------------------------------------------------------------------------------

# Plot average text length
ax = train_data_suicidal['length'].plot(kind='hist',title='Distribution of Text Length - Suicidal Text', figsize=(8,6))
ax.set_xlabel("Number of Words")
ax.set_ylabel("Count")
plt.show()

# --------------------------------------------------------------------------------

# Polarity score distribution
def get_polarity(text):
  return TextBlob(text).sentiment.polarity
train_data_suicidal['Polarity'] = train_data_suicidal['cleaned_text'].apply(get_polarity)

# --------------------------------------------------------------------------------

# Plot polarity score graph
ax = train_data_suicidal['Polarity'].plot(kind='hist', title='Polarity Score - Suicidal Text', figsize=(8,6))
ax.set_xlabel("Polarity Score")
ax.set_ylabel("Count")
plt.show()

# --------------------------------------------------------------------------------

!pip install nltk==3.6.2
#import nltk
#import sklearn

print('The nltk version is {}.'.format(nltk.__version__))

# --------------------------------------------------------------------------------

#Remove more stop words and do bigram
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')
train_data_suicidal['without_stopwords'] = train_data_suicidal['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

from nltk import bigrams
sentences = [text.split() for text in train_data_suicidal['without_stopwords']]

# Create list of lists containing bigrams in tweets
terms_bigram = [list(bigrams(text)) for text in sentences]

# Flatten list of bigrams in clean tweets
bigrams = list(itertools.chain(*terms_bigram))

# Create counter of words in clean bigrams
bigram_counts = collections.Counter(bigrams)

#Create a table of the top 20 most paired words
bigram_df = pd.DataFrame(bigram_counts.most_common(20),
                             columns=['Bigram', 'Count'])

bigram_df

# --------------------------------------------------------------------------------

#To make the table nicer
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

render_mpl_table(bigram_df, header_columns=0, col_width=3)

# --------------------------------------------------------------------------------

tokenizer_nonsuicidal = Tokenizer()
tokenizer_nonsuicidal.fit_on_texts(train_data_nonsuicidal["text"])

# --------------------------------------------------------------------------------

tokenizer_nonsuicidal.word_counts.items()

# --------------------------------------------------------------------------------

word_freq_nonsuicidal = pd.DataFrame(tokenizer_nonsuicidal.word_counts.items(), columns=['word','count']).sort_values(by='count', ascending=False)
word_freq_nonsuicidal.head()

# --------------------------------------------------------------------------------

# Word Frequency Bar Graph
plt.figure(figsize=(16, 8))
sns.barplot(x='count', y='word', data=word_freq_nonsuicidal.iloc[:20])
plt.title('Most Frequent Words - Non-suicidal Text')
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()

# --------------------------------------------------------------------------------

# Wordcloud  
feature_names_nonsuicidal = word_freq_nonsuicidal['word'].values
wc = WordCloud(max_words=100, background_color="white", width=2000, height=1000)
wc.generate(' '.join(word for word in feature_names_nonsuicidal))
plt.figure(figsize=(20,15))
plt.axis('off')
plt.imshow(wc)
plt.show()

# --------------------------------------------------------------------------------

# Get average text length
train_data_nonsuicidal['cleaned_text'] = train_data_nonsuicidal['cleaned_text'].astype('str')
train_data_nonsuicidal['length'] = train_data_nonsuicidal['cleaned_text'].apply(lambda x: len(x.split()))

# --------------------------------------------------------------------------------

# Plot distribution of text length
ax = train_data_nonsuicidal['length'].plot(kind='hist',title='Distribution of Text Length - Non-suicidal Text', color='orange', figsize=(8,6))
ax.set_xlabel("Number of Words")
ax.set_ylabel("Count")
plt.show()

# --------------------------------------------------------------------------------

# Polarity score distribution
train_data_nonsuicidal['cleaned_text'] = train_data_nonsuicidal['cleaned_text'].astype('str')
train_data_nonsuicidal['Polarity'] = train_data_nonsuicidal['cleaned_text'].apply(get_polarity)

# --------------------------------------------------------------------------------

# Plot polarity score graph
ax = train_data_nonsuicidal['Polarity'].plot(kind='hist', title='Polarity Score - Non-suicidal Text', color="orange", figsize=(8,6))
ax.set_xlabel("Polarity Score")
ax.set_ylabel("Count")
plt.show()

# --------------------------------------------------------------------------------

#Remove more stop words and do bigram
stop_words = stopwords.words('english')
train_data_nonsuicidal['without_stopwords'] = train_data_nonsuicidal['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

from nltk import bigrams
sentences = [text.split() for text in train_data_nonsuicidal['without_stopwords']]

# Create list of lists containing bigrams in tweets
terms_bigram = [list(bigrams(text)) for text in sentences]

# Flatten list of bigrams in clean tweets
bigrams = list(itertools.chain(*terms_bigram))

# Create counter of words in clean bigrams
bigram_counts = collections.Counter(bigrams)

#Create a table of the top 20 most paired words
bigram_df = pd.DataFrame(bigram_counts.most_common(20),
                             columns=['Bigram', 'Count'])

render_mpl_table(bigram_df, header_columns=0, col_width=3)


# Additional Processing Section

# Handling Missing Values
def handle_missing_values(df):
    df.fillna(df.mean(), inplace=True)
    return df

# Encoding Categorical Features
from sklearn.preprocessing import LabelEncoder
def encode_categorical_features(df):
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df

# Feature Scaling
from sklearn.preprocessing import StandardScaler
def scale_features(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

# Example Usage
if __name__ == "__main__":
    # Assuming 'data' is a preloaded DataFrame
    data = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    data = handle_missing_values(data)
    data = encode_categorical_features(data)
    data = scale_features(data)
    print("Processed Data:")
    print(data.head())
