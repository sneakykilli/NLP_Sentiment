import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


pd.options.mode.chained_assignment = None

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< START BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# HTML file of DM Comments and Creates a Dataframe.
# Uncomment if you add to the .html file to update.

# html_file = 'dailymail.html'
# with open(html_file) as f:
#     soup = BeautifulSoup(f, 'html.parser')
#
# user_name = soup.find_all(class_='js-usr')
# user_info = soup.find_all(class_='user-info')
# comments = soup.find_all(attrs={"class": ["comment-body comment-text", "reply-body reply-text"]})
#
# user_names = [tag.get_text() for tag in user_name]
# user_loc = [tag.get_text().split(',')[1] for tag in user_info]
# comment_text = [tag.get_text() for tag in comments]
#
# df = pd.DataFrame({'user_name':user_names, 'user_loc':user_loc, 'comment_text':comment_text})
# df.to_csv('dailymail.csv', index=False)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< END BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< START BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Load Dataframe from .csv created above. Add some preprocessing Cols.
df = pd.read_csv('dailymail.csv')
# 1 Add col: LOWER CASE COMMENTS
df['lower_case'] = df['comment_text'].str.lower()
# 2 Add col: LOWER CASE, NO PUNCTUATION
# nltk.download('stopwords')
PUNCT_TO_REMOVE = string.punctuation


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


df['lower_no_punct'] = df['lower_case'].apply(lambda text: remove_punctuation(text))

# 3 Add col: STOP WORDS REMOVED
STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


df["text_wo_stop"] = df["comment_text"].apply(lambda text: remove_stopwords(text))

# 4 Add col: STEMMING
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


df['text_stemmed'] = df["lower_no_punct"].apply(lambda text: stem_words(text))

# 5 Add Col: LEMMATIZED
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


df["text_lemmatized"] = df["text_stemmed"].apply(lambda text: lemmatize_words(text))

# 6 nltk.download('punkt')

df["text_tokenized"] = df["comment_text"].apply(lambda text: nltk.word_tokenize(text))

# 7 Add Col: POS Tag
# nltk.download('averaged_perceptron_tagger')

df["text_pos"] = df["text_tokenized"].apply(lambda text: nltk.pos_tag(text))

# 8 Add Col: Named Entity
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
df["named_entity"] = df["text_pos"].apply(lambda text: nltk.ne_chunk(text))
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< END BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< START BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Isolate comments associated with named individuals
named_entities = df["named_entity"]
people = []
organization = []
gpe = []
for sentence in named_entities:
    for element in sentence:
        try:
            if element.label() == 'PERSON':
                element_leaves = element.leaves()
                person_formatted = tuple([element[0] for element in element_leaves])
                if person_formatted not in people:
                    people.append(person_formatted)
            if element.label() == 'ORGANIZATION':
                element_leaves = element.leaves()
                organization_formatted = tuple([element[0] for element in element_leaves])
                if organization_formatted not in organization:
                    organization.append(organization_formatted)
            if element.label() == 'GPE':
                element_leaves = element.leaves()
                gpe_formatted = tuple([element[0] for element in element_leaves])
                if gpe_formatted not in gpe:
                    gpe.append(element.leaves())
        except AttributeError:
            pass

charles = ("Charles", "Charlie", "Charles III", "Prince Charles", "Klown Charles", "King", "Monarch", "MONARCHY", 'Charle', "Crown", 'The crown')
charles_regex = re.compile(r'\b(?:{}|{})\b'.format('|'.join(charles), '|'.join([re.escape(x) for x in charles])))

camila = ("Parker Bowles", "Camilla", "Queen Camille", "Queen Consort", "Camila", 'Carmilla')
camila_regex = re.compile(r'\b(?:{}|{})\b'.format('|'.join(camila), '|'.join([re.escape(x) for x in camila])))

harry = ("Harry", "Prince Harry", "HARRY", "Harkles", "Sussex", "Harried", "Harrys")
harry_regex = re.compile(r'\b(?:{}|{})\b'.format('|'.join(harry), '|'.join([re.escape(x) for x in harry])))

megan = ("Meghan", "Megan", "Megs", "Mrgan", "Harkles", "Princess Meghan", "American", "Sussex", "Montecito", "Meaghan")
megan_regex = re.compile(r'\b(?:{}|{})\b'.format('|'.join(megan), '|'.join([re.escape(x) for x in megan])))

kate = ("Wales", "Katie", "Catherine", "Waitey", "Saint Catherine", 'Saint Waitie', 'Waitie', 'Queen Kate', 'Katie Show', 'Katherine', "Queen Katherine", "Middleton")
kate_regex = re.compile(r'\b(?:{}|{})\b'.format('|'.join(kate), '|'.join([re.escape(x) for x in kate])))

william = ('Will', 'William', 'Duke of Cambridge', "Wales", 'William IV', 'WIllian V', 'Ill', )
william_regex = re.compile(r'\b(?:{}|{})\b'.format('|'.join(william), '|'.join([re.escape(x) for x in william])))

charles_df_filtered = df[df["comment_text"].str.contains(charles_regex)]
camila_df_filtered = df[df["comment_text"].str.contains(camila_regex)]
harry_df_filtered = df[df["comment_text"].str.contains(harry_regex)]
megan_df_filtered = df[df["comment_text"].str.contains(megan_regex)]
kate_df_filtered = df[df["comment_text"].str.contains(kate_regex)]
william_df_filtered = df[df["comment_text"].str.contains(william_regex)]
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< END BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


