import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from preprocessing import charles_df_filtered, camila_df_filtered, harry_df_filtered, megan_df_filtered, \
    kate_df_filtered, william_df_filtered
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

Index = ['user_name', 'user_loc', 'comment_text', 'lower_case', 'lower_no_punct',
       'text_wo_stop', 'text_stemmed', 'text_lemmatized', 'text_tokenized',
       'text_pos', 'named_entity']

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< START BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


def get_sentiment_nltk(corpus):
    series_to_string = ''.join(corpus["comment_text"].tolist())
    sentiment_score = sia.polarity_scores(series_to_string)
    return sentiment_score


charles_sentiment_nltk = get_sentiment_nltk(charles_df_filtered)
camila_sentiment_nltk = get_sentiment_nltk(camila_df_filtered)
harry_sentiment_nltk = get_sentiment_nltk(harry_df_filtered)
megan_sentiment_nltk = get_sentiment_nltk(megan_df_filtered)
kate_sentiment_nltk = get_sentiment_nltk(kate_df_filtered)
william_sentiment_nltk = get_sentiment_nltk(william_df_filtered)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< END BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< START BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

windsoer = ['Charles', 'Camila', 'Harry', 'Megan', 'Kate', 'William']
dicts = [charles_sentiment_nltk, camila_sentiment_nltk, harry_sentiment_nltk, megan_sentiment_nltk, kate_sentiment_nltk, william_sentiment_nltk]
neg = []
neu = []
pos = []
compound = []
for element in dicts:
    neg.append(element['neg'])
    neu.append(element['neu'])
    pos.append(element['pos'])
    compound.append(element['compound'])

sentiment = pd.DataFrame({'Windsor':windsoer, 'Negative':neg, 'Positive':pos, 'Neutral':neu, 'Compound':compound})
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< END BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< START BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
df = pd.DataFrame(sentiment.iloc[:,:3])
df.set_index('Windsor', inplace=True)
ax = df.plot(kind='bar', figsize=(10, 6))

def y_fmt(y, pos):
    return '{:.0%}'.format(y)

ax.set_xlabel('Windsors')
ax.set_ylabel('Sentiment Scores')
ax.set_title('Sentiment Analysis of Windsors')
ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
plt.show()
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< END BLOCK  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
