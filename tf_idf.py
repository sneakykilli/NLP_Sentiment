from preprocessing import charles_df_filtered, camila_df_filtered, harry_df_filtered, megan_df_filtered, \
    kate_df_filtered, william_df_filtered
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

list_of_royals = [charles_df_filtered, camila_df_filtered, harry_df_filtered, megan_df_filtered,
                  kate_df_filtered, william_df_filtered]

Index = ['user_name', 'user_loc', 'comment_text', 'lower_case', 'lower_no_punct',
       'text_wo_stop', 'text_stemmed', 'text_lemmatized', 'text_tokenized',
       'text_pos', 'named_entity']


def extract_part_speech(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    adjectives = [word for word, pos in pos_tags if pos.startswith('JJ')]
    return ' '.join(adjectives)

def top_adjectives_tf_idf(corpus):
    corpus['adjectives'] = corpus['text_wo_stop'].apply(extract_part_speech)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus['adjectives'])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    dense_matrix = tfidf_matrix.todense()
    dense_df = pd.DataFrame(dense_matrix, columns=feature_names)
    sums = dense_df.sum(axis=0)
    top_20_adjectives = sums.sort_values(ascending=False)[:10]
    return top_20_adjectives


tf_ifd_charles = top_adjectives_tf_idf(charles_df_filtered)
print(tf_ifd_charles)
tf_ifd_camila = top_adjectives_tf_idf(camila_df_filtered)
tf_ifd_harry = top_adjectives_tf_idf(harry_df_filtered)
tf_ifd_megan = top_adjectives_tf_idf(megan_df_filtered)
tf_ifd_kate = top_adjectives_tf_idf(kate_df_filtered)
tf_ifd_william = top_adjectives_tf_idf(william_df_filtered)






