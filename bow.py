from preprocessing import charles_df_filtered, camila_df_filtered, harry_df_filtered, megan_df_filtered, \
    kate_df_filtered, william_df_filtered
import nltk
from sklearn.feature_extraction.text import CountVectorizer

Index = ['user_name', 'user_loc', 'comment_text', 'lower_case', 'lower_no_punct', 'text_wo_stop', 'text_stemmed',
         'text_lemmatized', 'text_tokenized', 'text_pos', 'named_entity']


def top_adjectives_bow(corpus):
    vectorizer = CountVectorizer(stop_words='english', lowercase=True, token_pattern='[a-zA-Z]+', ngram_range=(1, 1),
                                 max_features=500)
    bow_matrix = vectorizer.fit_transform(corpus["comment_text"])
    feature_names = vectorizer.get_feature_names_out()
    word_freq = dict(zip(feature_names, bow_matrix.toarray().sum(axis=0)))
    adjectives = ['NN']
    top_adjectives = [word for (word, freq) in sorted([(word, freq) for word, freq in word_freq.items() if
                                                       nltk.pos_tag([word])[0][1] in adjectives], key=lambda x: x[1],
           reverse=True)[:20]]
    return top_adjectives


bow_charles = top_adjectives_bow(charles_df_filtered)
bow_camila = top_adjectives_bow(camila_df_filtered)
bow_harry = top_adjectives_bow(harry_df_filtered)
bow_megan = top_adjectives_bow(megan_df_filtered)
bow_kate = top_adjectives_bow(kate_df_filtered)
bow_william = top_adjectives_bow(william_df_filtered)

print(bow_charles)
print(bow_camila)
print(bow_harry)
print(bow_megan)
print(bow_kate)
print(bow_william)



