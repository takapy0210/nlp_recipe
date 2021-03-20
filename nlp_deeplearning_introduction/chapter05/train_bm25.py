import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from preprocessing import clean_html, tokenize
from utils import load_dataset, train_and_eval

sys.path.append('../../')
sys.path.append('../../vectorizer/')
from tokenize_ja import tokenizer
from okapi_bm25 import Okapi


def main():
    x, y = load_dataset('data/amazon_reviews_multilingual_JP_v1_00.tsv', n=5000)

    print('Tokenization')
    x = [clean_html(text, strip=True) for text in x]
    x = [' '.join(tokenize(text)) for text in x]
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=42)
    
    print('Okapi')
    vectorizer = Okapi()
    x_train_texts = pd.DataFrame(x_train)[0].apply(lambda x: x.split())
    x_test_texts = pd.DataFrame(x_test)[0].apply(lambda x: x.split())
    train_and_eval(x_train_texts, y_train, x_test_texts, y_test, vectorizer)

    print('TF-IDF')
    vectorizer = TfidfVectorizer()
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)

    print('Bigram')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)


if __name__ == '__main__':
    main()
