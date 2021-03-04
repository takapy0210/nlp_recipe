import sys
import os
import string
import time
import math
import psutil
import warnings
from contextlib import contextmanager

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.append('../../')
from tokenize_ja import tokenizer

warnings.filterwarnings("ignore")


@contextmanager
def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    yield
    m1 = p.memory_info()[0] / 2. ** 30
    delta = m1 - m0
    sign = '+' if delta >= 0 else '-'
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)


def load_dataset(filename, n=5000, state=6):

    def filter_by_ascii_rate(text, threshold=0.9):
        ascii_letters = set(string.printable)
        rate = sum(c in ascii_letters for c in text) / len(text)
        return rate <= threshold

    df = pd.read_csv(filename, sep='\t')

    # extracts Japanese texts.
    is_jp = df.review_body.apply(filter_by_ascii_rate)
    df = df[is_jp]

    # sampling.
    df = df.sample(frac=1, random_state=state)  # shuffle
    grouped = df.groupby('star_rating')
    df = grouped.head(n=n)
    return pd.DataFrame(df).reset_index(drop='True')


def train_and_eval(x_train, y_train, x_test, y_test):
    vectorizer = CountVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    clf = LogisticRegression(solver='liblinear')
    clf.fit(x_train_vec, y_train)
    y_pred = clf.predict(x_test_vec)
    score = accuracy_score(y_test, y_pred)
    print('{:.4f}'.format(score))


def main():
    with trace('load data'):
        df = load_dataset('data/amazon_reviews_multilingual_JP_v1_00.tsv', n=1000)
        # 特殊な文字列が入っているレビューを除外
        df = df[~df['review_body'].str.contains('ã')].reset_index(drop=True)

    X = df[['review_body']]
    y = df[['star_rating']]

    # tokenize
    mecab_tokenizer = tokenizer.MeCabTokenizer(sys_dic_path='/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
    with trace("mecab tokenize"):
        X.loc[:, 'mecab_token'] = X['review_body'].apply(mecab_tokenizer.tokenize)
        X.loc[:, 'mecab_token'] = X['mecab_token'].apply(lambda x: ' '.join(x))

    # train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with trace("train"):
        train_and_eval(X_train['mecab_token'], y_train['star_rating'].values,
                       X_test['mecab_token'], y_test['star_rating'].values)


if __name__ == '__main__':
    main()
