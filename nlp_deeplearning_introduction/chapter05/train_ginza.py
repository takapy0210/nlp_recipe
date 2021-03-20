import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from preprocessing import clean_html, tokenize
from utils import load_dataset, train_and_eval

sys.path.append('../../')
from tokenize_ja import tokenizer

def main():
    x, y = load_dataset('data/amazon_reviews_multilingual_JP_v1_00.tsv', n=5000)

    print('Tokenization')
    ginza_tokenizer_C = tokenizer.GinzaTokenizer(mode="C")
    x = [' '.join(ginza_tokenizer_C.tokenize(text)) for text in x]
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=42)

    print('TF-IDF')
    vectorizer = TfidfVectorizer()
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)

    print('Bigram')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    train_and_eval(x_train, y_train, x_test, y_test, vectorizer)


if __name__ == '__main__':
    main()
