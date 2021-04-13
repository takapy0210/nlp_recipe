from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocessing import clean_html, tokenize
from utils import load_dataset


def main():
    x, y = load_dataset('data/amazon_reviews_multilingual_JP_v1_00.tsv', n=5000)

    x = [clean_html(text, strip=True) for text in x]
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=42)

    vectorizer = TfidfVectorizer(tokenizer=tokenize)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    clf = LogisticRegression(solver='liblinear')
    clf.fit(x_train_vec, y_train)

    y_pred = clf.predict(x_test_vec)
    score = accuracy_score(y_test, y_pred)
    print('Accuracy(test): {:.4f}'.format(score))


if __name__ == '__main__':
    main()
