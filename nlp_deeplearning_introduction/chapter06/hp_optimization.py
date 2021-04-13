from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

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

    parameters = {'penalty': ['l1', 'l2'],
                  'C': [0.01, 0.03, 0.1, 0.3, 0.7, 1, 1.01, 1.03, 1.07, 1.1, 1.3, 1.7, 3]}
    lr = LogisticRegression(solver='liblinear')
    clf = GridSearchCV(lr, parameters, cv=5, n_jobs=-1)
    clf.fit(x_train_vec, y_train)

    best_clf = clf.best_estimator_
    print(clf.best_params_)
    print('Accuracy(best): {:.4f}'.format(clf.best_score_))
    y_pred = best_clf.predict(x_test_vec)
    score = accuracy_score(y_test, y_pred)
    print('Accuracy(test): {:.4f}'.format(score))


if __name__ == '__main__':
    main()
