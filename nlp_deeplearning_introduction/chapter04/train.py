from sklearn.model_selection import train_test_split

from preprocessing import clean_html, normalize_number, tokenize, tokenize_base_form
from utils import load_dataset, train_and_eval


def main():
    x, y = load_dataset('data/amazon_reviews_multilingual_JP_v1_00.tsv', n=1000)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.2,
                                                        random_state=42)

    print('Tokenization only.')
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize)

    print('Clean html.')
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize, preprocessor=clean_html)

    print('Normalize number.')
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize, preprocessor=normalize_number)

    print('Base form.')
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize_base_form)

    print('Lower text.')
    train_and_eval(x_train, y_train, x_test, y_test, tokenize=tokenize, lowercase=True)


if __name__ == '__main__':
    main()
