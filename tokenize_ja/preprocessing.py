"""文字列の前処理・正規化

全ての前処理を行う場合はclean_text()関数を呼ぶ

"""

import re
import unicodedata
import neologdn


def clean_specific_character(text: str) -> str:
    """形態素解析でエラーとなる特殊な文字を除外する

    Args:
        text (str): 処理前のテキスト

    Returns:
        str: 処理後のテキスト
    """
    text = text.replace(' ', '')
    text = text.replace('^ ^', '')  # jumanppのエラー回避のため
    return text


def clean_html(text: str) -> str:
    """urlを除外する

    Args:
        text (str): 処理前のテキスト

    Returns:
        str: 処理後のテキスト
    """
    url_string = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
    return url_string.sub(' ', text)


def clean_kaomoji(text: str) -> str:
    """顔文字を除外する

    Args:
        text (str): 処理前のテキスト

    Returns:
        str: 処理後のテキスト
    """
    kaomoji_pattern = re.compile(r'\([^あ-ん\u30A1-\u30F4\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+?\)')
    return kaomoji_pattern.sub(' ', text)


def clean_emoji(text: str) -> str:
    """絵文字を除外する

    Args:
        text (str): 処理前のテキスト

    Returns:
        str: 処理後のテキスト
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)


def clean_punctuation(text: str) -> str:
    """【】,（）,［］,@,＠,全角空白を、半角スペースに置換

    Args:
        text (str): [description]

    Returns:
        str: [description]
    """
    replaced_string = re.compile(r'[【】]|[,.。、]|[（）()]|[［］\[\]]|[@＠]|　')
    return replaced_string.sub(' ', text)


def clean_hashtag(text: str) -> str:
    """ハッシュタグを除外する

    Args:
        text (str): [description]

    Returns:
        str: [description]
    """
    text = re.sub(r'( #[a-zA-Z]+)+$', '', text)
    text = re.sub(r' #([a-zA-Z]+) ', r'\1', text)
    return text


def normalize_lower(text: str) -> str:
    """小文字に変換する

    Args:
        text (str): 処理前のテキスト

    Returns:
        str: 処理後のテキスト
    """
    return text.lower()


def normalize_unicodedata(text: str) -> str:
    """半角カタカナ、全角英数、ローマ数字・丸数字、異体字などなどの正規化
    例：㌔→キロ、①→1、ｷﾀｰ→キター、など

    Args:
        text (str): 処理前のテキスト

    Returns:
        str: [description]
    """
    return unicodedata.normalize('NFKC', text)


def normalize_neologdn(text: str) -> str:
    """neologdnの正規化を行う
    日本語テキストに対してneologd辞書を用いる前に推奨される正規化（表記ゆれの是正）

    Args:
        text (str): 処理前のテキスト

    Returns:
        str: 処理後のテキスト
    """
    return neologdn.normalize(text)


def normalize_number(text: str, reduce=False) -> str:
    """連続した数字を0で置換

    Args:
        text (str): 正規化する文字列
        reduce (book): 数字の文字数を変化させるか否か. Defaults to False.
            例:
                Trueの場合「2万1870ドル」→「0万0ドル」
                Falseの場合「2万1870ドル」→「0万0000ドル」

    Returns:
        str: 正規化後の文字列
    """
    if reduce:
        return re.sub(r'\d+', '0', text)
    else:
        return re.sub(r'\d', '0', text)


def clean_text(text: str) -> str:
    """全ての前処理を実行するサンプル

    Args:
        text (str): 前処理前のテキスト

    Returns:
        str: 前処理後のテキスト
    """
    text = clean_specific_character(text)
    text = clean_html(text)
    text = clean_kaomoji(text)
    text = clean_emoji(text)
    text = clean_punctuation(text)
    text = clean_hashtag(text)
    text = normalize_lower(text)
    text = normalize_unicodedata(text)
    text = normalize_neologdn(text)
    text = normalize_number(text)
    return text
