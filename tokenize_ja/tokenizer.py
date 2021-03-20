"""日本語のトークナイズ処理

"""


import MeCab
import spacy
import ginza
from pyknp import Juman
from sudachipy import tokenizer
from sudachipy import dictionary
from janome.tokenizer import Tokenizer
import pandas as pd
from tqdm import tqdm as tqdm

import preprocessing

PROPER_NOUN = '固有名詞'


def preprocess(text: str) -> str:
    """テキストの前処理を行う

    Args:
        text (str): 前処理後のテキスト

    Returns:
        str: 前処理後のテキスト
    """
    # text = preprocessing.clean_specific_character(text)
    text = preprocessing.clean_html(text)
    text = preprocessing.clean_kaomoji(text)
    text = preprocessing.clean_emoji(text)
    text = preprocessing.clean_punctuation(text)
    text = preprocessing.clean_hashtag(text)
    text = preprocessing.normalize_lower(text)
    text = preprocessing.normalize_unicodedata(text)
    text = preprocessing.normalize_neologdn(text)
    return text


class MeCabTokenizer(object):
    """mecabで形態素解析を行う
    表層形  品詞  品詞細分類1  品詞細分類2  品詞細分類3  活用形  活用型  原形  読み  発音を返却する

    sample code:
        option = '-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'
        mecab = MeCab.Tagger(option)
        parser = mecab.parse
        chunks = parser(text.rstrip()).splitlines()[:-1]
        for chunk in chunks:
            chunk.split('\t')[1].split(',')[0]
            surface, feature = chunk.split('\t')
            feature = feature.split(',')
            if len(feature) <= 7:  # 読みがない場合は追加
                feature.append('*')
            print(f'表層形: {surface}\t| 読み: {feature[7]}\t| 原型: {feature[6]}\t| 品詞: {feature[0]}\
            \t| 品詞細分類1: {feature[1]}\t| 品詞細分類2: {feature[2]}\t| 品詞細分類3: {feature[3]}\
            \t| 活用形: {feature[4]}\t| 活用型: {feature[5]}')

    ref:
        http://www.ic.daito.ac.jp/~mizutani/mining/morphology.html
    """

    def __init__(self, sys_dic_path: str = '', user_dic_path: str = '', stopwords=None, include_pos=None):

        option = ''
        if sys_dic_path:
            option += ' -d {0}'.format(sys_dic_path)
        if user_dic_path:
            option += ' -u {0}'.format(user_dic_path)
        mecab = MeCab.Tagger(option)
        self.parser = mecab.parse

        if stopwords is None:
            self.stopwords = []
        else:
            self.stopwords = stopwords
        if include_pos is None:
            self.include_pos = ["名詞", "動詞", "形容詞"]
        else:
            self.include_pos = include_pos

    def tokenize(self, text: str, pos: bool = False) -> list:
        """指定した品詞でトークナイズを行う

        Args:
            text (str): トークナイズ前のテキスト
            pos (bool, optional): 品詞を合わせて返却するか否か. Defaults to False.

        Returns:
            list: トークナイズを行った単語リスト
        """
        # 前処理
        text = preprocess(text)

        chunks = self.parser(text.rstrip()).splitlines()[:-1]
        res = []
        for chunk in chunks:
            chunk.split('\t')[1].split(',')[0]
            surface, feature = chunk.split('\t')
            feature = feature.split(',')
            p = feature[0]
            base = feature[6]
            if base == '*':
                base = surface  # 原型が存在しないものは、元の単語を返却する
            if p in self.include_pos and base not in self.stopwords:
                if pos:
                    res.append((base, p))
                else:
                    res.append(base)

        return res

    def tokenize_proper_noun(self, text):
        """固有名詞のみでトークナイズを行う

        Args:
            text (str): トークナイズ前のテキスト

        Returns:
            list: トークナイズを行った単語リスト
        """
        # 前処理
        text = preprocess(text)

        chunks = self.parser(text.rstrip()).splitlines()[:-1]
        res = []
        for chunk in chunks:
            chunk.split('\t')[1].split(',')[0]
            surface, feature = chunk.split('\t')
            feature = feature.split(',')
            base = feature[6]
            if base == '*':
                base = surface  # 原型が存在しないものは、元の単語を返却する
            pos2 = feature[1]  # 品詞細分類
            if pos2 in PROPER_NOUN and base not in self.stopwords:
                res.append(base)

        return res


class GinzaTokenizer():
    """Ginzaで形態素解析を行う

    sample code:
        ginza = spacy.load("ja_ginza")
        doc = ginza(text)
        for sent in doc.sents:  # 文に分割
            for token in sent:  # トークナイズ
                print(f'表層形: {token.text}\t| 読み: \t| 原型: {token.lemma_}\t| 品詞: {token.pos_}\
                \t| 品詞詳細: {token.tag_}\t| トークン番号: {str(token.i)}')

    ref:
        https://megagonlabs.github.io/ginza/
    """

    def __init__(self, mode="C", stopwords=None, include_pos=None):

        if mode not in ["A", "B", "C"]:
            raise Exception("invalid mode. 'A' ,'B' or 'C'")

        self.ginza = spacy.load("ja_ginza")
        ginza.set_split_mode(self.ginza, mode)

        if stopwords is None:
            self.stopwords = []
        else:
            self.stopwords = stopwords
        if include_pos is None:
            self.include_pos = ["名詞", "動詞", "形容詞"]
        else:
            self.include_pos = include_pos

        self.dep_tag_dict = {
            'acl': '名詞の節修飾語',
            'advcl': '副詞節修飾語',
            'advmod': '副詞修飾語',
            'amod': '形容詞修飾語',
            'appos': '同格',
            'aux': '助動詞',
            'case': '格表示',
            'cc': '等位接続詞',
            'ccomp': '補文',
            'clf': '類別詞',
            'compound': '複合名詞',
            'conj': '結合詞',
            'cop': '連結詞',
            'csubj': '主部',
            'dep': '不明な依存関係',
            'det': '限定詞',
            'discourse': '談話要素',
            'dislocated': '転置',
            'expl': '嘘辞',
            'fixed': '固定複数単語表現',
            'flat': '同格複数単語表現',
            'goeswith': '1単語分割表現',
            'iobj': '関節目的語',
            'list': 'リスト表現',
            'mark': '接続詞',
            'nmod': '名詞修飾語',
            'nsubj': '主語名詞',
            'nummod': '数詞修飾語',
            'obj': '目的語',
            'obl': '斜格名詞',
            'orphan': '独立関係',
            'parataxis': '並列',
            'punct': '句読点',
            'reparandum': '単語として認識されない表現',
            'ROOT': '文の根',
            'vocative': '発声関係',
            'xcomp': '補体',
        }

    def tokenize(self, text, pos=False):
        """指定した品詞でトークナイズを行う

        Args:
            text (str): トークナイズ前のテキスト
            pos (bool, optional): 品詞を合わせて返却するか否か. Defaults to False.

        Returns:
            list: トークナイズを行った単語リスト
        """
        # 前処理
        # text = preprocess(text)

        # 形態素解析
        doc = self.ginza(text)
        res = []
        for sent in doc.sents:
            for token in sent:
                p = token.tag_.split("-")[0]  # 品詞
                base = token.lemma_  # 原型
                if p in self.include_pos and base not in self.stopwords:
                    if pos:
                        res.append((base, p))
                    else:
                        res.append(base)

        return res

    def tokenize_proper_noun(self, text):
        """固有名詞のみでトークナイズを行う

        Args:
            text (str): トークナイズ前のテキスト

        Returns:
            list: トークナイズを行った単語リスト
        """
        # 前処理
        text = preprocess(text)

        # 形態素解析
        doc = self.ginza(text)
        res = []
        for sent in doc.sents:
            for token in sent:
                base = token.lemma_  # 原型
                group_pos = token.tag_.split("-")
                if len(group_pos) >= 2:  # 品詞詳細が存在する単語に絞る
                    pos2 = group_pos[1]  # 品詞詳細を取得
                    if pos2 in PROPER_NOUN and base not in self.stopwords:
                        res.append(base)

        return res

    def get_morphological_df(self, input_df, id_col, text_col, filter_dep_tag=None) -> pd.DataFrame:
        """形態素解析DFを生成する

        Args:
            input_df (pd.DataFrame): 形態素解析DFを生成する元のDF
            id_col (str): 文章IDカラム
            text_col (str): テキストデータが保存されているカラム
            filter_dep_tag (list of str): 対象とする係り受けタグ: https://qiita.com/kei_0324/items/400f639b2f185b39a0cf

        Returns:
            pd.DataFrame: DataFrame
                [id, sentence_id, word_id, serface, lemma, pos, posd, entity_type, dep_tag, dep_tag_jp, dep_word_id, dep_lemma, dep_relation]
        """

        res = []
        for i in tqdm(range(input_df.shape[0])):
            _id = input_df.iloc[i][id_col]
            _text = input_df.iloc[i][text_col]
            # 前処理
            text = preprocess(_text)

            # 形態素解析
            doc = self.ginza(text)
            for sent_id, sent in enumerate(doc.sents):
                for token in sent:
                    p = token.tag_.split("-")[0]  # 品詞
                    base = token.lemma_  # 原型
                    if p in self.include_pos and base not in self.stopwords:
                        info_dict = {}
                        info_dict["id"] = _id
                        info_dict["sentence_id"] = sent_id
                        info_dict["word_id"] = token.i  # トークン番号（複数文がある場合でも0に戻らず連番になる）
                        info_dict["serface"] = token.orth_  # オリジナルテキスト
                        info_dict["lemma"] = token.lemma_  # 基本形（名寄せ後）
                        info_dict["pos"] = token.tag_.split("-")[0]  # 品詞
                        info_dict["posd"] = token.tag_.split("-")[1] if len(token.tag_.split("-")) > 1 else ''  # 品詞詳細
                        info_dict['entity_type'] = token.ent_type_  # 固有表現タグ
                        dep_tag = token.dep_  # 係り受けの関係性
                        info_dict["dep_tag"] = dep_tag
                        info_dict["dep_tag_jp"] = self.dep_tag_dict.get(dep_tag, '不明なタグ')
                        info_dict["dep_word_id"] = token.head.i  # 係り受けの相手トークン番号
                        info_dict["dep_lemma"] = token.head.lemma_  # 係り受けの相手のテキスト
                        info_dict["dep_relation"] = token.lemma_ + ' -> ' + token.head.lemma_
                        if (filter_dep_tag is not None) and (dep_tag not in filter_dep_tag):
                            info_dict["dep_relation"] = ''
                        res.append(info_dict)
        return pd.DataFrame(res)


class JumanppTokenizer():
    """Juman++で形態素解析を行う

    memo:
        固有名詞のみを抽出するのはデフォルトでは難しそう
        品詞細分類には"地名"や"組織名"といったデータが格納されている（"固有名詞"という形で入っていない）

    sample code:
        result = jumanpp.analysis(text)
        for mrph in result.mrph_list():  # 各形態素にアクセス
            print(f'表層形: {mrph.midasi}\t| 読み: {mrph.yomi}\t| 原型: {mrph.genkei}\t| 品詞: {mrph.hinsi}\
            \t| 品詞細分類: {mrph.bunrui}\t| 活用形1: {mrph.katuyou1}\t| 活用形2: {mrph.katuyou2}\
            \t| 意味情報: {mrph.imis}\t| 代表表記: {mrph.repname}')

    ref:
        https://pyknp.readthedocs.io/en/latest/
    """
    def __init__(self, stopwords=None, include_pos=None):

        self.jumanpp = Juman()

        if stopwords is None:
            self.stopwords = []
        else:
            self.stopwords = stopwords
        if include_pos is None:
            self.include_pos = ["名詞", "動詞", "形容詞"]
        else:
            self.include_pos = include_pos

    def tokenize(self, text, pos=False):
        """指定した品詞でトークナイズを行う

        Args:
            text (str): トークナイズ前のテキスト
            pos (bool, optional): 品詞を合わせて返却するか否か. Defaults to False.

        Returns:
            list: トークナイズを行った単語リスト
        """
        # 前処理
        text = preprocess(text)

        try:
            # 形態素解析
            result = self.jumanpp.analysis(text)
            res = []
            for m in result.mrph_list():  # 各形態素にアクセス
                p = m.hinsi  # 品詞
                base = m.genkei  # 原型
                if p in self.include_pos and base not in self.stopwords:
                    if pos:
                        res.append((base, p))
                    else:
                        res.append(base)
        except:
            import pdb;pdb.set_trace()

        return res


class JanomeTokenizer():
    """Janomeで形態素解析を行う

    sample code:
        janome = Tokenizer()
        result = janome.tokenize(text)
        for m in result:
            print(f'表層形: {m.surface}\t| 読み: {m.reading}\t| 原型: {m.base_form}\t| 品詞: {m.part_of_speech.split(",")[0]}\
            \t| 品詞細分類: {m.part_of_speech}\t| 活用型: {m.infl_type}\t| 活用形: {m.infl_form}')

    ref:
        https://note.nkmk.me/python-janome-tutorial/
    """
    def __init__(self, stopwords=None, include_pos=None):

        self.janome = Tokenizer()

        if stopwords is None:
            self.stopwords = []
        else:
            self.stopwords = stopwords
        if include_pos is None:
            self.include_pos = ["名詞", "動詞", "形容詞"]
        else:
            self.include_pos = include_pos

    def tokenize(self, text, pos=False):
        """指定した品詞でトークナイズを行う

        Args:
            text (str): トークナイズ前のテキスト
            pos (bool, optional): 品詞を合わせて返却するか否か. Defaults to False.

        Returns:
            list: トークナイズを行った単語リスト
        """
        # 前処理
        text = preprocess(text)

        # 形態素解析
        result = self.janome.tokenize(text)
        res = []
        for m in result:
            p = m.part_of_speech.split(',')[0]  # 品詞
            base = m.base_form  # 原型
            if p in self.include_pos and base not in self.stopwords:
                if pos:
                    res.append((base, p))
                else:
                    res.append(base)
        return res

    def tokenize_proper_noun(self, text):
        """固有名詞のみでトークナイズを行う

        Args:
            text (str): トークナイズ前のテキスト

        Returns:
            list: トークナイズを行った単語リスト
        """
        # 前処理
        text = preprocess(text)

        # 形態素解析
        result = self.janome.tokenize(text)
        res = []
        for m in result:
            pos2 = m.part_of_speech.split(',')[1]  # 品詞細分類
            base = m.base_form  # 原型
            if pos2 in PROPER_NOUN and base not in self.stopwords:
                res.append(base)
        return res


class SudachiTokenizer():
    """sudachiで形態素解析を行う

    sample code:
        mode = getattr(tokenizer.Tokenizer.SplitMode, 'C')
        tokenizer_obj = dictionary.Dictionary().create()
        result = tokenizer_obj.tokenize(text, mode)
        for m in tokenizer_obj.tokenize(text, mode):
            print(f'表層形: {m.surface()}\t| 読み: {m.reading_form()}\t| 原型: {m.normalized_form()}\t| 品詞: {m.part_of_speech()[0]}\
            \t| 品詞細分類: {m.part_of_speech()[1]}\t| 活用型: {m.part_of_speech()[4]}\t| 活用形: {m.part_of_speech()[5]}')

    ref:
        https://github.com/WorksApplications/Sudachi
    """
    def __init__(self, mode="C", stopwords=None, include_pos=None):

        if mode not in ["A", "B", "C"]:
            raise Exception("invalid mode. 'A' ,'B' or 'C'")
        self.mode = getattr(tokenizer.Tokenizer.SplitMode, mode)
        self.tokenizer_obj = dictionary.Dictionary().create()

        if stopwords is None:
            self.stopwords = []
        else:
            self.stopwords = stopwords
        if include_pos is None:
            self.include_pos = ["名詞", "動詞", "形容詞"]
        else:
            self.include_pos = include_pos

    def tokenize(self, text, pos=False):
        """指定した品詞でトークナイズを行う

        Args:
            text (str): トークナイズ前のテキスト
            pos (bool, optional): 品詞を合わせて返却するか否か. Defaults to False.

        Returns:
            list: トークナイズを行った単語リスト
        """
        # 前処理
        text = preprocess(text)

        # 形態素解析
        result = self.tokenizer_obj.tokenize(text, self.mode)
        res = []
        for m in result:
            p = m.part_of_speech()
            base = m.normalized_form()  # 原型
            if p[0] in self.include_pos and base not in self.stopwords:
                if pos:
                    res.append((base, p[0]))
                else:
                    res.append(base)
        return res

    def tokenize_proper_noun(self, text, pos=False):
        """固有名詞のみでトークナイズを行う

        Args:
            text (str): トークナイズ前のテキスト

        Returns:
            list: トークナイズを行った単語リスト
        """
        # 前処理
        text = preprocess(text)

        # 形態素解析
        result = self.tokenizer_obj.tokenize(text, self.mode)
        res = []
        for m in result:
            pos2 = m.part_of_speech()[1]  # 品詞細分類
            base = m.normalized_form()  # 原型
            if pos2 in PROPER_NOUN and base not in self.stopwords:
                res.append(base)
        return res
