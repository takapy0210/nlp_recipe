from __future__ import annotations
import numpy as np
from scipy import sparse as sp


def is_key_exist(d: dict, key: any):
    return d.get(key) is not None


class Okapi:
    def __init__(self, b: float = 0.75, k1: float = 2.0, delta: float = 1.0, norm: bool = True):
        self.K1, self.B, self.delta = k1, b, delta  # 定数
        self.norm = norm  # 正規化するかしないか
        self.word2id_dict = {}  # 単語とインデックスの辞書
        self.idf = np.array([])  # inverse document frequency
        self.avg_word_count_in_doc = 0  # ドキュメント内の単語数の平均

    def fit_transform(self, documents: list[str]) -> sp.lil_matrix:
        """fitとtransformを行う
        """
        self.fit(documents)
        return self.transform(documents)

    def fit(self, documents: list[str]):
        """ベクトル化のセットアップ
        """
        counter = 0
        for document in documents:
            searched_dict = {}
            words = document
            self.avg_word_count_in_doc += len(words)
            for word in words:
                if is_key_exist(searched_dict, word):
                    continue
                searched_dict[word] = True
                # 他のドキュメントですでに出た単語
                if is_key_exist(self.word2id_dict, word):
                    self.idf[self.word2id_dict[word]] += 1.0
                    continue
                self.word2id_dict[word] = counter
                counter += 1
                self.idf = np.append(self.idf, [1.0])
        documents_len = len(documents)
        self.idf = np.log2(documents_len / (self.idf + 0.0000001))  # logに00が入らないようにする
        self.avg_word_count_in_doc = self.avg_word_count_in_doc / documents_len

    def transform(self, documents: list[str]) -> sp.lil_matrix:
        """ドキュメントを重み付け
        """
        result = sp.lil_matrix((len(documents), len(self.word2id_dict)))
        for i, doc in enumerate(documents):
            # 単語の出現頻度
            word_weight_dict, words_count = self._terms_frequency(doc)
            # Combine Weight重み付け
            for ind in word_weight_dict.keys():
                word_weight_dict[ind] = self._bm25_weight(ind, word_weight_dict[ind], words_count)

            if self.norm:
                # 正規化
                total_dist = sum(list(map(lambda item: item[1], word_weight_dict.items())))
                for ind in word_weight_dict.keys():
                    word_weight_dict[ind] /= total_dist

            # 疎行列にベクトル追加
            for item in word_weight_dict.items():
                result[i, item[0]] = item[1]
        return result

    def _terms_frequency(self, doc: str) -> tuple[dict[int, float], int]:
        """ドキュメント内の単語出現頻度を返す
        """
        word_weight_dict = {}  # key: 単語ID, value: 頻度
        words = doc

        # Term Frequency
        for word in words:
            if not is_key_exist(self.word2id_dict, word):
                # 辞書に無い単語の扱い
                continue

            if is_key_exist(word_weight_dict, self.word2id_dict[word]):
                word_weight_dict[self.word2id_dict[word]] += 1.0
            else:
                word_weight_dict[self.word2id_dict[word]] = 1.0
        return word_weight_dict, len(words)

    def _bm25_weight(self, word_index: int, word_freq: float, word_count_in_doc: int) -> float:
        """Okapi BM25+重み計算
        """
        return self.idf[word_index] * (self.delta + (word_freq * (self.K1 + 1.0))) / (
                word_freq + self.K1 * (1.0 - self.B + self.B * (word_count_in_doc / self.avg_word_count_in_doc)))

    def get_feature_names(self) -> list[str]:
        """重み付けする単語リストを返す
        """
        return list(self.word2id_dict.keys())
