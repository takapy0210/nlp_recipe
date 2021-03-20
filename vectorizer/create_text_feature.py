import umap
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
import multiprocessing
import gensim


class Text2Vec(object):
    """テキストデータをベクトル化するクラス
    下記方法でのベクトル化を実装済み
        - テキストの言語情報
        - テキストの長さ
        - 単語カウント
        - 特定の単語が含まれるか田否か
        - tfidf with [SVD/PCA/UMAP/TSNE]
        - LDA
        - word2vec（skip-gram）

    Args:
        object ([type]): [description]
    """

    def __init__(self, input_df: pd.DataFrame, seed=42):
        self.df = input_df
        self.seed = seed

    def language_type(self, col: str, fasttext_model) -> pd.DataFrame:
        """fasttextを用いてテキストの言語を判定する

        Args:
            col (str): 対象のカラム. スペース区切りのstr型を想定
            fasttext_model (model): fast_textのモデル. ref. https://fasttext.cc/docs/en/language-identification.html
                呼び出し元でのロード方法
                    from fasttext import load_model
                    model = load_model(EXTERNAL_DIR_NAME + 'lid.176.bin')

        Returns:
            pd.DataFrame: 言語ラベルを付与したDF
        """
        output_df = pd.DataFrame()
        output_df[f'{col}_lang_ft'] = self.df[col].fillna('')\
            .map(lambda x: fasttext_model.predict(x.replace("\n", ""))[0][0])
        return output_df

    def text_length(self, col: str) -> pd.DataFrame:
        """テキストの長さを計算する

        Args:
            col (str): 対象のカラム. スペース区切りのstr型を想定

        Returns:
            pd.DataFrame: 長さカラムを追加したDF. 単純な長さと単語数のDF
        """
        output_df = pd.DataFrame()
        # 文字数と単語数
        output_df[f'{col}_length'] = self.df[col].str.len()
        output_df[f'{col}_word_count'] = self.df[col].astype(str).map(lambda x: len(x.split()))
        return output_df

    def contain_word(self, col: str, words: list) -> pd.DataFrame:
        """特定の単語がテキストに存在するか否かの特徴量を生成する

        Args:
            col (str): 対象のカラム. スペース区切りのstr型を想定
            words (list of str): 対象の単語リストの

        Returns:
            pd.DataFrame: 各単語ごとに0 or 1のフラグを付与したDF
        """
        output_df = pd.DataFrame()
        for word in words:
            output_df[f'{col}_in_{word}'] = self.df[col].apply(lambda x: word in x.lower())
        # 0 or 1に変換
        output_df = output_df*1
        return output_df

    def tfidf_vec(self, col: str, dim_size: int = 50, decomposition: str = 'SVD') -> pd.DataFrame:
        """tfidfに変換したベクトルを取得する
        colに指定されたカラムを対象として、TFIDFでベクトル化する

        Args:
            col (str): TF-IDFでベクトル化するカラム. スペース区切りのstr型を想定
            compression (str): 圧縮タイプ. [None, SVD, UMAP, TSNE]のいずれかに対応. Noneを指定すると生のTF-IDF値が返却される
                TSNEの場合はdim_size=2にしないとエラーとなるので注意

        Returns:
            pd.DataFrame: [description]
        """
        output_df = pd.DataFrame()
        token = self.df[col].values.tolist()
        tfidf_vec = TfidfVectorizer().fit_transform(token).toarray()

        if decomposition is None:
            # 圧縮しない場合はTFIDF値をDFに変換してそのままreturn
            output_df = pd.DataFrame(tfidf_vec)

        elif decomposition == 'SVD':
            svd = TruncatedSVD(n_components=dim_size, random_state=self.seed)
            svd_df = svd.fit_transform(tfidf_vec)
            output_df = pd.DataFrame(svd_df).add_prefix(f'{col}_svd_')

        elif decomposition == 'PCA':
            pca = PCA(n_components=dim_size, random_state=self.seed)
            pca_df = pca.fit_transform(tfidf_vec)
            output_df = pd.DataFrame(pca_df).add_prefix(f'{col}_pca_')

        elif decomposition == 'UMAP':
            # 結構メモリ使います
            um = umap.UMAP(n_components=dim_size, random_state=self.seed)
            umap_df = um.fit_transform(tfidf_vec)
            output_df = pd.DataFrame(umap_df).add_prefix(f'{col}_umap_')

        elif decomposition == 'TSNE':
            # 結構時間かかります
            tsne = TSNE(n_components=dim_size, random_state=self.seed)
            tsne_df = tsne.fit_transform(tfidf_vec)
            output_df = pd.DataFrame(tsne_df).add_prefix(f'{col}_tsne_')

        return output_df

    def lda_vec(self, col: str, topic_size: int = 20, passes: int = 10) -> pd.DataFrame:
        """LDAを用いてトピックベクトルを取得する

        Args:
            col (str): 対象のカラム. スペース区切りのstr型を想定
            topic_size (int, optional): トピック数. Defaults to 20.
            passes (int, optional): passes数. Defaults to 10.

        Returns:
            pd.DataFrame: トピック分布DF
        """
        # 辞書の作成
        texts = self.df[col].apply(lambda x: x.split())
        dic = gensim.corpora.Dictionary(texts)
        bow_corpus = [dic.doc2bow(doc) for doc in texts]

        # CPUのコア数を取得する
        cpu_count = multiprocessing.cpu_count()
        ldamodel = gensim.models.LdaMulticore(
            bow_corpus,
            num_topics=topic_size,
            id2word=dic,
            workers=cpu_count,
            passes=passes,
            random_state=self.seed
        )

        # 全文書のトピック分布を取得
        all_topics = ldamodel.get_document_topics(bow_corpus, minimum_probability=0)
        temp_dict = {}
        for i, t in enumerate(all_topics):
            weight = [x[1] for x in t]  # タプル重みだけを取り出したリストを作成
            temp_dict[i] = weight

        # DFに変換
        output_df = pd.DataFrame.from_dict(temp_dict, orient='index')
        output_df = output_df.add_prefix(f'{col}_topic_')

        return output_df
