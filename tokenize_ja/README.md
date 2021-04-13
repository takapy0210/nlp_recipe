# 日本語における前処理・トークナイズスクリプトをまとめています

## usage

```python
import tokenizer

text = 'hogehoge'

# mecab
mecab_tokenizer = tokenizer.MeCabTokenizer(
    sys_dic_path='/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'
)

# ginza
ginza_tokenizer_A = tokenizer.GinzaTokenizer(mode="A")
ginza_tokenizer_B = tokenizer.GinzaTokenizer(mode="B")
ginza_tokenizer_C = tokenizer.GinzaTokenizer(mode="C")

# Juman++
jumanpp_tokenizer = tokenizer.JumanppTokenizer()

# janome
janome_tokenizer = tokenizer.JanomeTokenizer()

# sudachi
sudachi_tokenizer_A = tokenizer.SudachiTokenizer(mode="A")
sudachi_tokenizer_B = tokenizer.SudachiTokenizer(mode="B")
sudachi_tokenizer_C = tokenizer.SudachiTokenizer(mode="C")

# tokenize
token = mecab_tokenizer.tokenize(text)

# only proper noun
token = mecab_tokenizer.tokenize_proper_noun(text)
```

pd.DataFrameに適応する場合は下記のように記述します

```python
df.loc[:, 'tokenize'] = df['content'].apply(token.tokenize)
```
