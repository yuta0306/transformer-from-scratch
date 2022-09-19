# transformer-from-scratch

研究室インターンの対応をした時のTransformerをスクラッチから作るプログラム群

## データセット

[Tab-delimited Bilingual Sentence Pairs](http://www.manythings.org/anki/)

上記リンクの指すサイト上に**Japanese - English**とかいたデータがあるので，ダウンロードされたzip形式のデータを解凍し，data/以下にjpn.txtを配置する

## How to Use

```python
python main.py
```
で学習及びテストが回る仕様．

*main.py*の中でいくつかconfigを自由に設定して使う

```python
# example
if __name__ == "__main__":
    df = load_txt("data/jpn.txt")
    traindf, testdf = split_data(df)
    train("Helsinki-NLP/opus-mt-ja-en", traindf, testdf, batch_size=16, epoch=10)
```