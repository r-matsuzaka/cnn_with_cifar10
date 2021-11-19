# cnn_with_cifar10

## コミット前に
```
make format
make lint
```

## データの準備
[The CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)の`CIFAR-10 python version`を選択して、`cifar-10-python.tar.gz`をダウンロード。
レポジトリ直下に置く。

`tar -zxvf cifar-10-python.tar.gz`
で解凍

`python unpack.py`
で訓練データとテストデータに分割して展開


