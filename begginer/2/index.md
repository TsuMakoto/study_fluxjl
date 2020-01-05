# MNITSをもって、Flux.jlを入門する(2)

[Julia Advent Calendar 2019](https://qiita.com/advent-calendar/2019/julialang)の最終目の記事です。

といっても前回の続きみたいな形となってしまい。恐縮です。

本当はTuring.jlについてqiitaで書こうかと思ってましたが、とりあえず一個確実に使えるようになりたかったのでこれにしました。

## データセットが間違ってた

[前回](../1/index.html)の記事では、なんかよくわからず、推論がうまく行かなかったのですが、よくよく見るとデータセットの扱いを勘違いしてました。

## 基本的な事は同じだったんだよなあ

以下の部分まではほとんど問題なかったです。


```julia
# 訓練データの読込
using Flux
using Flux.Data.MNIST
imgs = MNIST.images(:train);

using Flux: onehotbatch
imgs = MNIST.images()
train_X = hcat(float.(vec.(imgs))...)
labels = MNIST.labels(:train)
train_Y = onehotbatch(labels, 0:9)

# 学習モデルの定義
using Flux: Chain
using Flux: Dense
using NNlib: softmax
using NNlib: relu
layeri_1 = Dense(28^2, 32, relu)
layer2_o = Dense(32, 10)
model = Chain(layeri_1, layer2_o, softmax)

# 訓練データを32個ずつに分割
using Base.Iterators: partition
batchsize = 32
serial_iterator = partition(1:size(train_Y)[2], batchsize)
train_dataset = map(batch -> (train_X[:, batch], train_Y[:, batch]), serial_iterator);
```


```julia
size(train_dataset[1][1]), size(train_dataset[1][2])
```




    ((784, 32), (10, 32))



今、上のようになっています。
ここで、[ドキュメント](https://fluxml.ai/Flux.jl/stable/training/training/#Datasets-1)の例をみてみると

---
```julia
x = rand(784)
y = rand(10)
data = [(x, y), (x, y), (x, y)]
# Or equivalently
data = Iterators.repeated((x, y), 3)
```
---
と、上のようになってます。

自分の理解では、`((Xのデータの次元, batchサイズ), (Yのデータの次元, batchサイズ))`で学習を実行するものかと思ってました。

純粋に `(Xのデータの次元, Yのデータの次元)`を引数に与えるんですね。

やはり、ドキュメントをみるのは大事。

## データを作り直し

下のデータセットの作り方は[genkuroki/using Flux.jl.ipynb](https://gist.github.com/genkuroki/49bdba858d4b6c7020f463c648e309f3)を参考にしています。

いつもありがとうございます🙇‍♂️


```julia
dataset = Iterators.repeated((train_X, train_Y), 200)
evalcb = () -> @show(loss(train_X, train_Y))
```




    #43 (generic function with 1 method)



---
普通は一つずつX,Yを設定するため、上のようにそれぞれで定義します。

そして、それをdatasetを準備すると、`(入力画像の次元数, 出力結果の次元数) = (784, 10)`の形で取得できます。

これでできるはず!!

## 学習をする


```julia
using Flux: crossentropy
using Flux: ADAM
using Flux: train!

opt = ADAM()
loss(x, y) = crossentropy(model(x), y)
train!(loss, Flux.params(model), dataset, opt, cb = Flux.throttle(evalcb, 10))
```

    loss(train_X, train_Y) = 2.3338146f0
    loss(train_X, train_Y) = 2.0817525f0
    loss(train_X, train_Y) = 1.8778219f0
    loss(train_X, train_Y) = 1.6883554f0
    loss(train_X, train_Y) = 1.5142949f0
    loss(train_X, train_Y) = 1.3592445f0
    loss(train_X, train_Y) = 1.2230834f0
    loss(train_X, train_Y) = 1.1031209f0
    loss(train_X, train_Y) = 0.9976298f0
    loss(train_X, train_Y) = 0.90592045f0
    loss(train_X, train_Y) = 0.8274122f0
    loss(train_X, train_Y) = 0.76068306f0
    loss(train_X, train_Y) = 0.70442736f0
    loss(train_X, train_Y) = 0.65698516f0
    loss(train_X, train_Y) = 0.6165953f0
    loss(train_X, train_Y) = 0.5819991f0
    loss(train_X, train_Y) = 0.55228555f0
    loss(train_X, train_Y) = 0.5266146f0
    loss(train_X, train_Y) = 0.5042526f0
    loss(train_X, train_Y) = 0.48459056f0
    loss(train_X, train_Y) = 0.4671772f0
    loss(train_X, train_Y) = 0.4516567f0
    loss(train_X, train_Y) = 0.4377649f0
    loss(train_X, train_Y) = 0.42524526f0
    loss(train_X, train_Y) = 0.41390404f0
    loss(train_X, train_Y) = 0.40357998f0
    loss(train_X, train_Y) = 0.39413762f0
    loss(train_X, train_Y) = 0.38546035f0
    loss(train_X, train_Y) = 0.37745515f0
    loss(train_X, train_Y) = 0.37004215f0
    loss(train_X, train_Y) = 0.36315277f0
    loss(train_X, train_Y) = 0.35672128f0
    loss(train_X, train_Y) = 0.35069942f0
    loss(train_X, train_Y) = 0.34504107f0
    loss(train_X, train_Y) = 0.3397092f0
    loss(train_X, train_Y) = 0.33466172f0
    loss(train_X, train_Y) = 0.32986608f0
    loss(train_X, train_Y) = 0.32527816f0
    loss(train_X, train_Y) = 0.32087335f0
    loss(train_X, train_Y) = 0.31662682f0
    loss(train_X, train_Y) = 0.31253207f0
    loss(train_X, train_Y) = 0.30859295f0
    loss(train_X, train_Y) = 0.30481726f0
    loss(train_X, train_Y) = 0.30115998f0
    loss(train_X, train_Y) = 0.29763132f0
    loss(train_X, train_Y) = 0.29422432f0
    loss(train_X, train_Y) = 0.2909183f0
    loss(train_X, train_Y) = 0.28770086f0
    loss(train_X, train_Y) = 0.28456584f0
    loss(train_X, train_Y) = 0.28150916f0


## 推論してみる
うまく行ってくれよ


```julia
using Statistics: mean
test_X = hcat(float.(vec.(MNIST.images(:test)))...)
test_Y = onehotbatch(MNIST.labels(:test), 0:9)
mean(onecold(model(test_X)) .== onecold(test_Y))
```




    0.9239



お、いいっすね！！

ようやく、これで入門できたかな。。。

## おわりに
次はirisとかファッションMNISTとかやってみようかな

とりあえず、アドカレ完走した！！

走り切った！
