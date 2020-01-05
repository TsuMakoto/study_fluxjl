# MNITSをもって、Flux.jlを入門する

[Julia Advent Calendar 2019](https://qiita.com/advent-calendar/2019/julialang)の13日目の記事です。

Juliaの界隈では一番有名と言っても過言ではないでしょう。

Flux.jlという機械学習ライブラリを紹介/入門しようと思います。

## 導入

早速導入していきましょう。
Pkg.jlを使って、インストールします。


```julia
import Pkg; Pkg.add("Flux");
```

    [32m[1m Resolving[22m[39m package versions...
    [32m[1m  Updating[22m[39m `/opt/julia/environments/v1.2/Project.toml`
    [90m [no changes][39m
    [32m[1m  Updating[22m[39m `/opt/julia/environments/v1.2/Manifest.toml`
    [90m [no changes][39m


無事、インストールできました！！

## MNISTのデータセットを取り込む
以前、Juliaのslackのworkspaceにて、MNISTのデータセットはFluxで使えるか？という質問がありました。

意外と知られてないのかもしれないですね。


```julia
# 訓練データの読込
using Flux
using Flux.Data.MNIST
imgs = MNIST.images(:train);
```

例えば、訓練画像として、`train_X`には以下のものが入っています。


```julia
imgs[1]
```




![svg](mnist_files/mnist_7_0.svg)



これを訓練用のデータへ変換し、学習をしていきます。

# 学習モデルの定義


```julia
using Flux: onehotbatch
imgs = MNIST.images(:train)
train_X = hcat(float.(vec.(imgs))...)
labels = MNIST.labels(:train)
train_Y = onehotbatch(labels, 0:9)

# 学習モデルの定義
using Flux: Chain
using Flux: Dense
using NNlib: softmax
using NNlib: relu
layeri_1 = Dense(28^2, 100, relu)
layer1_2 = Dense(100, 100, relu)
layer2_o = Dense(100, 10)
model = Chain(layeri_1, layer1_2, layer2_o, softmax)
```




    Chain(Dense(784, 100, relu), Dense(100, 100, relu), Dense(100, 10), softmax)



---

28x28の学習データから、10個の数字までの2層を定義します。

そして、訓練データをバッチサイズを32個に分割します。

ここは任意なので、まあ、いろいろ試してみてください。


```julia
# 訓練データを32個ずつに分割
using Base.Iterators: partition
batchsize = 32
serial_iterator = partition(1:size(train_Y)[2], batchsize)
train_dataset = map(batch -> (train_X[:, batch], train_Y[:, batch]), serial_iterator);
```

# 学習を実行！！
とりあえず、epochを10回程度回してみましょう。


```julia
# run training
using Flux: crossentropy
using Flux: @epochs
using Flux: ADAM
using Flux: train!
loss(x, y) = crossentropy(model(x), y)
opt = ADAM()
epochs = 10
@epochs epochs train!(loss, Flux.params(model), train_dataset, opt)
```

    ┌ Info: Epoch 1
    └ @ Main /opt/julia/packages/Flux/oX9Pi/src/optimise/train.jl:99
    ┌ Info: Epoch 2
    └ @ Main /opt/julia/packages/Flux/oX9Pi/src/optimise/train.jl:99
    ┌ Info: Epoch 3
    └ @ Main /opt/julia/packages/Flux/oX9Pi/src/optimise/train.jl:99
    ┌ Info: Epoch 4
    └ @ Main /opt/julia/packages/Flux/oX9Pi/src/optimise/train.jl:99
    ┌ Info: Epoch 5
    └ @ Main /opt/julia/packages/Flux/oX9Pi/src/optimise/train.jl:99
    ┌ Info: Epoch 6
    └ @ Main /opt/julia/packages/Flux/oX9Pi/src/optimise/train.jl:99
    ┌ Info: Epoch 7
    └ @ Main /opt/julia/packages/Flux/oX9Pi/src/optimise/train.jl:99
    ┌ Info: Epoch 8
    └ @ Main /opt/julia/packages/Flux/oX9Pi/src/optimise/train.jl:99
    ┌ Info: Epoch 9
    └ @ Main /opt/julia/packages/Flux/oX9Pi/src/optimise/train.jl:99
    ┌ Info: Epoch 10
    └ @ Main /opt/julia/packages/Flux/oX9Pi/src/optimise/train.jl:99


---

GPU使ってないので、こんなんでも割としんどいですね。

Colabとかがいいのかもしれません。無料でGPU枠があるので。

訓練が終わったら、モデルを保存する、ということになっていきます。

## 学習済みモデルの保存と読み込み
この程度なら必要ないですけど、普通はしますよね

https://github.com/FluxML/Flux.jl/blob/master/docs/src/saving.md　でも書いてあるように、BSON.jlというパッケージを使います。


```julia
# モデルの保存
using BSON: @save
pretrained = cpu(model)
weights = params(pretrained)
@save "pretrained.bson" pretrained
@save "weights.bson" weights
```


    UndefRefError: access to undefined reference

    

    Stacktrace:

     [1] getindex at ./array.jl:728 [inlined]

     [2] applychildren!(::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}}, ::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/BSON.jl:28

     [3] _lower_recursive(::Array{Any,1}, ::IdDict{Any,Any}, ::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [4] (::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}})(::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [5] applychildren!(::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}}, ::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/BSON.jl:28

     [6] _lower_recursive(::Array{Any,1}, ::IdDict{Any,Any}, ::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [7] (::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}})(::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [8] applychildren!(::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}}, ::Dict{Symbol,Any}) at /opt/julia/packages/BSON/Ryxwc/src/BSON.jl:21

     [9] _lower_recursive(::IdDict{Any,Nothing}, ::IdDict{Any,Any}, ::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [10] (::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}})(::IdDict{Any,Nothing}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [11] applychildren!(::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}}, ::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/BSON.jl:28

     [12] _lower_recursive(::Array{Any,1}, ::IdDict{Any,Any}, ::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [13] (::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}})(::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [14] applychildren!(::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}}, ::Dict{Symbol,Any}) at /opt/julia/packages/BSON/Ryxwc/src/BSON.jl:21

     [15] _lower_recursive(::Zygote.IdSet{Any}, ::IdDict{Any,Any}, ::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [16] (::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}})(::Zygote.IdSet{Any}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [17] applychildren!(::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}}, ::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/BSON.jl:28

     [18] _lower_recursive(::Array{Any,1}, ::IdDict{Any,Any}, ::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [19] (::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}})(::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [20] applychildren!(::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}}, ::Dict{Symbol,Any}) at /opt/julia/packages/BSON/Ryxwc/src/BSON.jl:21

     [21] _lower_recursive(::Zygote.Params, ::IdDict{Any,Any}, ::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [22] (::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}})(::Zygote.Params) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [23] applychildren!(::getfield(BSON, Symbol("##7#11")){IdDict{Any,Any},Array{Any,1}}, ::Dict{Symbol,Any}) at /opt/julia/packages/BSON/Ryxwc/src/BSON.jl:21

     [24] _lower_recursive(::Dict{Symbol,Zygote.Params}, ::IdDict{Any,Any}, ::Array{Any,1}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:62

     [25] lower_recursive(::Dict{Symbol,Zygote.Params}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:73

     [26] bson(::IOStream, ::Dict{Symbol,Zygote.Params}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:81

     [27] #14 at /opt/julia/packages/BSON/Ryxwc/src/write.jl:83 [inlined]

     [28] #open#312(::Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}}, ::typeof(open), ::getfield(BSON, Symbol("##14#15")){Dict{Symbol,Zygote.Params}}, ::String, ::Vararg{String,N} where N) at ./iostream.jl:375

     [29] open at ./iostream.jl:373 [inlined]

     [30] bson(::String, ::Dict{Symbol,Zygote.Params}) at /opt/julia/packages/BSON/Ryxwc/src/write.jl:83

     [31] top-level scope at In[91]:6



```julia
# モデルのロード
using BSON: @load
@load "pretrained.bson" pretrained
@load "weights.bson" weights
```


    EOFError: read end of file

    

    Stacktrace:

     [1] parse_doc(::IOStream) at ./iostream.jl:408

     [2] parse at /opt/julia/packages/BSON/Ryxwc/src/read.jl:101 [inlined]

     [3] #open#312(::Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}}, ::typeof(open), ::typeof(BSON.parse), ::String) at ./iostream.jl:375

     [4] open at ./iostream.jl:373 [inlined]

     [5] parse at /opt/julia/packages/BSON/Ryxwc/src/read.jl:102 [inlined]

     [6] load(::String) at /opt/julia/packages/BSON/Ryxwc/src/read.jl:104

     [7] top-level scope at /opt/julia/packages/BSON/Ryxwc/src/BSON.jl:52

     [8] top-level scope at In[146]:4


---

ふむ。

どこかの重みがNaNになってるみたいですね。

NaNの部分を0へ変えたらうまくいくのでしょうか。。。？

うーん。。。
割とhttps://github.com/FluxML/Flux.jl/blob/master/docs/src/training/training.md　の通りにしてるんですけどねえ。

とりあえず、このまま行きますか。

# 検証データの取り込み


```julia
# テストデータの読込
imgs = MNIST.images(:test)
test_X = hcat(float.(vec.(imgs))...)
labels = MNIST.labels(:test);
```

基本的に、trainデータの取り込みと同じです。


```julia
imgs[1]
```




![svg](mnist_files/mnist_23_0.svg)



こんな感じですね。
これが7と判定されるとありがたいですねー

## 訓練済みモデルを用いた推論の実行

訓練済みモデル`pretrained`を用いて、推論を実行します。

さて、試しに、上の1番目の画像の推論することにします。


```julia
using Flux: onecold
onecold(pretrained(test_X[:,1]))
```




    1




```julia
pretrained(test_X[:,1])
```




    10-element Array{Float32,1}:
     NaN
     NaN
     NaN
     NaN
     NaN
     NaN
     NaN
     NaN
     NaN
     NaN



---

なん。。。だと。。。

わけわかんねえ。

くそう。まだまだ、勉強が足りないですね。
一旦、制度だけでも求めてみますか。


```julia
using Statistics: mean
mean(onecold(pretrained(test_X[:,1])) .== labels)
```




    0.1135



---
うん。どうも、これ。たまたま正解ラベルに１がついたものだけが成功した結果みたいですね。

まじなんなんだ。

# 重みを用いた推論
あー、上の失敗したから、モチベ上がんねえ。

おまけに、NaNがあるのに、できるのかしら。

やるだけやってみます💪


```julia
p_layeri_1 = Dense(28^2, 100, relu)
p_layer1_2 = Dense(100, 100, relu)
p_layer2_o = Dense(100, 10)
p_model = Chain(layeri_1, layer1_2, layer2_o, softmax)
Flux.loadparams!(p_model, weights)
mean(onecold(model(test_X)) .== labels)
```




    0.1135



---

はい。ダメでした。あーくそ。

全然ダメですね。まじでわかんねえ。

ソースコード追うしかないな。

## 終わりに

全く、成功しなかったので、引き続き勉強していきたいとおもます！

もし、ここはこうしたほうがいいで！とか

ここ間違ってるわ、ぼけえ

とか教えてくれる心優しい方々、[issue](https://github.com/TsuMakoto/study_fluxjl/issues)で投げてください！
