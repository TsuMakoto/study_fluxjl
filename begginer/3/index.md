# 自然言語処理をやってみる vol.1

ひさびさに書いてみます。

今回は、自然言語処理について書いてみようかなと思います。

## 動機について少々

この記事を書く気になったのはqiitaでCOTOHA APIという自然言語のためのAPIを使用し、記事にするといい記事を書いた人にはプレゼントがあるという優れたものがあるからです！！

まあ、プレゼントはもらえないにせよ、記事を書くことに意味があるので、やっていきまする。

## COTOHA API for Developersに無料登録
これをやっていきます。
[COTOHA API for Developers](https://api.ce-cotoha.com/contents/developers/index.html)のリンクからやっていきましょう！

結構めんどくさいですが、そこでのAPI Key等はメモっておいてください。

## 早速アクセスしましょう！

access_tokenはアクセスするたびに更新され、24時間の有効期限があるので、お試しする際はお気をつけて。


```julia
using HTTP
using JSON
```


```julia
include("secret.jl")
data = Dict("grantType"    => "client_credentials",
            "clientId"     => clientid, 
            "clientSecret" => clientsecret)
header = ["Content-Type"   => "application/json"]

result = HTTP.request("POST", access_token_publish_url, header, JSON.json(data))
```




    HTTP.Messages.Response:
    """
    HTTP/1.1 201 Created
    Date: Sun, 09 Feb 2020 03:46:44 GMT
    Content-Type: application/json
    Content-Length: 249
    Connection: keep-alive
    Access-Control-Allow-Origin: 
    
    
              {
                "access_token": "uETK8l84XlxxXlWw6fyoHGcGaGHe", 
                "token_type": "bearer",
                "expires_in": "86399" ,
                "scope": "" ,    
                "issued_at": "1581220004395"           
               }
            """




```julia
access_token = JSON.parse(String(result.body))["access_token"]
```




    "uETK8l84XlxxXlWw6fyoHGcGaGHe"




```julia
using Printf
```


```julia
function post(access_token, data, api_base_url) 
    header = ["Content-Type"  => "application/json;charset=UTF-8",
              "Authorization" => "Bearer $(access_token)"]
    
    HTTP.request("POST", "$(api_base_url)/nlp/v1/parse", header, JSON.json(data))
end
```




    post (generic function with 1 method)




```julia
result_json = post(access_token, Dict("sentence" => "昨日母と銀座で焼き肉を食べた"), api_base_url)
```




    HTTP.Messages.Response:
    """
    HTTP/1.1 200 OK
    Date: Sun, 09 Feb 2020 03:46:46 GMT
    Content-Type: application/json;charset=utf-8
    Content-Length: 3460
    Connection: keep-alive
    Cache-Control: no-cache
    Pragma: no-cache
    X-Frame-Options: DENY
    X-Content-Type-Options: nosniff
    X-XSS-Protection: 1; mode=block
    Via: 1.1 google
    Access-Control-Allow-Origin: 
    
    {
      "result" : [ {
        "chunk_info" : {
          "id" : 0,
          "head" : 4,
          "dep" : "D",
          "chunk_head" : 0,
          "chunk_func" : 0,
          "links" : [ ]
        },
        "tokens" : [ {
          "id" : 0,
          "form" : "昨日",
          "kana" : "サクジツ",
          "lemma" : "昨日",
          "pos" : "名詞",
          "features" : [ "日時" ],
          "dependency_labels" : [ ],
          "attributes" : { }
        } ]
      }, {
        "chunk_info" : {
          "id" : 1,
          "head" : 4,
          "dep" : "D",
          "chunk_head" : 0,
          "chunk_func" : 1,
          "links" : [ ]
        },
        "tokens" : [ {
          "id" : 1,
          "form" : "母",
          "kana" : "ハハ",
          "lemma" : "母",
          "pos" : "名詞",
          "features" : [ ],
          "dependency_labels" : [ {
            "token_id" : 2,
            "label" : "cc"
          } ],
          "attributes" : { }
        }, {
          "id" : 2,
          "form" : "と",
          "kana" : "ト",
          "lemma" : "と",
          "pos" : "格助詞",
          "features" : [ "連用" ],
          "attribute
    ⋮
    3460-byte body
    """



---
上のような結果が返ってきます。

ちなみに、この例は[スタートガイド](https://api.ce-cotoha.com/contents/gettingStarted.html)で載ってる例です。

これをJuliaで扱えるようにparseして中身を覗いてみます。


```julia
result = JSON.parse(String(result_json.body))["result"];
```


```julia
print(result[1])
```

    Dict{String,Any}("tokens" => Any[Dict{String,Any}("features" => Any["日時"],"attributes" => Dict{String,Any}(),"kana" => "サクジツ","id" => 0,"lemma" => "昨日","pos" => "名詞","form" => "昨日","dependency_labels" => Any[])],"chunk_info" => Dict{String,Any}("head" => 4,"links" => Any[],"chunk_head" => 0,"chunk_func" => 0,"id" => 0,"dep" => "D"))

まあ、こんな感じで、文章解析ができるようになります。

Mecabよりも結構多い情報がえられてますが、今のところあまり違いは感じられないかな？

むしろ、情報が多すぎて選択しにくいのが煩わしいというか、Optionありそうですけどね。

今回はこんな感じで。

いずれはセンター国語、評論くらい解けるようにしたいですね。
