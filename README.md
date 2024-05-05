# cartoon-segmentation-docker

[CartoonSegmentation](https://github.com/CartoonSegmentation/CartoonSegmentation)を使って簡単に背景削除できるようにしたスクリプト

# 事前準備

https://huggingface.co/dreMaz/AnimeInstanceSegmentation から refine_last.ckpt と rtmdetl_e60.ckpt をダウンロードして models/AnimeInstanceSegmentation に保存する。

# 起動方法

imagesディレクトリに背景除去したい画像を入れて以下のコマンドを実行する

```
$ docker compose up -d
```

すると背景除去された画像がimages_outputディレクトリに一括保存される。
