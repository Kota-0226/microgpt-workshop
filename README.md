# MicroGPT × wandb × 量子化 ハンズオン勉強会

## 概要

Karpathy の [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) をベースに、以下の 3 つのテーマを **1 本のハンズオン** として体験します。

データセットを **日本語のひらがな名前** に変更しているので、生成される名前も日本語です。

```
microgpt.py + wandb: GPT の仕組みを理解しつつ学習をリアルタイム可視化
        ↓
学習済みモデルの重みを量子化して軽量化を体感
```

## ファイル構成

```
study-session/
├── README.md                    ← このファイル
└── docs/
    ├── japanese_names.txt       ← 日本語ひらがな名前データセット（約 220 件）
    ├── 01_microgpt.py           ← Karpathy microgpt 完全解説版 + wandb ログ
    └── 02_quantization.py       ← 量子化デモ（FP32 vs INT8 比較）
```

## セットアップ & 実行

### 01_microgpt.py

```bash
# wandb なしでも動く（pure Python、外部ライブラリ不要）
cd docs
python 01_microgpt.py

# wandb ありで実行する場合
pip install wandb
wandb login
python 01_microgpt.py
# → ダッシュボードで train/loss, perplexity をリアルタイム確認
```

### 02_quantization.py

```bash
pip install torch wandb
python 02_quantization.py
```

## 参加者向け事前準備

- Python 3.9+
- wandb アカウント（https://wandb.ai）
- 01 は外部ライブラリ不要、02 は PyTorch が必要

## 01_microgpt.py の構成

| セクション | 内容 |
|-----------|------|
| データセット | japanese_names.txt（ひらがな名前）を文字レベルでトークン化 |
| Autograd | `Value` クラスで計算グラフ構築 → `backward()` で自動微分 |
| アーキテクチャ | GPT-2 ベース: RMSNorm, Multi-Head Attention, MLP, 残差接続 |
| 学習 | Adam + 線形学習率減衰、1,000 ステップ |
| wandb | train/loss, perplexity, lr をリアルタイム記録（オプショナル） |
| 推論 | temperature 制御で新しい日本語名前を生成 |
