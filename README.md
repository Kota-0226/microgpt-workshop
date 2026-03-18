# MicroGPT Workshop

Karpathy の [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) をベースに、GPT の仕組みを体験するハンズオン勉強会です。

データセットを **日本語のひらがな名前** に変更しているので、生成される名前も日本語になります。

勉強会スライド: [Google Slides](https://docs.google.com/presentation/d/1NuXir096PdWesgxzEzfiCWBodf8fqHHC8uitAMy91WE/edit?usp=drive_link)

※ 2WINSアカウントでのみ閲覧可能。

## ファイル構成

```
microgpt-workshop/
├── README.md              ← このファイル
├── pyproject.toml         ← プロジェクト設定（uv 用）
├── japanese_names.txt     ← 日本語ひらがな名前データセット（約 230 件）
├── microgpt_train.py      ← 学習スクリプト（GPT 学習 + wandb ログ + 重み保存）
├── microgpt_generate.py   ← 推論スクリプト（学習済み重みから名前生成）
└── weights/               ← 学習済みの重みファイル（.pkl）が保存される
```

## セットアップ

### 1. uv のインストール

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 依存関係のインストール

```bash
uv sync
```

### 3. wandb の設定（オプション）

wandb がなくても学習は動きますが、あるとダッシュボードで学習曲線をリアルタイム確認できます。

```bash
wandb login
```

## 使い方

### 学習（microgpt_train.py）

```bash
# デフォルト設定で学習（1000 ステップ）
uv run microgpt_train.py

# ハイパーパラメータを変更して学習
uv run microgpt_train.py --n_embd 64 --n_head 8
uv run microgpt_train.py --num_steps 2000 --lr 0.005
uv run microgpt_train.py --n_layer 2 --num_steps 3000
```

学習完了後、重みが `weights/L{n_layer}_E{n_embd}_H{n_head}_S{num_steps}.pkl` に保存されます。

#### 学習のオプション一覧

| オプション       | デフォルト | 説明               |
| ---------------- | ---------- | ------------------ |
| `--n_layer`    | 1          | Transformer の層数 |
| `--n_embd`     | 48         | 埋め込み次元数     |
| `--block_size` | 16         | 最大コンテキスト長 |
| `--n_head`     | 4          | Attention ヘッド数 |
| `--num_steps`  | 1000       | 学習ステップ数     |
| `--lr`         | 0.01       | 学習率             |
| `--seed`       | 42         | 乱数シード         |

### 推論（microgpt_generate.py）

```bash
# 最新の重みでランダムに名前を生成
uv run microgpt_generate.py

# 「さ」から始まる名前を生成
uv run microgpt_generate.py --start さ

# 「ゆ」から始まる名前を 10 個生成
uv run microgpt_generate.py --start ゆ -n 10

# 特定の重みファイルを指定
uv run microgpt_generate.py --weights weights/L1_E64_H8_S2000.pkl

# 生成の多様性を上げる
uv run microgpt_generate.py --temperature 0.8
```

#### 推論のオプション一覧

| オプション         | デフォルト | 説明                                   |
| ------------------ | ---------- | -------------------------------------- |
| `--start`        | なし       | 最初の一文字（ひらがな）               |
| `-n` / `--num` | 20         | 生成する名前の数                       |
| `--temperature`  | 0.5        | 生成の多様性（低い→確実、高い→多様） |
| `--weights`      | 最新の重み | 重みファイルのパス                     |

## 参加者向け事前準備

- Python 3.10+
- uv（上記の手順でインストール）
- wandb アカウント（https://wandb.ai）- オプション
