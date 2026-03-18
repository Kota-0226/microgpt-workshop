"""
microgpt_train.py - Karpathy's microgpt 完全解説版（日本語名前データセット + wandb）

原典: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

"The most atomic way to train and run inference for a GPT
 in pure, dependency-free Python.
 This file is the complete algorithm. Everything else is just efficiency."
                                                            — @karpathy

このファイルは Karpathy の microgpt の全コードに日本語の解説コメントを付けた学習スクリプトです。
wandb がインストールされていれば、学習ログをリアルタイムで可視化できます。
推論（テキスト生成）は microgpt_generate.py で行います。

変更点（オリジナルとの差分）:
  - データセットを英語の名前 → 日本語のひらがな名前に変更
  - 語彙がアルファベット 26 文字 → ひらがな約 70 文字に拡大
  - 学習と推論を別ファイルに分離（学習: microgpt_train.py、推論: microgpt_generate.py）
  - wandb によるログ記録を追加（オプショナル: なくても動く）

構成:
  1. データセットの読み込み & トークナイザ
  2. Autograd エンジン（Value クラス）
  3. パラメータの初期化
  4. モデルアーキテクチャ（GPT-2 ベース）
  5. 学習ループ（Adam オプティマイザ + wandb ログ）
  6. 重みの保存
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
import argparse # コマンドライン引数
import pickle   # 学習済みの重みを保存・読み込みするためのシリアライズライブラリ

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# =============================================================================
# コマンドライン引数（ハイパーパラメータ）
# =============================================================================
# uv run microgpt_train.py                          # デフォルト設定で学習
# uv run microgpt_train.py --n_embd 64 --n_head 8   # 埋め込み次元・ヘッド数を変更
# uv run microgpt_train.py --num_steps 2000 --lr 0.005

parser = argparse.ArgumentParser(description="MicroGPT 学習スクリプト")
parser.add_argument('--n_layer', type=int, default=1, help='Transformer の層数（デフォルト: 1）')
parser.add_argument('--n_embd', type=int, default=48, help='埋め込み次元数（デフォルト: 48）')
parser.add_argument('--block_size', type=int, default=16, help='最大コンテキスト長（デフォルト: 16）')
parser.add_argument('--n_head', type=int, default=4, help='Attention ヘッド数（デフォルト: 4）')
parser.add_argument('--num_steps', type=int, default=1000, help='学習ステップ数（デフォルト: 1000）')
parser.add_argument('--lr', type=float, default=0.01, help='学習率（デフォルト: 0.01）')
parser.add_argument('--seed', type=int, default=42, help='乱数シード（デフォルト: 42）')
args = parser.parse_args()

random.seed(args.seed)  # 再現性のためシードを固定


# =============================================================================
# 0. データセット: 日本語のひらがな名前
# =============================================================================
# japanese_names.txt を読み込む。
# 各行が1つの名前（例: "はると", "ゆい", "さくら"）。
# このモデルはひらがな名前のパターンを学習し、新しい名前を生成する。
#
# オリジナルは英語の names.txt（a-z の 26 文字、約 32,000 件）。
# 日本語版はひらがな（約 70 文字）を使うため語彙が大きくなるが、
# Transformer の仕組み自体は全く同じ。

NAMES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'japanese_names.txt')

if not os.path.exists(NAMES_FILE):
    # フォールバック: 同じディレクトリになければカレントディレクトリを試す
    NAMES_FILE = 'japanese_names.txt'

if not os.path.exists(NAMES_FILE):
    # それでもなければオリジナルの英語版にフォールバック
    print("japanese_names.txt が見つかりません。オリジナルの英語版を使用します。")
    if not os.path.exists('input.txt'):
        import urllib.request
        names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
        urllib.request.urlretrieve(names_url, 'input.txt')
    NAMES_FILE = 'input.txt'

docs = [line.strip() for line in open(NAMES_FILE, encoding='utf-8') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
# => 約 220 個の日本語名前


# =============================================================================
# 0.a トークナイザ: 文字 → 整数 ID への変換のための準備
# =============================================================================
#
# ステップ 1: 元データ（文字列）から文字を集める
#   docs = ["はると", "ゆい", "さくら", ...]
#          ↓ 全文字列を結合 → 重複を除去 → ソート
#   uchars = ['あ', 'い', 'う', ..., 'ん']
#            各文字のインデックスがそのまま ID になる
#            例: 'あ'→0, 'い'→1, 'う'→2, ...
#
uchars = sorted(set(''.join(docs)))

# ステップ 2: 特別トークン（BOS）を追加
#   BOS = len(uchars)  ← 文字 ID の次の番号を割り当て
#   例: ひらがなが 70 種類なら BOS = 70
#   BOS は文の開始と終了の両方に使う区切り記号
#
BOS = len(uchars)

# ステップ 3: 語彙数（vocab_size）を決める
#   vocab_size = 文字の種類数 + BOS の 1 つ
#   例: 70 文字 + 1 (BOS) = 71
#
#   トークン化の例:
#     "さくら" → [BOS, さのID, くのID, らのID, BOS]
#
vocab_size = len(uchars) + 1

print(f"vocab size: {vocab_size}")
print(f"characters: {''.join(uchars[:20])}...")  # 最初の 20 文字を表示


# =============================================================================
# 2.d Autograd エンジン: 自動微分
# =============================================================================
# Value クラスは「計算グラフのノード」。
# 順伝播でグラフを構築し、逆伝播で勾配を自動計算する。
#
# 仕組み:
#   - 各演算（+, *, ** など）が新しい Value ノードを作る
#   - ノードは子ノード（_children）と局所勾配（_local_grads）を記憶
#   - backward() でトポロジカルソート → 逆順に連鎖律を適用
#
# 例: z = x * y の場合
#   ∂z/∂x = y（局所勾配）
#   ∂L/∂x = ∂L/∂z * ∂z/∂x = z.grad * y（連鎖律）

class Value:
    # __slots__ でメモリ使用量を最適化（数万ノードを作るため重要）
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data              # このノードのスカラー値（順伝播で計算）
        self.grad = 0                 # 損失に対するこのノードの勾配（逆伝播で計算）
        self._children = children     # 計算グラフ上の子ノード
        self._local_grads = local_grads  # 子ノードに対する局所的な偏微分値

    # --- 算術演算: 順伝播 + 局所勾配の記録 ---

    def __add__(self, other):
        # z = x + y → ∂z/∂x = 1, ∂z/∂y = 1
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        # z = x * y → ∂z/∂x = y, ∂z/∂y = x
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # z = x^n → ∂z/∂x = n * x^(n-1)
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        # z = log(x) → ∂z/∂x = 1/x
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        # z = exp(x) → ∂z/∂x = exp(x)
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        # z = max(0, x) → ∂z/∂x = 1 (if x > 0) else 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # --- ヘルパー演算（上の演算の組み合わせ）---
    def __neg__(self):       return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other):  return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):  return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """
        逆伝播: 損失からすべてのパラメータへの勾配を計算する。

        手順:
          1. トポロジカルソートで計算グラフをノードの依存順に並べる
          2. 逆順にたどり、連鎖律で勾配を伝播する
             child.grad += local_grad * v.grad
        """
        # トポロジカルソート（深さ優先探索）
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 逆伝播: 出力（self）の勾配を 1 に設定し、逆順に伝播
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad  # 連鎖律（chain rule）


# =============================================================================
# パラメータの初期化
# =============================================================================
# GPT の「知識」はすべてこのパラメータ（重み行列）に格納される。
# 初期値はガウス分布からのランダムな小さい値（std=0.08）。

n_layer = args.n_layer           # Transformer の層数（深さ）
n_embd = args.n_embd            # 埋め込み次元数（幅）※語彙増加に合わせて拡大
block_size = args.block_size     # 最大コンテキスト長（注意の窓）
n_head = args.n_head            # Attention ヘッド数
head_dim = n_embd // n_head      # 各ヘッドの次元数
learning_rate = args.lr          # 学習率
num_steps = args.num_steps       # 学習ステップ数

# 行列を初期化するヘルパー関数
# 各パラメータは Value オブジェクト → 自動微分が可能
def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# state_dict: PyTorch 風のパラメータ辞書
state_dict = {
    'wte': matrix(vocab_size, n_embd),      # Token Embedding: 各トークンを n_embd 次元ベクトルに変換
    'wpe': matrix(block_size, n_embd),       # Position Embedding: 位置情報を n_embd 次元ベクトルに変換
    'lm_head': matrix(vocab_size, n_embd),   # 出力層: n_embd 次元 → 語彙サイズの logits
}

for i in range(n_layer):
    # 各 Transformer 層のパラメータ
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)   # Query 投影
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)   # Key 投影
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)   # Value 投影
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)   # 出力投影
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # MLP 拡張層（×4 に拡張）
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # MLP 圧縮層（元に戻す）

# 全パラメータをフラットなリストに展開（オプティマイザ用）
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")


# =============================================================================
# wandb の初期化（オプショナル）
# =============================================================================
# wandb.init() でプロジェクトとランを初期化。
# config にハイパーパラメータを渡すと、ダッシュボードで自動記録される。
# → 後から「どの設定でどの結果が出たか」を比較できる。
if HAS_WANDB:
    wandb.init(
        project="workshop-microgpt",
        name=f"jp-names_L{n_layer}_E{n_embd}_H{n_head}",
        config={
            "n_layer": n_layer,
            "n_embd": n_embd,
            "n_head": n_head,
            "head_dim": head_dim,
            "block_size": block_size,
            "vocab_size": vocab_size,
            "learning_rate": learning_rate,
            "num_steps": num_steps,
            "dataset": "japanese_names",
            "num_docs": len(docs),
        },
    )
    print("wandb 初期化完了！ダッシュボードで学習を監視できます。")


# =============================================================================
# 1. モデルアーキテクチャ: GPT-2 ベース（ミニマル版）
# =============================================================================
# GPT-2 を忠実に再現。ただし以下の簡略化:
#   - LayerNorm → RMSNorm（バイアス不要で簡潔）
#   - GeLU → ReLU（実装が簡単）
#   - バイアスなし

def linear(x, w):
    """
    線形変換: y = xW^T
    x: 入力ベクトル [n_in]
    w: 重み行列 [n_out, n_in]
    戻り値: [n_out]
    """
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    """
    Softmax: logits → 確率分布
    数値安定性のため max を引いてからexp を取る（オーバーフロー防止）。
    """
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    """
    RMSNorm（Root Mean Square Normalization）
    LayerNorm の簡略版。平均を引かず、二乗平均平方根でスケーリングするだけ。
    バイアスパラメータが不要なので実装がシンプル。

    計算: x_i / sqrt(mean(x^2) + eps)
    """
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values):
    """
    GPT の順伝播: 1トークンずつ処理する（自己回帰的）。

    引数:
      token_id: 現在のトークン ID（整数）
      pos_id:   現在の位置（整数）
      keys:     各層の Key キャッシュ（KV キャッシュ）
      values:   各層の Value キャッシュ

    戻り値:
      logits: 次のトークンの確率分布（softmax 前）[vocab_size]

    KV キャッシュの仕組み:
      過去のトークンの K, V を保存しておくことで、
      新しいトークンを処理する際に再計算が不要になる。
      これが GPT の推論を効率的にする仕組み。
    """
    # --- 1.a 文字のベクトル化（embedding） ---
    tok_emb = state_dict['wte'][token_id]  # Token Embedding: 文字 ID → n_embd 次元ベクトル

    # --- 1.b 位置情報付加 ---
    pos_emb = state_dict['wpe'][pos_id]    # Position Embedding: 位置 → n_embd 次元ベクトル
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 1.a + 1.b を合成（文字の意味 + 位置の情報）

    x = rmsnorm(x)

    for li in range(n_layer):
        # ================================================================
        # 1.c Multi-Head Self-Attention
        # ================================================================
        # 「どのトークンに注目すべきか」を学習する仕組み、、スライド参照
        x_residual = x  # 残差接続用に保存
        x = rmsnorm(x)

        # Q, K, V を線形変換で計算
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # Query: 「何を探しているか」
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # Key:   「何を持っているか」
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # Value: 「どんな情報を渡すか」

        # KV キャッシュに追加（過去のトークンの K, V を保持）
        keys[li].append(k)
        values[li].append(v)

        # Multi-Head Attention: ヘッドごとに独立して Attention を計算
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim  # このヘッドの開始インデックス

            # このヘッドに対応する Q, K, V の部分を切り出す
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]    # 過去の全 K
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]  # 過去の全 V

            # Attention スコア = Q・K^T / sqrt(d_k)
            # スケーリングで勾配の安定性を確保
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]

            # Softmax → Attention 重み（確率分布）
            attn_weights = softmax(attn_logits)

            # 重み付き和で Value を集約
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)  # ヘッドの出力を結合

        # 出力投影 + 残差接続
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]  # 残差接続: 勾配消失を防ぐ

        # ================================================================
        # 1.d MLP（Feed-Forward Network）
        # ================================================================
        # 「各トークンの表現を非線形に変換する」仕組み
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])   # n_embd → 4*n_embd に拡張
        x = [xi.relu() for xi in x]                        # ReLU 活性化関数
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])   # 4*n_embd → n_embd に圧縮
        x = [a + b for a, b in zip(x, x_residual)]         # 残差接続

    # 1.e 最終出力: n_embd → vocab_size（語彙数） の logits（'あ' : 0.1, 'い' : 0.02, ... のようなスコア）
    logits = linear(x, state_dict['lm_head'])
    return logits


# =============================================================================
# 2. Adam オプティマイザ & 学習ループ
# =============================================================================
# Adam: 学習率を各パラメータごとに適応的に調整するオプティマイザ。
# m = 勾配の移動平均（momentum）、v = 勾配の二乗の移動平均。
# これにより、勾配が大きいパラメータは慎重に、小さいパラメータは大胆に更新。

beta1, beta2, eps_adam = 0.85, 0.99, 1e-8

# Adam のバッファ（1次モーメント、2次モーメント）
m = [0.0] * len(params)
v = [0.0] * len(params)

for step in range(num_steps):
    # --- トークン化 ---
    # 2.a 名前を1つ取り出し、BOS で挟む
    # 例: "さくら" → [BOS, 'さ'のID, 'く'のID, 'ら'のID, BOS]
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # --- 順伝播: 計算グラフを構築しながら損失を計算 ---
    # KV キャッシュを初期化（各ドキュメントごとにリセット）
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    # 2.b 1トークンずつ処理。スライド参照
    for pos_id in range(n):

        token_id = tokens[pos_id]       # 入力トークン
        target_id = tokens[pos_id + 1]  # 正解の次トークン

        logits = gpt(token_id, pos_id, keys, values)  # モデルの予測
        probs = softmax(logits)                         # 確率分布に変換
        loss_t = -probs[target_id].log()                # クロスエントロピー損失
        losses.append(loss_t)

    # 2.c その単語の平均損失
    loss = (1 / n) * sum(losses)

    # 2.d 逆伝播: 勾配を計算
    loss.backward()

    # 2.e Adam オプティマイザ: パラメータを更新 
    lr_t = learning_rate * (1 - step / num_steps)  # 線形学習率減衰
    for i, p in enumerate(params):
        # 1次モーメント（勾配の移動平均）
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        # 2次モーメント（勾配の二乗の移動平均）
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        # バイアス補正（初期ステップでの過小評価を修正）
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        # パラメータ更新: p -= lr * m_hat / (sqrt(v_hat) + eps)
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0  # 勾配をリセット（次のステップ用）

    # --- wandb ログ ---
    # loss（学習ロス）と perplexity（= exp(loss)、低いほど良い）を記録。
    # wandb ダッシュボードで学習曲線がリアルタイムに描画される。
    if HAS_WANDB:
        wandb.log({
            "train/loss": loss.data,
            "train/perplexity": math.exp(loss.data),
            "lr": lr_t,
        }, step=step)

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')


# =============================================================================
# 重みの保存
# =============================================================================
# 学習済みの重み（state_dict）を pickle で保存する。
# 推論スクリプト（microgpt_generate.py）でこの重みを読み込んで名前を生成できる。
# pickle: Python 標準のバイナリシリアライズ。パラメータ名付き dict をそのまま保存。

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)
WEIGHTS_FILE = os.path.join(
    WEIGHTS_DIR,
    f"L{n_layer}_E{n_embd}_H{n_head}_S{num_steps}.pkl"
)

save_data = {
    'uchars': uchars,
    'hyperparams': {
        'n_layer': n_layer,
        'n_embd': n_embd,
        'block_size': block_size,
        'n_head': n_head,
    },
    'state_dict': {
        key: [[p.data for p in row] for row in mat]
        for key, mat in state_dict.items()
    },
}

with open(WEIGHTS_FILE, 'wb') as f:
    pickle.dump(save_data, f)

print(f"\n重みを保存しました: {WEIGHTS_FILE}")
print("パラメータ名一覧:")
for key, mat in state_dict.items():
    print(f"  {key}: [{len(mat)}, {len(mat[0])}]")

# --- wandb 終了 ---
if HAS_WANDB:
    wandb.finish()
    print("wandb にログが記録されました。ダッシュボードを確認してください！")
