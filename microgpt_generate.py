"""
microgpt_generate.py - 学習済み MicroGPT で日本語名前を生成する推論スクリプト

使い方:
  uv run microgpt_generate.py                    # 最新の重みでランダムに 20 個生成
  uv run microgpt_generate.py --start さ         # 「さ」から始まる名前を生成
  uv run microgpt_generate.py --start ゆ -n 10   # 「ゆ」から始まる名前を 10 個
  uv run microgpt_generate.py --weights weights/L1_E48_H4_S1000.pkl  # 特定の重みを指定
  uv run microgpt_generate.py --temperature 0.8  # より多様な生成
"""

import os
import sys
import glob
import pickle
import math
import random
import argparse
random.seed()


# =============================================================================
# コマンドライン引数
# =============================================================================

parser = argparse.ArgumentParser(description="MicroGPT 推論スクリプト（名前生成）")
parser.add_argument('--start', type=str, default=None, help='最初の一文字（ひらがな）')
parser.add_argument('-n', '--num', type=int, default=20, help='生成する名前の数（デフォルト: 20）')
parser.add_argument('--temperature', type=float, default=0.5, help='生成の多様性（デフォルト: 0.5）')
parser.add_argument('--weights', type=str, default=None, help='重みファイルのパス（デフォルト: 最新の重み）')
args = parser.parse_args()


# =============================================================================
# 重みの読み込み
# =============================================================================

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')

if args.weights is not None:
    weights_path = args.weights
else:
    # weights/ フォルダから最新（更新日時が最も新しい）の .pkl を選択
    pkl_files = glob.glob(os.path.join(WEIGHTS_DIR, '*.pkl'))
    if not pkl_files:
        print("エラー: weights/ フォルダに .pkl ファイルがありません。")
        print("先に uv run microgpt_train.py で学習してください。")
        sys.exit(1)
    weights_path = max(pkl_files, key=os.path.getmtime)

if not os.path.exists(weights_path):
    print(f"エラー: {weights_path} が見つかりません。")
    sys.exit(1)

with open(weights_path, 'rb') as f:
    save_data = pickle.load(f)

uchars = save_data['uchars']
hp = save_data['hyperparams']
n_layer = hp['n_layer']
n_embd = hp['n_embd']
block_size = hp['block_size']
n_head = hp['n_head']
head_dim = n_embd // n_head

BOS = len(uchars)
vocab_size = len(uchars) + 1

state_dict = save_data['state_dict']

print(f"重み: {os.path.basename(weights_path)} (vocab={vocab_size}, embd={n_embd}, head={n_head})")


# =============================================================================
# 3. モデルアーキテクチャ（推論専用・float のみ）
# =============================================================================

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def relu(x):
    return max(0.0, x)


def gpt(token_id, pos_id, keys, values):
    """GPT の順伝播（推論専用・float 版）"""
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]

    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]

            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [relu(xi) for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits


# =============================================================================
# 4. 名前生成
# =============================================================================

def generate_name(start_char=None, temperature=0.5):
    """
    名前を1つ生成する。
    start_char: 最初の一文字（ひらがな）。None ならモデルに自由に生成させる。
    """
    keys_cache = [[] for _ in range(n_layer)]
    values_cache = [[] for _ in range(n_layer)]

    sample = []
    pos_id = 0

    # 3.a 生成開始
    if start_char is not None:
        # BOS → start_char の順に入力し、その続きを生成
        gpt(BOS, pos_id, keys_cache, values_cache)
        pos_id += 1

        char_id = uchars.index(start_char)
        sample.append(start_char)

        # 3.b start_char を入力して次のトークンの logits を得る
        logits = gpt(char_id, pos_id, keys_cache, values_cache)
        pos_id += 1
    else:
        # BOS から自由に生成
        # BOSを入力して次のトークンの logits を得る
        logits = gpt(BOS, pos_id, keys_cache, values_cache)
        pos_id += 1

    # 続きを自己回帰的に生成
    while pos_id < block_size:
        # 3.c logits を温度スケーリングして確率に変換
        probs = softmax([l / temperature for l in logits])

        # 3.d 確率に従って次のトークンをサンプリング
        token_id = random.choices(range(vocab_size), weights=probs)[0]

        # 3.e サンプリングしたトークンが BOS なら終了。そうでなければ文字を追加して続行。
        if token_id == BOS: 
            break
        sample.append(uchars[token_id])

        # 3.b サンプリングしたトークンを入力して次のトークンの logits を得る
        logits = gpt(token_id, pos_id, keys_cache, values_cache)
        pos_id += 1

    return ''.join(sample)


# =============================================================================
# メイン
# =============================================================================

if __name__ == '__main__':
    if args.start is not None:
        if args.start not in uchars:
            print(f"エラー: '{args.start}' は学習データに含まれていない文字です。")
            print(f"使える文字: {''.join(uchars)}")
            sys.exit(1)
        print(f"--- 「{args.start}」から始まる名前を {args.num} 個生成 ---")
    else:
        print(f"--- ランダムに名前を {args.num} 個生成 ---")

    for i in range(args.num):
        name = generate_name(start_char=args.start, temperature=args.temperature)
        print(f"  {i+1:2d}: {name}")
