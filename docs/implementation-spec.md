# Bonsai Emoji Retriever — 実装仕様 v0.2

## 0. スコープ

**本仕様のゴール**: Qwen3 Embedding 論文 (arXiv 2506.05176v3) のテクニックを取り入れた
学習パイプラインが、ダミーデータでエラーなく 1 epoch 完走すること。

### フェーズ

| Phase | 内容 | 環境 |
|-------|------|------|
| **Phase 1a — POC 基本** | 基本学習コードが 1 epoch 回る | ホスト直 (venv) — **完了** |
| **Phase 1b — Qwen3 Embedding テクニック** | Extended InfoNCE, FN mask, MRL, QLoRA, hard negatives | ホスト直 (venv) |
| **Phase 2 — Container** | Phase 1 を Docker イメージに固める | `docker run` → コンテナ内で学習 |
| Phase 3 (future) | 実データ投入・評価・推論 API・Slack bot | — |

### v0.1 → v0.2 の変更点

- **Extended InfoNCE**: q-q, d-d, cross q-d negatives を denominator に追加 (論文 §3.2)
- **False negative mitigation mask**: margin=0.1 で false negative をマスク (論文 §3.2)
- **Hard negatives**: データ・loss で K hard negatives をサポート (論文 §3.2)
- **Matryoshka Representation Learning (MRL)**: 複数次元で loss 合算 (論文 §3.1)
- **QLoRA**: 4-bit NF4 量子化オプション追加
- **Bonsai 補足**: 8.19B params (Qwen3-8B ベース)、1-bit 量子化で 1.15GB デプロイ可能

---

## 1. アーキテクチャ

```
┌─────────────────────────────────────────────┐
│            Shared Bonsai-8B Encoder          │
│         (prism-ml/Bonsai-8B-unpacked)        │
│        + LoRA adapters (+ optional QLoRA)    │
└──────────────┬──────────────┬───────────────┘
               │              │
        query input      label input (+ hard negatives)
               │              │
     ┌─────────▼──────┐ ┌────▼─────────────┐
     │ Instruct: ...   │ │ {label_text}     │
     │ Query: {msg}    │ │ <|endoftext|>    │
     │ <|endoftext|>   │ │                  │
     └─────────┬──────┘ └────┬─────────────┘
               │              │
        last token hidden     last token hidden
               │              │
           L2 norm         L2 norm
               │              │
     ┌─────────▼──────────────▼───────────────┐
     │  MRL: loss at dims [256, 1024, 4096]   │
     │  Extended InfoNCE + FN mask per dim     │
     └───────────────────┬───────────────────-─┘
                         │
                    total loss
```

---

## 2. モデル・LoRA 設定

### ベースモデル

```
model_id = "prism-ml/Bonsai-8B-unpacked"
```

Bonsai-8B は Qwen3-8B ベースの 8.19B params モデル。
hidden_size=4096, num_layers=36, GQA (32 query / 8 KV heads)。
1-bit 量子化 (Q1_0) で 1.15GB にデプロイ可能だが、
LoRA 学習には unpacked FP16 版 (16.4GB) を使う。

### LoRA 設定

```python
lora_config = {
    "r": 16,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
```

### QLoRA 設定 (optional)

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

VRAM 節約が必要な場合に有効化。FP16 の ~16GB → 4-bit の ~5GB に削減。
aarch64 (GB10 統合メモリ 120GB) では FP16 のままでも十分だが、
他環境 (8GB GPU 等) への移植性のためにサポートする。

### Temperature

```python
temperature = nn.Parameter(torch.tensor(0.05))  # learnable, clamped to [0.01, 1.0]
```

---

## 3. 入力テンプレート

### Query (投稿テキスト側)

```
Instruct: 与えられたSlack投稿に対して最も適切なリアクション名テキストを検索せよ
Query: {normalized_message}<|endoftext|>
```

### Label (スタンプ側)

```
{label_text}<|endoftext|>
```

Chat template は使わない。raw text + 終端トークンのみ。

### Tokenization ルール

- `<|endoftext|>` は tokenizer の既存 special token を使う
- padding は左詰め (`padding_side="left"`)
- `attention_mask` で pad token を pooling から除外

---

## 4. データスキーマ

### スタンプ辞書 (`emoji_catalog.json`)

```json
[
  {
    "emoji_id": "eyes",
    "surface_ja": "両目",
    "surface_en": "eyes",
    "aliases": ["見てる", "watching"],
    "glosses": ["見ました", "確認中", "監視中", "注目"],
    "label_text": "両目 / eyes / 見ました / 確認中 / 監視中 / 注目"
  }
]
```

### 学習データ (`train.jsonl`)

```json
{
  "query": "確認しました！",
  "positive_label": "eyes",
  "negative_labels": ["thumbsup", "pray"]
}
```

- `negative_labels`: hard negatives の emoji_id リスト (空リスト可)

### POC 用ダミーデータ (3 ペア + hard negatives)

```json
{"query": "確認しました！", "positive_label": "eyes", "negative_labels": ["white_check_mark"]}
{"query": "ありがとうございます！助かりました", "positive_label": "pray", "negative_labels": ["white_check_mark"]}
{"query": "対応完了です", "positive_label": "white_check_mark", "negative_labels": ["eyes"]}
```

---

## 5. Extended InfoNCE Loss (Qwen3 Embedding 論文 §3.2)

### 標準 InfoNCE との差分

Qwen3 Embedding では denominator を拡張し、batch 内の **全ペア** を負例候補にする:

```
Z_i = exp(s(q_i, d_i+) / τ)
    + Σ_k  m_ik * exp(s(q_i, d_{i,k}^-) / τ)    # K hard negatives
    + Σ_{j≠i} m_ij * exp(s(q_i, q_j) / τ)        # in-batch query-query
    + Σ_{j≠i} m_ij * exp(s(d_i+, d_j+) / τ)      # in-batch doc-doc
    + Σ_{j≠i} m_ij * exp(s(q_i, d_j+) / τ)       # in-batch cross q-d
```

### False Negative Mitigation Mask

```
m_ij = 0  if s_ij > s(q_i, d_i+) + margin  OR  d_j == d_i+
m_ij = 1  otherwise
```

- `margin = 0.1` (論文の値)
- 正例より類似度が高い in-batch サンプルを false negative と判断しマスク

### 実装

```python
class ExtendedInfoNCELoss(nn.Module):
    def __init__(self, init_temperature=0.05, fn_margin=0.1):
        ...

    def forward(self, query_embs, pos_label_embs, hard_neg_embs=None):
        # query_embs: (B, D)
        # pos_label_embs: (B, D)
        # hard_neg_embs: (B, K, D) or None
        #
        # 1. Compute all similarity pairs
        # 2. Build false negative mask
        # 3. Combine all negatives in denominator
        # 4. Return -log(positive / denominator)
```

---

## 6. Matryoshka Representation Learning (MRL)

### 概要

Qwen3 Embedding は MRL をサポートし、埋め込みを小さい次元に truncate しても
品質が保たれる。学習時に複数次元で loss を計算し合算する。

### 実装

```python
mrl_dims = [256, 1024, 4096]  # full hidden_size=4096

total_loss = 0
for dim in mrl_dims:
    q_trunc = F.normalize(query_embs[:, :dim], p=2, dim=-1)
    l_trunc = F.normalize(pos_embs[:, :dim], p=2, dim=-1)
    # hard_neg も同様に truncate
    total_loss += loss_fn(q_trunc, l_trunc, hard_neg_trunc)
total_loss /= len(mrl_dims)
```

- 各次元で再正規化してから loss を計算
- 最終 loss は各次元の平均

---

## 7. QLoRA 設定

### config 追加

```yaml
quantization:
  enabled: false
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true
```

`enabled: true` にすると BitsAndBytesConfig を適用。
bitsandbytes が aarch64 未対応の場合は graceful fallback で FP16 にする。

---

## 8. モジュール構成

```
bonsai-emoji-retriever/
├── README.md
├── .gitignore
├── Dockerfile
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── emoji_catalog.json
│   └── train.jsonl
├── src/
│   ├── __init__.py
│   ├── model.py            # BonsaiEmbedder + MRL support
│   ├── loss.py             # ExtendedInfoNCELoss + FN mask
│   ├── dataset.py          # Dataset + Collator (hard negatives)
│   ├── trainer.py          # 学習ループ (MRL integration)
│   ├── preprocess.py       # テキスト正規化
│   └── config.py           # yaml → dataclass
├── scripts/
│   ├── train.py            # CLI エントリポイント
│   └── show_embeddings.py  # デバッグ: 埋め込み確認
└── docs/
    └── implementation-spec.md
```

---

## 9. 完了条件

### Phase 1a — POC 基本 ✅ 完了

- [x] `python scripts/train.py` が 1 epoch 完走
- [x] loss が数値として出力される
- [x] LoRA adapter が `outputs/` に保存される
- [x] `show_embeddings.py` で sim matrix が表示される

### Phase 1b — Qwen3 Embedding テクニック ✅ 完了

- [x] Extended InfoNCE (q-q, d-d, cross negatives) が動作する
- [x] False negative mask が適用される (margin=0.1)
- [x] Hard negatives がデータから loss に流れる
- [x] MRL が複数次元で loss を計算する
- [x] QLoRA オプションがコード上存在する (aarch64 未対応時は graceful skip)
- [x] 全統合で 1 epoch 完走、loss が NaN/Inf でない (step1=1.0027, step2=0.2347)
- [x] GitHub リポジトリに push 済み (yukimaru77/bonsai-emoji-retriever)

### Phase 2 — Container

- [ ] `docker build` が成功する
- [ ] `docker run --gpus all -it ...` でコンテナに入れる
- [ ] コンテナ内で Phase 1 と同じ学習が完走する

---

## 10. Future Work (本仕様のスコープ外)

- SLERP model merging (複数チェックポイントの球面線形補間) — 論文で +1.8pt
- 実 Slack データ収集パイプライン
- 大規模学習データ作成 (downsample、時系列分割)
- GradCache (VRAM 不足時)
- 評価パイプライン (top-1 acc, MRR, abstain precision)
- 推論 API (label 事前埋め込み + cosine 検索 + abstain ロジック)
- Slack bot 統合
- LLM による label augmentation / synthetic data
- NV-Embed 方式 (latent attention pooling) の比較実験
- LLM2Vec 方式 (bidirectional + mean pooling) の比較実験
- `gate_proj, up_proj, down_proj` への LoRA 拡張
- Chat template 経由の入力方式との比較
- 1-bit GGUF での推論パイプライン
