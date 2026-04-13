# Bonsai Emoji Retriever

Slack 投稿テキストに最適なリアクション絵文字を推薦する text-text dual-encoder。

## アーキテクチャ

- **ベースモデル**: [prism-ml/Bonsai-8B-unpacked](https://huggingface.co/prism-ml/Bonsai-8B-unpacked) (Qwen3-8B ベース, 8.19B params, 1-bit で 1.15GB デプロイ可)
- **方式**: Shared encoder + LoRA fine-tuning + last-token pooling + L2 normalize
- **損失関数**: Extended InfoNCE (Qwen3 Embedding 論文, arXiv 2506.05176v3)
  - In-batch query-query, doc-doc, cross query-doc negatives
  - False negative mitigation mask (margin=0.1)
  - Hard negatives サポート
  - Learnable temperature
- **MRL**: Matryoshka Representation Learning — 埋め込みを [256, 1024, 4096] 次元で学習
- **QLoRA**: 4-bit NF4 量子化オプション (bitsandbytes)

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 学習

```bash
python scripts/train.py --config configs/default.yaml
```

## 埋め込み確認

```bash
python scripts/show_embeddings.py --config configs/default.yaml
```

## 設定

`configs/default.yaml` で全 hyperparameter を制御:

- `model.model_id`: ベースモデル
- `lora.*`: LoRA rank, alpha, dropout, target modules
- `quantization.enabled`: QLoRA ON/OFF
- `loss.fn_margin`: false negative mask margin
- `mrl.enabled` / `mrl.dims`: MRL ON/OFF と対象次元
- `training.*`: epochs, batch_size, lr, etc.

## 参照

- [Qwen3 Embedding 論文](https://arxiv.org/abs/2506.05176) — Extended InfoNCE, MRL, false negative mask
- [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec) — LoRA / temperature / GradCache 設計参考
- [Bonsai-8B](https://huggingface.co/prism-ml/Bonsai-8B-unpacked) — 1-bit 量子化 LLM

## ライセンス

Apache-2.0
