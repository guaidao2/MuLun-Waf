"""
train_waf.py — 使用 WAF 模拟器生成的轨迹数据训练 Mulun 模型的决策头。

核心策略:
  1. 冻结骨干（不需要训练 64M MiniMind，只训练 ~1M 的决策头）
  2. 使用 WAF 专用的动作/策略空间命名
  3. 输出可以直接部署到防火墙的轻量决策头权重

Usage:
    # 先生成数据
    python scripts/waf_simulator.py --episodes 500 --output ./data

    # 训练 WAF 决策头（冻结骨干）
    python trainers/train_waf.py --data-path ./data/waf_trajectories_500ep.jsonl

    # 训练 WAF 决策头（指定参数）
    python trainers/train_waf.py \
        --data-path ./data/waf_trajectories_500ep.jsonl \
        --epochs 20 \
        --batch-size 32 \
        --lr 1e-3 \
        --freeze-backbone \
        --save-dir ./out_waf
"""
import os, sys, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_here = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_here)
sys.path.insert(0, _ROOT)

from mulun_model import MulunConfig, MulunForCausalLM


# ─── WAF 动作空间（供 infer 时用，不参与训练）───────────────────
WAF_ACTION_NAMES = [
    'ALLOW',             # 0: 放行
    'LOG_ONLY',          # 1: 仅记录日志
    'RATE_LIMIT',        # 2: 限速
    'CHALLENGE',         # 3: CAPTCHA/JS挑战
    'BLOCK_IP',          # 4: 封禁源IP
    'BLOCK_SESSION',     # 5: 封禁会话
    'REDIRECT_HONEYPOT', # 6: 重定向到蜜罐
    'ESCALATE',          # 7: 升级人工审核
]

WAF_STRATEGY_NAMES = ['aggressive', 'balanced', 'defensive']


# ─── 数据集 ─────────────────────────────────────────────────────

class WAFSFTDataset(Dataset):
    """
    从 WAF 模拟器的 JSONL 输出加载训练数据。

    每行格式:
        {
            "conversations": [...],
            "structured": {"state": [...], "strategy": N, "action": N, "value": F},
            "metadata": {...}
        }
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        conversations = sample['conversations']
        structured = sample.get('structured', {})

        # ── 文本编码 ──
        prompt = self.tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
        encoding = self.tokenizer(
            prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        input_ids = encoding['input_ids']
        labels = input_ids.copy()

        # ── 决策标签 ──
        strategy = structured.get('strategy', -1)
        action = structured.get('action', -1)
        value = structured.get('value', -1.0)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'strategy': torch.tensor(strategy, dtype=torch.long),
            'action': torch.tensor(action, dtype=torch.long),
            'value': torch.tensor(value, dtype=torch.float),
        }


def _waf_collate(batch):
    """简单的 collate：直接 stack。"""
    result = {}
    for k in batch[0].keys():
        result[k] = torch.stack([b[k] for b in batch])
    return result


# ─── WAF 专用 Config ────────────────────────────────────────────

def make_waf_config(tokenizer, state_dim=16, n_strategies=3, n_actions=8,
                    hidden_size=768, num_hidden_layers=8, max_seq_len=512):
    """创建 WAF 场景的 MulunConfig。"""
    return MulunConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        vocab_size=len(tokenizer),
        state_dim=state_dim,
        n_strategies=n_strategies,
        n_actions=n_actions,
        max_position_embeddings=max_seq_len,
        use_moe=False,
    )


# ─── 训练 ───────────────────────────────────────────────────────

def train_waf(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Tokenizer ──
    from transformers import PreTrainedTokenizerFast
    tokenizer_path = args.tokenizer_dir or os.path.join(_ROOT, 'tokenizer')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f'Tokenizer: {len(tokenizer)} vocab')

    # ── Config ──
    config = make_waf_config(
        tokenizer=tokenizer,
        state_dim=args.state_dim,
        n_strategies=args.n_strategies,
        n_actions=args.n_actions,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
    )
    print(f'Config: state_dim={config.state_dim}, '
          f'strategies={config.n_strategies}, actions={config.n_actions}')

    # ── Model ──
    model = MulunForCausalLM(config, init_decision_head=True).to(device)
    total = sum(p.numel() for p in model.parameters())
    dh = sum(p.numel() for p in model.decision_head.parameters())
    print(f'Model total: {total/1e6:.1f}M parameters')
    print(f'  Decision head (trainable): {dh/1e3:.1f}K parameters')
    backbone_total = total - dh
    print(f'  Backbone (frozen): {backbone_total/1e6:.1f}M parameters')

    # 更新决策头的动作/策略名称（仅用于 interpretability）
    model.decision_head.action_names = WAF_ACTION_NAMES
    model.decision_head.strategy_names = WAF_STRATEGY_NAMES

    # ── 加载预训练骨干权重（可选） ──
    if args.from_weight and os.path.exists(args.from_weight):
        print(f'Loading backbone from {args.from_weight} ...')
        ckpt = torch.load(args.from_weight, map_location=device, weights_only=False)
        model_sd = model.state_dict()
        loaded = 0
        for k, v in ckpt.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v.to(dtype=model_sd[k].dtype, device=device)
                loaded += 1
        model.load_state_dict(model_sd, strict=False)
        print(f'  Loaded {loaded}/{len(ckpt)} keys')

    # ── 冻结骨干（默认） ──
    if args.freeze_backbone:
        for name, param in model.model.named_parameters():
            param.requires_grad = False
        print('  Backbone frozen (requires_grad=False)')
    else:
        print('  Backbone unfrozen (will be finetuned)')

    # ── Data ──
    dataset = WAFSFTDataset(args.data_path, tokenizer, args.max_seq_len)
    print(f'Dataset: {len(dataset)} samples')

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_waf_collate,
    )

    # ── Optimizer ──
    # 只训练决策头（骨干被冻结）
    if args.freeze_backbone:
        optimizer = optim.AdamW(
            model.decision_head.parameters(),
            lr=args.lr,
            weight_decay=0.01,
        )
    else:
        optimizer = optim.AdamW([
            {'params': model.model.parameters(), 'lr': args.lr * 0.1},
            {'params': model.decision_head.parameters(), 'lr': args.lr},
        ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    # ── Training ──
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_text_loss = 0.0
        total_decision_loss = 0.0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}', ncols=100)

        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            kwargs = {
                'step_strategies': batch['strategy'].unsqueeze(1).to(device),
                'step_actions': batch['action'].unsqueeze(1).to(device),
                'step_values': batch['value'].unsqueeze(1).to(device),
            }

            with torch.amp.autocast(device_type='cuda', enabled=(args.dtype != 'float32')):
                out = model(input_ids, labels=labels, mode='decision', **kwargs)
                loss = out.loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item() * args.accumulation_steps
            total_loss += loss_val
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})
            scheduler.step()

        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}')

        # Save checkpoint
        tag = f'_{args.run_name}' if args.run_name else ''
        if (epoch + 1) % args.save_every == 0:
            save_path = f'{args.save_dir}/waf{tag}_epoch{epoch+1}.pth'
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f'  Saved: {save_path}')

    # Final save
    final_path = f'{args.save_dir}/waf{tag}_final.pth'
    torch.save(model.state_dict(), final_path)
    config.save_pretrained(args.save_dir)

    # 同时保存纯决策头权重（供边缘部署用）
    dh_path = f'{args.save_dir}/waf{tag}_decision_head.pth'
    torch.save(model.decision_head.state_dict(), dh_path)
    print(f'\nFull model: {final_path}')
    print(f'Decision head only (for edge deploy): {dh_path}')
    print(f'  Size: ~{os.path.getsize(dh_path) / 1024:.0f} KB')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WAF 决策头训练')
    # 数据参数
    parser.add_argument('--data-path', type=str, required=True,
                        help='WAF 训练数据 (jsonl)')
    parser.add_argument('--tokenizer-dir', type=str, default=None,
                        help='Tokenizer path')
    parser.add_argument('--from-weight', type=str, default=None,
                        help='预训练骨干权重')
    parser.add_argument('--save-dir', type=str, default='../out_waf',
                        help='保存目录')

    # 模型架构参数
    parser.add_argument('--hidden-size', type=int, default=768,
                        help='骨干 hidden size')
    parser.add_argument('--num-hidden-layers', type=int, default=8,
                        help='骨干层数')
    parser.add_argument('--state-dim', type=int, default=16,
                        help='决策状态维度')
    parser.add_argument('--n-strategies', type=int, default=3,
                        help='策略数')
    parser.add_argument('--n-actions', type=int, default=8,
                        help='动作数')
    parser.add_argument('--max-seq-len', type=int, default=512,
                        help='最大序列长度')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮次')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='决策头学习率')
    parser.add_argument('--freeze-backbone', action='store_true', default=True,
                        help='冻结骨干（只训练决策头）')
    parser.add_argument('--accumulation-steps', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        help='训练精度 (float16/bfloat16/float32)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='训练设备')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers')
    parser.add_argument('--save-every', type=int, default=5,
                        help='每 N 轮保存一次')
    parser.add_argument('--run-name', type=str, default='waf16',
                        help='输出文件名标签')
    args = parser.parse_args()

    train_waf(args)
