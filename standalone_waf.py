"""
standalone_waf.py — 纯侧枝 WAF 决策模型（不含 LLM 骨干）。

架构:
  raw_state[16] → DomainEncoder → obs_emb[32]
  → StateEncoder → state[16] + uncertainty[16]
  → RNNDecisionStep(state, obs_emb, prev_action) → new_state
  ├→ ActionValueHead → strategy(3), action(8), value(1)
  └→ WorldModelStep → next_state[16], containment

总参数量: ~50K（可部署在防火墙 CPU）
训练方式: 纯 SFT，无 LLM，无 GPU 依赖
"""
import os, sys, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# ─── 常量 ──────────────────────────────────────────────────────────

WAF_ACTION_NAMES = [
    'ALLOW',    # 0: 放行
    'MONITOR',  # 1: 观察（原 LOG_ONLY + CHALLENGE + RATE_LIMIT）
    'BLOCK',    # 2: 封禁（原 BLOCK_IP + BLOCK_SESSION + HONEYPOT + ESCALATE）
]
WAF_STRATEGY_NAMES = ['aggressive', 'balanced', 'defensive']

# 8 动 → 3 动映射
ACTION_MAP_8TO3 = {0:0, 1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:2}

OBS_DIM = 32
STATE_DIM = 64
N_STRATEGIES = 3
N_ACTIONS = 3
MARKOV_RANK = 32  # 动作嵌入维度


# ═══════════════════════════════════════════════════════════════════
# 模型定义（纯侧枝，无 LLM）
# ═══════════════════════════════════════════════════════════════════

class DomainEncoder(nn.Module):
    """将原始 WAF 状态[16]映射到观测嵌入[obs_dim=32]。"""
    def __init__(self, in_dim=OBS_DIM, out_dim=OBS_DIM*2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class StateEncoder(nn.Module):
    """观测嵌入 → 结构化潜状态 + 不确定性。"""
    def __init__(self, obs_dim, state_dim=STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim * 2),  # mean + logvar
        )

    def forward(self, x):
        out = self.net(x)
        mean, logvar = out.chunk(2, dim=-1)
        state = torch.sigmoid(mean)
        uncertainty = torch.sigmoid(logvar)
        return state, uncertainty


class RNNDecisionStep(nn.Module):
    """
    GRU 风格决策步。
    输入: (prev_state[16], obs_emb[64], prev_action_emb[16])
    输出: (new_state[16], bias[64])
    """
    def __init__(self, state_dim=STATE_DIM, obs_dim=64, markov_rank=MARKOV_RANK):
        super().__init__()
        self.state_dim = state_dim

        self.joint_proj = nn.Linear(state_dim + markov_rank + obs_dim, 3 * state_dim)
        self.action_embed = nn.Linear(N_ACTIONS, markov_rank, bias=False)
        self.output_proj = nn.Linear(state_dim, obs_dim, bias=False)

    def forward(self, state, obs_emb, prev_action=None):
        B = state.shape[0]
        device = state.device

        if prev_action is None:
            prev_action = torch.zeros(B, dtype=torch.long, device=device)
        action_emb = self.action_embed(
            F.one_hot(prev_action, num_classes=N_ACTIONS).float()
        )

        z = torch.cat([state, action_emb, obs_emb], dim=-1)
        proj = self.joint_proj(z)
        gate_raw, candidate_raw, output_raw = proj.chunk(3, dim=-1)

        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        new_state = gate * state + (1.0 - gate) * candidate
        bias = self.output_proj(torch.tanh(output_raw))

        return new_state, bias

    def init_state(self, batch, device):
        return torch.zeros(batch, self.state_dim, device=device)


class ActionValueHead(nn.Module):
    """从状态输出分层决策。"""
    def __init__(self, state_dim=STATE_DIM):
        super().__init__()
        self.strategy_net = nn.Linear(state_dim, N_STRATEGIES)
        self.action_net = nn.Linear(state_dim, N_ACTIONS)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, 1),
        )

    def forward(self, state):
        return {
            'strategy_logits': self.strategy_net(state),
            'action_logits': self.action_net(state),
            'value': self.value_net(state),
        }


class WorldModelStep(nn.Module):
    """单步世界模型：(state, action) → next_state + containment。"""
    def __init__(self, state_dim=STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + N_ACTIONS, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim + 1),
        )

    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=-1)
        out = self.net(x)
        pred_state = torch.sigmoid(out[:, :-1])
        containment = torch.sigmoid(out[:, -1:])
        return pred_state, containment


class StandaloneWAFModel(nn.Module):
    """
    完全独立的 WAF 决策模型。

    总参数量 ~50K。不需要 LLM。不需要 GPU。
    输入: 32 维 WAF 请求特征向量
    输出: strategy, action, value, containment, predicted_state
    """
    def __init__(self):
        super().__init__()
        obs_dim = OBS_DIM * 2  # 64

        self.domain_encoder = DomainEncoder(OBS_DIM, obs_dim)
        self.state_encoder = StateEncoder(obs_dim, STATE_DIM)
        self.rnn_step = RNNDecisionStep(STATE_DIM, obs_dim, MARKOV_RANK)
        self.action_value = ActionValueHead(STATE_DIM)
        self.world_model = WorldModelStep(STATE_DIM)

        # 名称（仅用于可读性）
        self.action_names = WAF_ACTION_NAMES
        self.strategy_names = WAF_STRATEGY_NAMES

    def forward(self, features, rnn_state, prev_action=None):
        """
        Args:
            features: [B, 16] WAF 态势状态（来自模拟器的 WAFState）
            rnn_state: [B, 16] 上一步 RNN 状态
            prev_action: [B] 上一步动作索引（None 表示初始步）

        Returns:
            dict with all decision outputs + new_rnn_state
        """
        # 1. 特征编码
        obs_emb = self.domain_encoder(features)

        # 2. 状态压缩
        state, uncertainty = self.state_encoder(obs_emb)

        # 3. RNN 决策步
        new_state, bias = self.rnn_step(state, obs_emb, prev_action)

        # 4. 决策头
        decision = self.action_value(new_state)
        strategy_probs = F.softmax(decision['strategy_logits'], dim=-1)
        action_probs = F.softmax(decision['action_logits'], dim=-1)
        strategy_idx = strategy_probs.argmax(dim=-1)
        action_idx = action_probs.argmax(dim=-1)

        # 5. 世界模型
        action_onehot = F.one_hot(action_idx, num_classes=N_ACTIONS).float()
        pred_state, containment = self.world_model(new_state, action_onehot)

        return {
            'new_rnn_state': new_state,
            'state': state,
            'uncertainty': uncertainty,
            'strategy_probs': strategy_probs,
            'strategy_idx': strategy_idx,
            'action_probs': action_probs,
            'action_idx': action_idx,
            'value': decision['value'],
            'containment_prob': containment,
            'predicted_state': pred_state,
        }

    @torch.inference_mode()
    def decide(self, features, rnn_state=None, prev_action=None):
        """
        单步推理接口（部署用）。
        features: [32] 或 [B, 32]
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
            single = True
        else:
            single = False

        if rnn_state is None:
            rnn_state = self.rnn_step.init_state(features.shape[0], features.device)

        out = self.forward(features, rnn_state, prev_action)

        if single:
            return {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                    for k, v in out.items()}
        return out

    def init_state(self, batch, device):
        return self.rnn_step.init_state(batch, device)


# ═══════════════════════════════════════════════════════════════════
# 轨迹数据集（按 episode 分组重建 RNN 序列）
# ═══════════════════════════════════════════════════════════════════

class WAFTrajectoryDataset(Dataset):
    """
    将 WAF 模拟器的单步数据按 episode 分组为轨迹。

    每轮 (episode) 是一系列连续决策步：
      step_0: features[32], strategy, action, value, step=0
      step_1: features[32], strategy, action, value, step=1
      ...

    训练时 RNN 状态在 episode 内传递。
    """

    def __init__(self, data_path, max_steps=12):
        with open(data_path, encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]

        # 按 episode 分组，按 step 排序
        episodes = {}
        for s in lines:
            ep = s['metadata']['episode']
            step = s['metadata']['step']
            if ep not in episodes:
                episodes[ep] = []
            # 从 structured 和 metadata 中提取训练所需的字段
            # 优先使用 32 维 features，fallback 到 16 维 state
            feats = s['structured'].get('features', s['structured'].get('state', []))
            if len(feats) < OBS_DIM:
                feats = feats + [0.0] * (OBS_DIM - len(feats))
            feats = feats[:OBS_DIM]
            episodes[ep].append({
                'features': feats,
                'strategy': s['structured']['strategy'],
                'action': ACTION_MAP_8TO3.get(s['structured']['action'], 0),
                'value': s['structured']['value'],
                'step': step,
            })

        # 按 step 排序
        self.trajectories = []
        for ep_id in sorted(episodes.keys()):
            traj = sorted(episodes[ep_id], key=lambda x: x['step'])
            # 只保留 max_steps 步
            self.trajectories.append(traj[:max_steps])

        print(f'  {len(lines)} 条样本 → {len(self.trajectories)} 条轨迹')

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        features = torch.tensor([s['features'] for s in traj], dtype=torch.float)
        strategies = torch.tensor([s['strategy'] for s in traj], dtype=torch.long)
        actions = torch.tensor([s['action'] for s in traj], dtype=torch.long)
        values = torch.tensor([s['value'] for s in traj], dtype=torch.float)

        return {
            'features': features,          # [T, 16]
            'strategies': strategies,       # [T]
            'actions': actions,             # [T]
            'values': values,               # [T]
            'traj_len': torch.tensor(len(traj), dtype=torch.long),
        }


def _traj_collate(batch):
    """Collate: padding 到 batch 内最大步数。"""
    keys = batch[0].keys()
    result = {}

    max_len = max(b['traj_len'] for b in batch)
    feat_dim = batch[0]['features'].shape[-1]

    feats = []
    strategies = []
    actions = []
    values = []
    lengths = []

    for b in batch:
        T = b['traj_len'].item()
        lengths.append(T)
        # padding
        pad_t = max_len - T
        feats.append(F.pad(b['features'], (0, 0, 0, pad_t)))
        strategies.append(F.pad(b['strategies'], (0, pad_t), value=-1))
        actions.append(F.pad(b['actions'], (0, pad_t), value=-1))
        values.append(F.pad(b['values'], (0, pad_t), value=-1.0))

    result['features'] = torch.stack(feats)             # [B, max_T, 32]
    result['strategies'] = torch.stack(strategies)      # [B, max_T]
    result['actions'] = torch.stack(actions)
    result['values'] = torch.stack(values)
    result['traj_len'] = torch.tensor(lengths)           # [B]

    return result


# ═══════════════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════════════

def train_standalone_waf(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── 数据 ──
    print(f'Loading data from {args.data_path}...')
    dataset = WAFTrajectoryDataset(args.data_path)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=_traj_collate, num_workers=args.num_workers,
    )

    # ── 模型 ──
    model = StandaloneWAFModel().to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f'Model: {total:,} parameters ({total/1024:.1f}K)')

    # ── 优化器 ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    # ── 训练循环 ──
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}', ncols=100)

        for batch in pbar:
            feats = batch['features'].to(device)        # [B, T, 32]
            strategies = batch['strategies'].to(device)  # [B, T]
            actions = batch['actions'].to(device)        # [B, T]
            values = batch['values'].to(device)          # [B, T]
            traj_len = batch['traj_len']                 # [B]

            B, T, _ = feats.shape

            # ── 沿时间步展开 RNN ──
            rnn_state = model.init_state(B, device)
            prev_action = None

            total_step_loss = 0.0
            n_valid_steps = 0

            for t in range(T):
                # 找出该时间步有效的样本
                valid = (traj_len > t)  # [B]

                out = model.forward(feats[:, t, :], rnn_state, prev_action)

                # ── 计算损失：只在有效步上 ──
                if valid.any():
                    # 策略 CE
                    strat_loss = F.cross_entropy(
                        out['strategy_probs'][valid],
                        strategies[valid, t],
                    )
                    # 动作 CE
                    act_loss = F.cross_entropy(
                        out['action_probs'][valid],
                        actions[valid, t],
                    )
                    # 价值 MSE
                    val_pred = torch.sigmoid(out['value'][valid].squeeze(-1))
                    val_loss = F.mse_loss(val_pred, values[valid, t])

                    step_loss = strat_loss + act_loss + val_loss
                    total_step_loss += step_loss
                    n_valid_steps += 1

                # 更新 RNN 状态（所有样本）
                rnn_state = out['new_rnn_state']
                prev_action = actions[:, t]  # 用真实动作而非预测动作（teacher forcing）

            # 平均损失
            avg_loss = total_step_loss / max(n_valid_steps, 1)
            (avg_loss / args.accumulation_steps).backward()

            if (pbar.n + 1) % args.accumulation_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += avg_loss.item()
            pbar.set_postfix({'loss': f'{avg_loss.item():.4f}'})
            scheduler.step()

        avg_epoch_loss = total_loss / len(loader)
        print(f'  Epoch {epoch+1} avg_loss={avg_epoch_loss:.4f}')

        # 保存
        if (epoch + 1) % args.save_every == 0:
            _save(model, args.save_dir, epoch + 1)

    _save(model, args.save_dir, 'final')
    print(f'\nDone. Model saved to {args.save_dir}/')
    return model


def _save(model, save_dir, tag):
    os.makedirs(save_dir, exist_ok=True)
    path = f'{save_dir}/standalone_waf_{tag}.pth'
    torch.save(model.state_dict(), path)
    size_kb = os.path.getsize(path) / 1024
    print(f'  Saved: {path} ({size_kb:.1f} KB)')


# ═══════════════════════════════════════════════════════════════════
# 推理示例
# ═══════════════════════════════════════════════════════════════════

@torch.inference_mode()
def infer_one_request(model, features_32d, rnn_state=None, prev_action=None):
    """对单个请求做 WAF 决策。"""
    model.eval()
    feat = torch.tensor(features_32d, dtype=torch.float).unsqueeze(0)
    out = model.decide(feat, rnn_state, prev_action)

    action_name = model.action_names[out['action_idx'].item()]
    strategy_name = model.strategy_names[out['strategy_idx'].item()]

    return {
        'action': action_name,
        'strategy': strategy_name,
        'value': out['value'].item(),
        'containment': out['containment_prob'].item(),
        'uncertainty': out['uncertainty'].tolist(),
    }, out['new_rnn_state'], out['action_idx'].item()


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='训练/推理纯侧枝 WAF 模型')
    parser.add_argument('--data-path', type=str, default=None,
                        help='训练数据路径 (jsonl)')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'infer'],
                        help='train 或 infer')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save-dir', type=str, default='./out_standalone_waf')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--accumulation-steps', type=int, default=2)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--load-path', type=str, default=None,
                        help='推理时加载的模型权重')
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data_path:
            print('请指定 --data-path')
            sys.exit(1)
        train_standalone_waf(args)
    else:
        # 推理模式：交互式测试
        if not args.load_path:
            args.load_path = f'{args.save_dir}/standalone_waf_final.pth'
        model = StandaloneWAFModel()
        model.load_state_dict(torch.load(args.load_path, map_location='cpu',
                                          weights_only=True))
        model.eval()
        print(f'Loaded model from {args.load_path}')
        print(f'总参数: {sum(p.numel() for p in model.parameters()):,} ({sum(p.numel() for p in model.parameters())/1024:.1f}K)')

        # 模拟连续决策以演示 RNN 状态跟踪
        rnn_state = None
        prev_action = None
        step = 0

        print('\n输入 "q" 退出，输入特征值（32个逗号分隔的浮点数）继续')
        print('或输入 "auto N" 从数据文件取第 N 条样本测试\n')

        while True:
            cmd = input(f'\n[Step {step}] 输入: ').strip()
            if cmd.lower() == 'q':
                break

            try:
                if cmd.startswith('auto'):
                    # 从数据文件自动取样本
                    n = int(cmd.split()[1]) if len(cmd.split()) > 1 else 0
                    if not hasattr(infer_one_request, '_test_data'):
                        with open(args.data_path, encoding='utf-8') as f:
                            infer_one_request._test_data = [json.loads(l) for l in f]
                    sample = infer_one_request._test_data[n]
                    feats = sample['structured']['state'][:16]
                    print(f'  从样本 #{n} 取状态: 攻击={sample["metadata"]["is_attack"]}, '
                          f'类型={sample["metadata"]["attack_type"]}')
                else:
                    feats = [float(x) for x in cmd.split(',')[:16]]
                    if len(feats) < 16:
                        feats = feats + [0.0] * (16 - len(feats))

                result, rnn_state, prev_action = infer_one_request(
                    model, feats, rnn_state, prev_action
                )

                print(f'  → 策略: {result["strategy"]:>10s}'
                      f'  动作: {result["action"]:<16s}'
                      f'  置信度: {result["value"]:.3f}'
                      f'  遏制: {result["containment"]:.3f}'
                      f'  不确定度: {np.mean(result["uncertainty"]):.3f}')
                step += 1

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f'  错误: {e}')
