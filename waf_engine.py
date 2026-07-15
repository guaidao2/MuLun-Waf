"""
waf_engine.py — 基于 GameNN 侧枝模型的轻量 WAF 引擎。

设计目标:
  - 纯 CPU 运行，单次决策 ~10μs
  - 每 IP 独立 RNN 状态跟踪
  - IP 白名单/黑名单支持
  - 置信度门控降级（不确定时放行+记录）
  - 可对接真实 HTTP 请求或模拟器数据

用法:
    from waf_engine import WAFEngine

    engine = WAFEngine(model_path='./out_waf_3act/standalone_waf_final.pth')

    # IP 管理
    engine.allowlist_add('192.168.1.0/24')
    engine.blocklist_add('10.0.0.5')

    # 单请求决策
    result = engine.decide(
        source_ip='203.0.113.5',
        features=[0.1]*32,  # 32 维特征向量
    )
    print(result['action'])       # 'ALLOW' | 'MONITOR' | 'BLOCK'
    print(result['confidence'])   # 0.0 ~ 1.0
    print(result['uncertainty'])  # 模型不确定性
"""
import os, sys, json, ipaddress, time, threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import numpy as np
import torch

# 确保能找到 standalone_waf.py
_here = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_here)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from standalone_waf import StandaloneWAFModel, ACTION_MAP_8TO3


# ─── 常量 ──────────────────────────────────────────────────────────

ACTION_NAMES = ['ALLOW', 'MONITOR', 'BLOCK']
ACTION_LEVEL = {'ALLOW': 0, 'MONITOR': 1, 'BLOCK': 2}

# 置信度阈值
CONF_THRESHOLDS = {
    'ALLOW':   0.55,   # 确信 ≥0.55 才 ALLOW，否则升 MONITOR
    'MONITOR': 0.45,   # 确信 ≥0.45 才 MONITOR
    'BLOCK':   0.50,   # 确信 ≥0.50 才 BLOCK，否则降 ALLOW+记录
}

UNCERTAINTY_THRESHOLD = 0.65  # 模型不确定性高于此值 → 降级为 ALLOW+记录


# ─── 特征提取 ──────────────────────────────────────────────────────

def extract_features_from_request(
    method: str,
    path: str,
    query_params: Dict[str, str],
    body: str = '',
    headers: Dict[str, str] = None,
    source_ip: str = '',
) -> List[float]:
    """
    从 HTTP 请求中提取 32 维特征向量。

    维度说明:
      [0-7]:   攻击类型 one-hot (由检测模块填充)
      [8-15]:  请求形态特征
      [16-23]: 时序特征 (由引擎维护)
      [24-31]: 会话特征 (由引擎维护)

    返回 32 维 float 列表。
    """
    headers = headers or {}
    feats = [0.0] * 32

    # ── 请求形态特征 [8-15] ──
    # 8: 请求长度归一化
    total_len = len(method) + len(path) + len(body) + sum(len(v) for v in query_params.values())
    feats[8] = min(1.0, total_len / 5000.0)

    # 9: 特殊字符比率
    all_text = path + ' ' + ' '.join(query_params.values()) + ' ' + body
    special = sum(c in "'\"%;<>()|&`$\\/\\-\\'" for c in all_text)
    feats[9] = min(1.0, special / max(len(all_text), 1) * 5)

    # 10: 路径深度
    depth = len([p for p in path.split('/') if p])
    feats[10] = min(1.0, depth / 15.0)

    # 11: 参数数量
    n_params = len(query_params)
    feats[11] = min(1.0, n_params / 20.0)

    # 12: 方法编码
    method_map = {'GET': 0.0, 'POST': 0.33, 'PUT': 0.5, 'DELETE': 0.67, 'PATCH': 0.5, 'OPTIONS': 0.17}
    feats[12] = method_map.get(method.upper(), 0.25)

    # 13-15: 头部异常特征
    ua = headers.get('user-agent', headers.get('User-Agent', ''))
    feats[13] = 0.5 if not ua else 0.0  # 缺少 UA
    has_cookie = 'cookie' in {k.lower() for k in headers.keys()}
    feats[14] = 0.0 if has_cookie else 0.8  # 正常请求通常有 cookie
    has_referer = 'referer' in {k.lower() for k in headers.keys()}
    feats[15] = 0.0 if has_referer else 0.3

    return feats


# ─── 特征增强（注入检测模块的信号）───────────────────────────────────

def inject_detection_signals(feats: List[float], detections: Dict[str, float] = None):
    """
    将外部检测引擎（如 SQLi 检测、XSS 检测）的打分注入到特征向量中。

    detections: {'sqli': 0.8, 'xss': 0.0, 'path_traversal': 0.0, ...}

    特征 [0-7] 对应 ATTACK_TYPES 的 one-hot 编码。
    """
    if not detections:
        return feats

    attack_types = ['SQLi', 'XSS', 'PathTraversal', 'CmdInjection',
                    'SSRF', 'FileUpload', 'BruteForce', 'ScannerBot']

    for i, atk in enumerate(attack_types):
        score = detections.get(atk.lower(), 0.0)
        feats[i] = max(feats[i], score)

    return feats


# ─── IP 工具 ────────────────────────────────────────────────────────

def ip_in_network(ip: str, network: str) -> bool:
    """检查 IP 是否在 CIDR 范围内。"""
    try:
        return ipaddress.ip_address(ip) in ipaddress.ip_network(network, strict=False)
    except:
        return False


# ─── 引擎主类 ──────────────────────────────────────────────────────

@dataclass
class RequestContext:
    """每个 IP 的请求上下文（由引擎自动维护）。"""
    ip: str
    rnn_state: Optional[torch.Tensor] = None
    prev_action: Optional[torch.Tensor] = None
    request_count: int = 0
    attack_count: int = 0
    last_seen: float = 0.0
    blocked_until: float = 0.0  # 封禁到期时间戳


class WAFEngine:
    """
    轻量 WAF 决策引擎。

    核心能力:
      1. 每 IP 独立 RNN 状态跟踪（跨请求记忆）
      2. IP 白名单/黑名单
      3. 置信度门控自动降级
      4. 临时封禁（block IP for N seconds）
      5. 外部检测引擎信号注入
    """

    def __init__(
        self,
        model_path: str = './out_waf_3act/standalone_waf_final.pth',
        device: str = 'cpu',
        uncertainty_threshold: float = UNCERTAINTY_THRESHOLD,
        # RNN 上下文超时（秒），超过此时间未活动的 IP 重置状态
        context_timeout: float = 300.0,
        # 临时封禁默认时长
        block_duration: float = 600.0,
    ):
        self.device = torch.device(device)
        self.uncertainty_threshold = uncertainty_threshold
        self.context_timeout = context_timeout
        self.default_block_duration = block_duration

        # ── 加载模型 ──
        self.model = StandaloneWAFModel()
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f'[WAF] 模型加载完成: {sum(p.numel() for p in self.model.parameters()):,} 参数')

        # ── IP 列表 ──
        self.allowlist: List[str] = []     # CIDR 列表
        self.blocklist: List[str] = []     # CIDR 列表

        # ── 每 IP 上下文 ──
        self._contexts: Dict[str, RequestContext] = OrderedDict()
        self._lock = threading.Lock()

        # ── 统计 ──
        self.stats = {
            'total_requests': 0,
            'allowed': 0,
            'monitored': 0,
            'blocked': 0,
            'allowlist_hits': 0,
            'blocklist_hits': 0,
            'confidence_downgrades': 0,
        }

    # ── IP 列表管理 ──────────────────────────────────────────────

    def allowlist_add(self, cidr: str):
        """添加白名单 CIDR。白名单 IP 始终 ALLOW 且不记录 RNN。"""
        try:
            ipaddress.ip_network(cidr, strict=False)
            self.allowlist.append(cidr)
            print(f'[WAF] 白名单添加: {cidr}')
        except:
            print(f'[WAF] 无效 CIDR: {cidr}')

    def allowlist_remove(self, cidr: str):
        """移除白名单 CIDR。"""
        self.allowlist = [c for c in self.allowlist if c != cidr]

    def blocklist_add(self, cidr: str):
        """添加黑名单 CIDR。黑名单 IP 直接 BLOCK。"""
        try:
            ipaddress.ip_network(cidr, strict=False)
            self.blocklist.append(cidr)
            print(f'[WAF] 黑名单添加: {cidr}')
        except:
            print(f'[WAF] 无效 CIDR: {cidr}')

    def blocklist_remove(self, cidr: str):
        """移除黑名单 CIDR。"""
        self.blocklist = [c for c in self.blocklist if c != cidr]

    def temp_block(self, ip: str, duration: float = None):
        """临时封禁一个 IP（指定秒数）。到期自动解除。"""
        dur = duration or self.default_block_duration
        ctx = self._get_context(ip)
        ctx.blocked_until = time.time() + dur
        print(f'[WAF] 临时封禁 {ip}  {dur:.0f}s')

    # ── 核心决策 ─────────────────────────────────────────────────

    def decide(
        self,
        source_ip: str,
        features: List[float],
        detections: Dict[str, float] = None,
        update_stats: bool = True,
    ) -> dict:
        """
        对一个 HTTP 请求做出 WAF 决策。

        Args:
            source_ip: 源 IP 地址
            features: 32 维特征向量（可用 extract_features_from_request 生成）
            detections: 外部检测引擎评分，如 {'sqli':0.9, 'xss':0.0}
            update_stats: 是否更新统计

        Returns:
            dict 包含 action, confidence, uncertainty, reason, 等
        """
        total_start = time.time()

        # ── 注入检测信号 ──
        feats = list(features)
        if detections:
            feats = inject_detection_signals(feats, detections)

        # ═══ L1: IP 黑白名单 ═══
        # 白名单 → 标记但不跳过（仍走 ML 收集训练数据）
        is_whitelisted = False
        for cidr in self.allowlist:
            if ip_in_network(source_ip, cidr):
                is_whitelisted = True
                if update_stats: self.stats['allowlist_hits'] += 1
                break

        # 黑名单 → 直接 BLOCK
        for cidr in self.blocklist:
            if ip_in_network(source_ip, cidr):
                if update_stats: self.stats['blocklist_hits'] += 1
                if update_stats: self.stats['blocked'] += 1
                return self._make_result('BLOCK', 1.0, source_ip,
                                         reason='黑名单封禁')

        # ═══ L2: 临时封禁检查（白名单跳过）═══
        ctx = self._get_context(source_ip)
        now = time.time()
        if not is_whitelisted and ctx.blocked_until > now:
            remaining = ctx.blocked_until - now
            if update_stats: self.stats['blocked'] += 1
            return self._make_result('BLOCK', 0.95, source_ip,
                                     reason=f'临时封禁中 (剩余 {remaining:.0f}s)',
                                     block_remaining=remaining)

        # ═══ L3: 模型推理 ═══
        # 构造输入张量
        feat_tensor = torch.tensor(feats[:32], dtype=torch.float, device=self.device).unsqueeze(0)

        # 获取/初始化 RNN 状态
        if ctx.rnn_state is None:
            ctx.rnn_state = self.model.init_state(1, self.device)
            prev_action = None
        else:
            prev_action = ctx.prev_action

        # 模型推理
        infer_start = time.time()
        with torch.no_grad():
            out = self.model(feat_tensor, ctx.rnn_state, prev_action)
        infer_time = (time.time() - infer_start) * 1000  # ms

        # 更新上下文
        pred_action = out['action_idx'].item()
        ctx.rnn_state = out['new_rnn_state']
        ctx.prev_action = torch.tensor([pred_action], device=self.device)
        ctx.request_count += 1
        ctx.last_seen = now

        action_name = ACTION_NAMES[pred_action]
        uncertainty = out['uncertainty'].mean().item()
        confidence = 1.0 - uncertainty

        # ═══ L4: 置信度门控 ═══
        # 如果模型不确定性太高 → 降级到保守决策
        downgraded = False
        if uncertainty > self.uncertainty_threshold:
            downgraded = True
            if update_stats: self.stats['confidence_downgrades'] += 1
            # 不确定性高时：ALLOW + 记录，不执行高风险动作
            action_name = 'ALLOW'
            confidence = max(0.3, 0.5 - uncertainty)
            reason = f'不确定性高 ({uncertainty:.2f})，保守放行'
        else:
            # 根据类型设置最低置信度门槛
            min_conf = CONF_THRESHOLDS[action_name]
            if confidence < min_conf:
                downgraded = True
                if update_stats: self.stats['confidence_downgrades'] += 1
                # 置信度不足时降一级
                if pred_action == 2:  # BLOCK→MONITOR
                    action_name = 'MONITOR'
                    confidence = min_conf
                    reason = f'BLOCK 置信度不足 ({confidence:.2f}<{min_conf:.2f})，降级 MONITOR'
                elif pred_action == 1:  # MONITOR→ALLOW
                    action_name = 'ALLOW'
                    confidence = min_conf
                    reason = f'MONITOR 置信度不足 ({confidence:.2f}<{min_conf:.2f})，降级 ALLOW'
                else:  # ALLOW 置信度不足 → 升 MONITOR
                    action_name = 'MONITOR'
                    confidence = min_conf
                    reason = f'ALLOW 置信度不足 ({confidence:.2f}<{min_conf:.2f})，升级 MONITOR'
            else:
                reason = f'{action_name} (confidence={confidence:.2f})'

        # ═══ 攻击计数(用于 RNN 上下文) ═══
        if action_name == 'BLOCK' or (detections and max(detections.values(), default=0) > 0.5):
            ctx.attack_count += 1

        # 统计
        if update_stats:
            self.stats['total_requests'] += 1
            self.stats[{'ALLOW': 'allowed', 'MONITOR': 'monitored', 'BLOCK': 'blocked'}[action_name]] += 1

        total_time = (time.time() - total_start) * 1000

        # 白名单覆盖：模型照跑但强制放行
        if is_whitelisted:
            action_name = 'ALLOW'
            reason = f'白名单放行 (模型原始: {ACTION_NAMES[out["action_idx"].item()]})'
            # 修正统计
            self.stats['allowed'] += 1

        return {
            'action': action_name,
            'action_level': ACTION_LEVEL[action_name],
            'raw_action': out['action_idx'].item(),  # 置信度门控前的原始输出
            'is_whitelisted': is_whitelisted,
            'confidence': round(confidence, 4),
            'uncertainty': round(uncertainty, 4),
            'strategy': out['strategy_idx'].item(),
            'value': round(out['value'].item(), 4),
            'containment': round(out['containment_prob'].item(), 4),
            'predicted_state': out['predicted_state'][0].tolist()[:8],
            'state': out['state'][0].tolist()[:8],
            'reason': reason,
            'downgraded': downgraded,
            'infer_time_ms': round(infer_time, 3),
            'total_time_ms': round(total_time, 3),
            'ip': source_ip,
        }

    # ── 内部方法 ─────────────────────────────────────────────────

    def _get_context(self, ip: str) -> RequestContext:
        """获取或创建 IP 的请求上下文。同时清理过期上下文。"""
        now = time.time()
        with self._lock:
            # 清理过期上下文（300 秒无活动）
            stale_ips = [k for k, v in self._contexts.items()
                         if now - v.last_seen > self.context_timeout]
            for k in stale_ips:
                del self._contexts[k]

            if ip not in self._contexts:
                self._contexts[ip] = RequestContext(ip=ip)
            return self._contexts[ip]

    def _make_result(self, action: str, confidence: float, ip: str, **extra) -> dict:
        """构造标准结果字典。"""
        return {
            'action': action,
            'action_level': ACTION_LEVEL[action],
            'confidence': round(confidence, 4),
            'uncertainty': 0.0,
            'reason': extra.pop('reason', ''),
            'infer_time_ms': 0.0,
            'total_time_ms': 0.0,
            'ip': ip,
            'downgraded': False,
            **extra,
        }

    def reset_context(self, ip: str = None):
        """重置指定 IP（或全部）的 RNN 上下文。"""
        with self._lock:
            if ip:
                self._contexts.pop(ip, None)
            else:
                self._contexts.clear()

    def get_stats(self) -> dict:
        """获取引擎统计。"""
        with self._lock:
            active_ips = len(self._contexts)
        return {
            **self.stats,
            'active_contexts': active_ips,
        }


# ═══════════════════════════════════════════════════════════════════
# 演示/测试入口
# ═══════════════════════════════════════════════════════════════════

def demo():
    """从模拟器数据加载样本，演示 WAFEngine 的连续请求决策。"""
    # 加载测试数据
    data_path = os.path.join(_here, 'data', 'waf_trajectories_500ep.jsonl')
    if not os.path.exists(data_path):
        print(f'[Demo] 数据文件不存在: {data_path}')
        print('[Demo] 请先生成数据: python scripts/waf_simulator.py --episodes 200')
        return

    with open(data_path, encoding='utf-8') as f:
        samples = [json.loads(l) for l in f]

    # 初始化引擎
    engine = WAFEngine(
        model_path=os.path.join(_here, 'out_waf_3act', 'standalone_waf_final.pth'),
    )
    engine.allowlist_add('192.168.0.0/16')

    print('\n' + '='*60)
    print('WAF 引擎演示: 连续请求决策 (每 IP 独立 RNN 状态)')
    print('='*60)
    print()

    # 模拟连续请求（每个 episode 是一个 IP 的连续请求序列）
    current_ep = -1
    for i, s in enumerate(samples[:30]):
        ep = s['metadata']['episode']
        ip = s['metadata']['source_ip'] or f'10.0.0.{ep % 255}'
        feats = s['structured'].get('features', [])

        # 新 episode = 新 IP
        if ep != current_ep:
            current_ep = ep
            print(f'\n── 新 IP {ip} 的请求序列 ──')

        if not feats:
            break

        # 注入检测信号（从 metadata 的攻击类型映射）
        detections = {}
        if s['metadata']['is_attack']:
            atk = s['metadata']['attack_type']
            detections = {atk.lower(): 0.85}

        result = engine.decide(ip, feats, detections)

        atk_tag = f"[{s['metadata']['attack_type']:>12s}]" if s['metadata']['is_attack'] else '[  正常  ]'
        unc = result['uncertainty']
        action_label = result['action']
        conf_label = result['confidence']
        reason_label = result['reason']
        dg = result.get('downgraded', False)
        dm = ' !!DEGRADE' if dg else ''
        print(f'  {atk_tag} {action_label:<8s}  conf={conf_label:.2f}  '
              f'unc={unc:.2f}  {reason_label}{dm}')


if __name__ == '__main__':
    demo()
