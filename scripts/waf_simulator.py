"""
WAFSimulator — WAF 版 NetworkWorld：HTTP 请求级别攻击-防御模拟器。

生成结构化的多步决策轨迹，直接用于 GameNNDecisionHead 训练。

工作流程：
  1. 模拟一个 Web 应用收到连续 HTTP 请求流（正常 + 攻击混合）
  2. 每个请求触发一个 WAF 决策点（放行/记录/限速/封禁等）
  3. 用 16 维状态向量跟踪全局安全态势
  4. 输出与 gen_simulator_data.py 完全兼容的 JSONL 格式

Usage:
    python scripts/waf_simulator.py --episodes 200 --output ../data/waf_trajectories.jsonl
"""
import os, sys, json, random, math, argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

# ─── 共享常量（与 decision_head.py 对齐）───────────────────────────

WAF_ACTIONS = [
    ('ALLOW', '放行请求', 2),            # 确信安全
    ('LOG_ONLY', '仅记录日志', 2),       # 不确定但无害，记录观察
    ('RATE_LIMIT', '限速', 2),           # 可疑但非明显恶意
    ('CHALLENGE', '发起挑战', 1),        # CAPTCHA/JS 挑战验证
    ('BLOCK_IP', '封禁源IP', 0),         # 封禁 IP
    ('BLOCK_SESSION', '封禁会话', 1),    # 封禁会话/token
    ('REDIRECT_HONEYPOT', '重定向到蜜罐', 0),  # 诱捕
    ('ESCALATE', '升级至人工审核', 1),   # 扔给安全分析师
]

STRATEGY_NAMES = ['aggressive', 'balanced', 'defensive']
STRATEGY_DESC = {
    0: '主动反制：封锁 IP、蜜罐诱捕、攻击溯源',
    1: '均衡应对：限速验证、日志分析、风险定价',
    2: '防御优先：放行 benign、低误报、保守阈值',
}

# 策略 → 允许的动作映射
STRATEGY_ACTION_MAP = {
    0: [4, 5, 6, 7],  # aggressive → BLOCK_IP, BLOCK_SESSION, HONEYPOT, ESCALATE
    1: [1, 2, 3, 0],  # balanced  → LOG_ONLY, RATE_LIMIT, CHALLENGE, ALLOW
    2: [0, 1, 2, 3],  # defensive → ALLOW, LOG_ONLY, RATE_LIMIT, CHALLENGE
}

# WAF 状态下 16 维命名（与 decision_head.py STATE_NAMES 对齐但重新诠释为 WAF 场景）
WAF_STATE_NAMES = [
    'threat_severity',         # 0: 当前请求威胁严重度
    'threat_type_code',        # 1: 攻击类型编码 (0-7 映射 8 种攻击)
    'attack_surface',          # 2: 受攻击面（不同 URL/参数 比例）
    'lateral_movement_risk',   # 3: IP 跳转/代理链风险
    'data_exfil_risk',         # 4: 数据外泄风险（敏感接口请求频率）
    'persistence_risk',        # 5: 同一源持续攻击强度
    'detection_level',         # 6: 检测特征匹配度
    'attacker_sophistication', # 7: 攻击者绕过能力估计
    'critical_asset_count',    # 8: 受影响关键资产/接口数
    'blocked_ratio',           # 9: 该 IP 历史封禁率
    'false_positive_risk',     # 10: 误报风险累积
    'compromised_ratio',       # 11: 已被突破的端点比例
    'alert_level',             # 12: 告警风暴级别
    'rate_ratio',              # 13: 请求速率比（相对基线）
    'session_anomaly',         # 14: 会话异常得分
    'response_phase',          # 15: 响应阶段 (0=正常, 1=监测, 2=响应, 3=恢复)
]

# ─── 攻击类型定义 ──────────────────────────────────────────────

ATTACK_TYPES = [
    ('SQLi', 'SQL注入'),
    ('XSS', '跨站脚本'),
    ('PathTraversal', '路径遍历'),
    ('CmdInjection', '命令注入'),
    ('SSRF', '服务端请求伪造'),
    ('FileUpload', '恶意文件上传'),
    ('BruteForce', '暴力破解'),
    ('ScannerBot', '扫描器/爬虫'),
]

ATTACK_DIFFICULTY = {  # 绕过难度 (0=easy, 1=medium, 2=hard)
    'SQLi': 0,
    'XSS': 1,
    'PathTraversal': 0,
    'CmdInjection': 1,
    'SSRF': 2,
    'FileUpload': 2,
    'BruteForce': 0,
    'ScannerBot': 0,
}

# ─── WAF 状态类 ─────────────────────────────────────────────────

@dataclass
class WAFState:
    """16 维 WAF 安全态势状态。"""
    values: List[float] = field(default_factory=lambda: [0.0] * 16)

    def __getitem__(self, idx):
        return self.values[idx]

    def __setitem__(self, idx, val):
        self.values[idx] = max(0.0, min(1.0, val))

    def copy(self) -> 'WAFState':
        return WAFState(self.values.copy())

    @property
    def severity(self): return self.values[0]
    @property
    def compromised_ratio(self): return self.values[11]
    @property
    def alert_level(self): return self.values[12]
    @property
    def rate_ratio(self): return self.values[13]

    def to_dict(self):
        return {WAF_STATE_NAMES[i]: self.values[i] for i in range(16)}

    def describe(self) -> str:
        """生成本轮态势的自然语言描述。"""
        parts = []
        sev = self.severity
        if sev < 0.2:
            parts.append("当前威胁低")
        elif sev < 0.5:
            parts.append("当前威胁中等")
        elif sev < 0.8:
            parts.append("当前威胁高")
        else:
            parts.append("当前威胁严重")

        al = self.alert_level
        if al > 0.7:
            parts.append("告警量大")
        elif al < 0.2:
            parts.append("告警量低")

        rr = self.rate_ratio
        if rr > 0.8:
            parts.append("请求速率异常高")
        elif rr > 0.5:
            parts.append("请求速率偏高")

        comp = self.compromised_ratio
        if comp > 0.3:
            parts.append(f"疑似失陷端点 {comp:.0%}")

        return "，".join(parts)


# ─── HTTP 请求类 ──────────────────────────────────────────────

@dataclass
class HTTPRequest:
    """模拟的 HTTP 请求。"""
    method: str               # GET/POST/PUT/DELETE
    path: str                 # /api/user/login
    query_params: dict = field(default_factory=dict)
    body: str = ''
    headers: dict = field(default_factory=dict)
    source_ip: str = ''
    session_id: str = ''
    is_attack: bool = False
    attack_type: str = ''
    payload: str = ''
    timestamp: float = 0.0
    feature_vector: List[float] = field(default_factory=lambda: [0.0] * 32)
    # 已提取的特征向量，供 DomainEncoder 用
    # [0-7]: 八大攻击类型的特征分
    # [8-15]: 请求形态特征（长度/参数数/特殊字符率等）
    # [16-23]: 时序特征（近期同类请求频率）
    # [24-31]: 会话特征（cookie/UA/Referer一致性）


# ─── IP 地址池 ────────────────────────────────────────────────

def random_ip() -> str:
    return f'{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}'

def ip_in_same_subnet(ip: str, other: str, prefix: int = 24) -> bool:
    """检查两个 IP 是否在同一子网。"""
    mask = (0xFFFFFFFF << (32 - prefix)) & 0xFFFFFFFF
    def ip_to_int(s):
        parts = [int(x) for x in s.split('.')]
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    return (ip_to_int(ip) & mask) == (ip_to_int(other) & mask)


# ─── 攻击生成器 ────────────────────────────────────────────────

class AttackGenerator:
    """生成各种 Web 攻击的 HTTP 请求。"""

    PATHS_BY_TYPE = {
        'SQLi':      ['/api/user/login', '/api/search', '/product', '/article', '/query'],
        'XSS':       ['/api/comment', '/feedback', '/profile', '/search'],
        'PathTraversal': ['/download', '/file', '/static', '/api/attachment'],
        'CmdInjection':  ['/api/ping', '/api/dns', '/api/exec', '/tools'],
        'SSRF':      ['/api/fetch', '/api/proxy', '/api/redirect', '/webhook'],
        'FileUpload':    ['/api/upload', '/import', '/attachments'],
        'BruteForce':   ['/api/user/login', '/admin', '/api/token', '/oauth'],
        'ScannerBot':   ['/robots.txt', '/.env', '/wp-admin', '/admin.php', '/backup'],
    }

    SQLI_PAYLOADS = [
        "' OR '1'='1", "1' AND 1=1--", "' UNION SELECT * FROM users--",
        "1; DROP TABLE users--", "' OR 1=1#", "\" OR \"1\"=\"1",
        "admin'--", "1' ORDER BY 10--", "' AND SLEEP(5)--",
    ]
    XSS_PAYLOADS = [
        "<script>alert(1)</script>", "<img src=x onerror=alert(1)>",
        "javascript:alert(1)", "\"><script>alert(1)</script>",
        "<svg onload=alert(1)>", "';alert(1)//",
    ]
    TRAVERSAL_PAYLOADS = [
        "../../../etc/passwd", "..\\..\\..\\windows\\win.ini",
        "%2e%2e%2f%2e%2e%2fetc/passwd", "....//....//etc/passwd",
        "../../../../etc/shadow",
    ]
    CMDI_PAYLOADS = [
        ";id", "|id", "`id`", "$(id)", "| ping -c 10 127.0.0.1",
        "& whoami &", "|| dir",
    ]
    SSRF_PAYLOADS = [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:22", "file:///etc/passwd",
        "gopher://localhost:6379/_...", "dict://localhost:6379/info",
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(self, attack_type: str, sophistication: float = 0.3) -> HTTPRequest:
        """生成指定类型的攻击请求。sophistication 越高，绕过特征越隐蔽。"""
        paths = self.PATHS_BY_TYPE.get(attack_type, ['/'])
        path = self.rng.choice(paths)
        method = self.rng.choice(['GET', 'POST'])

        payload = self._get_payload(attack_type, sophistication)
        query_params = {}
        body = ''

        if method == 'GET':
            param_name = self.rng.choice(['id', 'q', 'page', 'file', 'url', 'cmd', 'search', 'debug'])
            query_params = {param_name: payload}
        else:
            param_name = self.rng.choice(['username', 'password', 'content', 'data', 'input'])
            body = f'{param_name}={requests_quote(payload)}'

        # 根据 sophistication 决定是否加绕过
        evasive_headers = {}
        if sophistication > 0.5:
            evasive_headers['X-Forwarded-For'] = random_ip()
        if sophistication > 0.7:
            evasive_headers['User-Agent'] = self.rng.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15',
                'curl/7.68.0',
            ])
            # 加垃圾参数混淆
            query_params['_t'] = str(self.rng.randint(100000, 999999))
            query_params['utm_source'] = 'google'

        if sophistication > 0.8 and attack_type == 'SQLi':
            # 高级绕过：注释内联
            payload = payload.replace("'", "%25%37")  # 双重编码

        feat = [0.0] * 32
        attack_idx = [a[0] for a in ATTACK_TYPES].index(attack_type)
        feat[attack_idx] = 1.0  # 攻击类型 one-hot
        feat[8] = len(payload) / 1000.0  # 载荷长度
        feat[9] = sum(c in "'\"%;<>()|&`$" for c in payload) / max(len(payload), 1)  # 特殊字符率
        feat[10] = 1.0 if '..' in payload else 0.0
        feat[11] = 1.0 if 'alert' in payload or 'SLEEP' in payload else 0.0
        feat[12] = min(1.0, len(query_params) / 10.0)
        feat[16] = sophistication  # 攻击者复杂度

        return HTTPRequest(
            method=method,
            path=path,
            query_params=query_params,
            body=body,
            headers=evasive_headers,
            is_attack=True,
            attack_type=attack_type,
            payload=payload,
            feature_vector=feat,
        )

    def _get_payload(self, attack_type: str, sophistication: float) -> str:
        """根据攻击类型和复杂度选取 payload。"""
        pools = {
            'SQLi': self.SQLI_PAYLOADS,
            'XSS': self.XSS_PAYLOADS,
            'PathTraversal': self.TRAVERSAL_PAYLOADS,
            'CmdInjection': self.CMDI_PAYLOADS,
            'SSRF': self.SSRF_PAYLOADS,
            'FileUpload': ['shell.php', 'test.aspx;.jpg', 'image.svg', '.htaccess'],
            'BruteForce': ['admin', 'root', 'test', 'password123', 'admin123'],
            'ScannerBot': ['', '/', 'index.php'],
        }
        pool = pools.get(attack_type, ['test'])
        # 低 sophistication 用明显 payload，高用复杂变种
        if sophistication < 0.3:
            return pool[0]
        elif sophistication < 0.6:
            return self.rng.choice(pool[:max(1, len(pool)//2)])
        else:
            return self.rng.choice(pool)


def requests_quote(s: str) -> str:
    """简单 URL 编码。"""
    import urllib.parse
    return urllib.parse.quote(s, safe='')


# ─── 正常流量生成器 ────────────────────────────────────────────

class BenignTrafficGenerator:
    """生成正常的用户行为请求。"""

    PATHS = [
        '/', '/api/user/login', '/api/user/profile', '/api/products',
        '/api/product/1', '/api/product/2', '/api/search',
        '/api/cart', '/api/checkout', '/api/order',
        '/static/styles.css', '/static/app.js', '/static/logo.png',
        '/about', '/contact', '/help',
    ]
    METHODS = ['GET', 'GET', 'GET', 'GET', 'POST']  # 80% GET

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed + 100)

    def generate(self, session_id: str = '') -> HTTPRequest:
        path = self.rng.choice(self.PATHS)
        method = self.rng.choice(self.METHODS)
        query_params = {}
        body = ''

        if 'login' in path:
            body = 'username=user&password=pass'
        elif 'search' in path:
            query_params['q'] = self.rng.choice(['phone', 'laptop', 'book', 'shoes'])
            query_params['page'] = str(self.rng.randint(1, 5))

        feat = [0.0] * 32
        feat[8] = self.rng.gauss(0.2, 0.05)  # 正常长度
        feat[9] = self.rng.gauss(0.05, 0.02)  # 低特殊字符率
        feat[12] = self.rng.gauss(0.15, 0.05)  # 参数数适中
        feat[13] = self.rng.gauss(0.1, 0.03)   # 正常UA

        return HTTPRequest(
            method=method,
            path=path,
            query_params=query_params,
            body=body,
            source_ip='',
            session_id=session_id,
            is_attack=False,
            feature_vector=feat,
        )


# ─── WAF 模拟环境 ──────────────────────────────────────────────

class WAFSimEnv:
    """
    WAF 攻击-防御模拟环境。

    每轮 = 一个 HTTP 请求（正常或攻击）。
    在特定触发点（检测到攻击 or 达到决策间隔）停下要求 WAF 决策。
    """

    # 各攻击类型的 WAF 特征分（检测特征匹配 0~1）
    DETECTION_SIGNATURES = {
        'SQLi': 0.85,
        'XSS': 0.75,
        'PathTraversal': 0.70,
        'CmdInjection': 0.80,
        'SSRF': 0.60,
        'FileUpload': 0.55,
        'BruteForce': 0.90,
        'ScannerBot': 0.85,
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.attack_gen = AttackGenerator(seed)
        self.benign_gen = BenignTrafficGenerator(seed)

        # 当前状态
        self.state = WAFState()
        self.prev_state = WAFState()
        self.time_step = 0
        self.max_steps = 100
        self.total_requests = 0
        self.blocked_ips = set()
        self.blocked_sessions = set()
        self.attack_history = []  # (type, success, ip)
        self.recent_requests = []  # 最近 50 个请求记录
        self.ip_request_count = {}  # IP → 请求数
        self.ip_attack_count = {}   # IP → 攻击次数
        self.ip_last_blocked = {}   # IP → 上次被封禁时间
        self.session_request_count = {}

        # 攻击者正在使用的 IP/策略
        self.attacker_ips = [random_ip() for _ in range(5)]
        self.current_attacker_ip = self.attacker_ips[0]
        self.attacker_sophistication = 0.1
        self.attacker_phase = 0  # 0=probe, 1=exploit, 2=escalate, 3=persist
        self.consecutive_blocks = 0
        self.last_action = -1
        self.last_strategy = 2

        # 是否在等待决策
        self.waiting_for_decision = False
        self.last_request = None
        self.done = False

    def reset(self):
        """重置环境开始新的 episode。"""
        self.state = WAFState()
        self.prev_state = WAFState()
        self.time_step = 0
        self.total_requests = 0
        self.blocked_ips.clear()
        self.blocked_sessions.clear()
        self.attack_history.clear()
        self.recent_requests.clear()
        self.ip_request_count.clear()
        self.ip_attack_count.clear()
        self.ip_last_blocked.clear()
        self.session_request_count.clear()

        self.attacker_ips = [random_ip() for _ in range(5)]
        self.current_attacker_ip = self.attacker_ips[0]
        self.attacker_sophistication = 0.1
        self.attacker_phase = 0
        self.consecutive_blocks = 0
        self.last_action = -1
        self.last_strategy = 2
        self.waiting_for_decision = False
        self.last_request = None
        self.done = False

        # 初始状态值
        self.state[9] = 0.0   # blocked_ratio
        self.state[10] = 0.0  # false_positive_risk
        self.state[11] = 0.0  # compromised_ratio
        self.state[15] = 0.0  # response_phase

        return self._get_observation()

    def step(self, waf_action: int) -> Tuple[dict, float, bool, dict]:
        """
        执行一次 WAF 决策。
        不推进到下一个请求——调用方需要自己调用 _advance_to_next_decision。
        """
        if not self.waiting_for_decision or self.last_request is None:
            return {'request_feats': [0.0]*32, 'state': self.state.values.copy(), 'time_step': self.time_step,
                    'source_ip': '', 'session_id': '', 'path': '', 'method': 'GET'}, 0.0, self.done, {}

        req = self.last_request
        self.prev_state = self.state.copy()

        # ─── 执行 WAF 动作 ──────────────────────────────────
        action_name, _, _ = WAF_ACTIONS[waf_action]
        reward = 0.0

        if req.is_attack:
            if waf_action in (4, 5):  # BLOCK_IP / BLOCK_SESSION
                if waf_action == 4:
                    self.blocked_ips.add(req.source_ip)
                    self.ip_last_blocked[req.source_ip] = self.time_step
                else:
                    self.blocked_sessions.add(req.session_id)
                reward = 1.0
                self.consecutive_blocks += 1
                self.state[3] = min(1.0, self.state[3] + 0.1)

            elif waf_action == 0:  # ALLOW（放行攻击 → 重大扣分）
                reward = -2.0
                self.state[11] = min(1.0, self.state[11] + 0.05)
                self.attack_history.append((req.attack_type, True, req.source_ip))
                self.consecutive_blocks = 0

            elif waf_action in (1, 2, 3):  # LOG/RATE_LIMIT/CHALLENGE
                reward = 0.3
                self.consecutive_blocks = 0

            elif waf_action == 6:  # HONEYPOT
                reward = 0.8
                self.consecutive_blocks += 1

            elif waf_action == 7:  # ESCALATE
                reward = 0.6
                self.consecutive_blocks = 0

        else:  # 正常请求
            if waf_action == 0:
                reward = 0.2
            elif waf_action in (4, 5):
                reward = -3.0
                self.state[10] = min(1.0, self.state[10] + 0.1)
            elif waf_action in (2, 3):
                reward = -0.5
                self.state[10] = min(1.0, self.state[10] + 0.05)
            elif waf_action in (1, 7):
                reward = 0.0
            elif waf_action == 6:
                reward = -1.0

        self.last_action = waf_action
        self.waiting_for_decision = False

        # ─── 更新状态 ──────────────────────────────────────
        self._update_state(req, waf_action)

        # ─── 推进攻击者策略 ────────────────────────────────
        self._update_attacker()

        # ─── 检查终止 ──────────────────────────────────────
        self.time_step += 1
        if self.time_step >= self.max_steps:
            self.done = True

        return self._extract_features(req), reward, self.done, {
            'action_taken': action_name,
            'state_after': self.state.values.copy(),
        }

    def _update_state(self, req: HTTPRequest, action: int):
        """根据当前请求和 WAF 动作更新 16 维状态。"""
        s = self.state

        # 0: threat_severity — 最近攻击的加权严重度
        if req.is_attack:
            base = self.DETECTION_SIGNATURES.get(req.attack_type, 0.5)
            s[0] = min(1.0, s[0] * 0.7 + base * 0.3)
        else:
            s[0] = s[0] * 0.95  # 自然衰减

        # 1: threat_type_code — 当前主要攻击类型
        if req.is_attack:
            type_idx = [a[0] for a in ATTACK_TYPES].index(req.attack_type)
            s[1] = (type_idx + 1) / 8.0

        # 2: attack_surface — 受攻击的不同端点比例
        endpoints_hit = len(set(r.path for r in self.recent_requests if r.is_attack))
        s[2] = min(1.0, endpoints_hit / 15.0)

        # 3: lateral_movement_risk — IP 切换频率
        unique_attack_ips = len(set(r.source_ip for r in self.recent_requests if r.is_attack))
        s[3] = min(1.0, unique_attack_ips / 8.0)

        # 4: data_exfil_risk — 敏感接口请求频率
        sensitive_paths = sum(1 for r in self.recent_requests if '/api/user' in r.path or '/api/order' in r.path)
        s[4] = min(1.0, sensitive_paths / 20.0)

        # 5: persistence_risk — 同一源持续攻击强度
        ip_attacks = self.ip_attack_count.get(self.current_attacker_ip, 0)
        s[5] = min(1.0, ip_attacks / 10.0)

        # 6: detection_level — 检测特征匹配度
        if req.is_attack:
            s[6] = self.DETECTION_SIGNATURES.get(req.attack_type, 0.5)
        else:
            s[6] = s[6] * 0.9  # 正常请求衰减特征分

        # 7: attacker_sophistication
        s[7] = self.attacker_sophistication

        # 8: critical_asset_count
        endpoints_total = len(set(r.path for r in self.recent_requests))
        s[8] = min(1.0, endpoints_total / 20.0)

        # 9: blocked_ratio — 已封禁比例
        total_attackers = max(len(self.attacker_ips), 1)
        blocked_count = sum(1 for ip in self.attacker_ips if ip in self.blocked_ips)
        s[9] = blocked_count / total_attackers

        # 10: false_positive_risk
        s[10] = max(0.0, s[10])  # 已在误封时更新

        # 11: compromised_ratio — 被突破端点比例
        successful_attacks = len(self.attack_history)
        s[11] = min(1.0, successful_attacks / 15.0)

        # 12: alert_level
        recent_attacks = sum(1 for r in self.recent_requests if r.is_attack)
        s[12] = min(1.0, recent_attacks / 10.0)

        # 13: rate_ratio — 请求速率比
        total_recent = len(self.recent_requests)
        s[13] = min(1.0, total_recent / 30.0)

        # 14: session_anomaly
        s[14] = min(1.0, s[11] * 0.5 + s[5] * 0.3 + s[3] * 0.2)

        # 15: response_phase
        if s[11] > 0.3:
            s[15] = 0.5  # 应急响应
        elif s[11] > 0.6:
            s[15] = 0.75  # 灾备恢复
        elif s[0] < 0.2 and s[12] < 0.2:
            s[15] = 0.0  # 正常

    def _update_attacker(self):
        """推进攻击者策略：随时间升级攻击复杂度。"""
        self.attacker_sophistication = min(1.0, self.attacker_sophistication + 0.02)

        # 如果被封禁则切换 IP
        if self.current_attacker_ip in self.blocked_ips:
            available = [ip for ip in self.attacker_ips if ip not in self.blocked_ips]
            if available:
                self.current_attacker_ip = self.rng.choice(available)
            else:
                # 所有 IP 被封，换一批
                self.attacker_ips = [random_ip() for _ in range(5)]
                self.current_attacker_ip = self.attacker_ips[0]

        # 攻击阶段推进
        if self.time_step > self.max_steps * 0.7:
            self.attacker_phase = 3
        elif self.time_step > self.max_steps * 0.5:
            self.attacker_phase = 2
        elif self.time_step > self.max_steps * 0.3:
            self.attacker_phase = 1

    def _advance_to_next_decision(self, base_reward: float = 0.0):
        """推进到下一个需要 WAF 决策的请求。返回 (obs, reward, done, info)。"""
        while not self.done:
            # ── 决定下一个请求是正常还是攻击 ──
            # 攻击概率随时间上升
            attack_prob = min(0.6, 0.2 + self.attacker_sophistication * 0.3)

            if self.rng.random() < attack_prob:
                # 生成攻击请求
                attack_types = [a[0] for a in ATTACK_TYPES]
                # 攻击阶段决定攻击类型
                phase_attacks = {
                    0: ['ScannerBot', 'SQLi', 'PathTraversal'],
                    1: ['SQLi', 'XSS', 'CmdInjection', 'BruteForce'],
                    2: ['SSRF', 'FileUpload', 'CmdInjection'],
                    3: ['SSRF', 'FileUpload', 'XSS'],
                }
                candidates = phase_attacks.get(self.attacker_phase, attack_types)
                atk_type = self.rng.choice(candidates)
                req = self.attack_gen.generate(atk_type, self.attacker_sophistication)
                req.source_ip = self.current_attacker_ip
                req.session_id = f'sess_{self.rng.randint(1000, 9999)}'
            else:
                # 生成正常请求
                req = self.benign_gen.generate()
                req.source_ip = random_ip()
                req.session_id = f'sess_{self.rng.randint(1000, 9999)}'

            req.timestamp = self.time_step
            self.total_requests += 1

            # 更新 IP 计数
            self.ip_request_count[req.source_ip] = self.ip_request_count.get(req.source_ip, 0) + 1
            if req.is_attack:
                self.ip_attack_count[req.source_ip] = self.ip_attack_count.get(req.source_ip, 0) + 1

            # 保存到最近请求列表
            self.recent_requests.append(req)
            if len(self.recent_requests) > 50:
                self.recent_requests.pop(0)

            # ── 更新特征向量中的时序维度 ──
            self._update_feature_timing(req)

            # ── 判断是否需要 WAF 决策 ──
            should_trigger = False
            trigger_reason = ''

            if req.is_attack:
                # 攻击请求必须触发决策
                should_trigger = True
                trigger_reason = f'检测到{req.attack_type}攻击'
            elif self.rng.random() < 0.02:
                # 少量正常请求也触发（训练模型认识正常流量）
                should_trigger = True
                trigger_reason = '抽样检查'
            elif self.state[12] > 0.6:
                # 告警风暴中更频繁触发
                should_trigger = True
                trigger_reason = '告警级别高，主动巡检'

            if should_trigger:
                self.waiting_for_decision = True
                self.last_request = req

                obs = self._extract_features(req)
                # 保存决策点信息
                info = {
                    'request': req,
                    'state': self.state.values.copy(),
                    'trigger_reason': trigger_reason,
                    'attack_type': req.attack_type if req.is_attack else '',
                    'is_attack': req.is_attack,
                }
                return obs, base_reward, self.done, info

            self.time_step += 1
            if self.time_step >= self.max_steps:
                self.done = True
                return self._extract_features(req), base_reward, True, {}

        return self._extract_features(self.last_request or HTTPRequest('GET', '/')), base_reward, True, {}

    def _update_feature_timing(self, req: HTTPRequest):
        """更新请求的时序特征维度。"""
        atk_type = req.attack_type if req.is_attack else ''
        type_idx = [a[0] for a in ATTACK_TYPES].index(atk_type) if atk_type in [a[0] for a in ATTACK_TYPES] else -1

        # [16-23]: 近期同类攻击频率
        for i in range(8):
            if i == type_idx:
                req.feature_vector[16 + i] = 1.0
            else:
                # 衰减
                if len(self.recent_requests) > 1:
                    req.feature_vector[16 + i] = req.feature_vector[16 + i] * 0.9

        # [24]: 该 IP 历史攻击次数归一化
        req.feature_vector[24] = min(1.0, self.ip_attack_count.get(req.source_ip, 0) / 20.0)

        # [25]: 同一 session 请求数
        sess_count = self.session_request_count.get(req.session_id, 0)
        self.session_request_count[req.session_id] = sess_count + 1
        req.feature_vector[25] = min(1.0, sess_count / 50.0)

        # [26]: 历史封禁指标
        req.feature_vector[26] = 1.0 if req.source_ip in self.blocked_ips else 0.0

        # [27]: 请求间隔
        req.feature_vector[27] = min(1.0, self.time_step / self.max_steps)

    def _extract_features(self, req: HTTPRequest) -> dict:
        """提取当前观测（供 DomainEncoder 用）。"""
        return {
            'request_feats': req.feature_vector.copy(),    # 32 维请求特征
            'state': self.state.values.copy(),              # 16 维态势状态
            'time_step': self.time_step,
            'source_ip': req.source_ip,
            'session_id': req.session_id,
            'path': req.path,
            'method': req.method,
        }

    def _get_observation(self):
        """reset 后返回初始观测。"""
        return self._extract_features(HTTPRequest('GET', '/'))


# ═══════════════════════════════════════════════════════════
# 数据生成
# ═══════════════════════════════════════════════════════════

def episode_to_samples(env: WAFSimEnv, episode_log: List[dict],
                       ep_id: int) -> List[dict]:
    """将一轮完整的 episode 转换为一组训练样本（多步轨迹）。"""
    samples = []

    for step_idx, log in enumerate(episode_log):
        state_vec = log['state']
        action_idx = log['action']
        reward = log['reward']
        strategy = log['strategy']
        action_name, action_desc, _ = WAF_ACTIONS[action_idx]

        # ── 构建用户消息 ──
        situation = log['state_description']
        attack_info = f"检测到{log['attack_type']}攻击" if log['is_attack'] else '正常请求'

        if step_idx == 0:
            user_text = (
                f'WAF 监测到新的 HTTP 请求。{attack_info}。'
                f'当前态势：{situation}。'
                f'请做出 WAF 决策。'
            )
        else:
            prev_action_name, _, _ = WAF_ACTIONS[episode_log[step_idx - 1]['action']]
            user_text = (
                f'上一步执行{prev_action_name}（reward={log["prev_reward"]:+.2f}），'
                f'{attack_info}，当前：{situation}。继续决策。'
            )

        # ── 构建助手回复 ──
        think_lines = []
        sev = state_vec[0]
        if action_idx in (4, 5):
            think_lines.append(f'威胁严重度{sev:.0%}，需要立即封禁')
        elif action_idx in (2, 3):
            think_lines.append(f'威胁中等（严重度{sev:.0%}），先验证再决定')
        elif action_idx == 0:
            think_lines.append(f'威胁低（严重度{sev:.0%}），放行')
        else:
            think_lines.append(f'威胁度{sev:.0%}，记录日志持续观察')

        if reward > 0:
            think_lines.append(f'决策效果正面（reward={reward:+.2f})')
        elif reward < 0:
            think_lines.append(f'决策效果不佳（reward={reward:.2f}），需调整')
        think_lines.append(f'执行{action_name}：{action_desc}')

        think_text = '\n'.join(think_lines)
        containment = max(0.1, min(0.95, 0.5 + reward * 0.3))
        assistant_text = (
            f'<think>\n{think_text}\n</think>\n\n'
            f'建议执行{action_name}，预期遏制概率{containment:.0%}。'
        )

        conversations = [
            {'role': 'system', 'content': '你是一个专业的 WAF 安全引擎。'},
            {'role': 'user', 'content': user_text},
            {'role': 'assistant', 'content': assistant_text},
        ]

        req_feats = log.get('features', [0.0]*32)[:32]

        structured = {
            'state': state_vec,
            'features': req_feats,
            'strategy': strategy,
            'strategy_name': STRATEGY_NAMES[strategy],
            'action': action_idx,
            'action_name': action_name,
            'value': containment,
        }

        samples.append({
            'conversations': conversations,
            'structured': structured,
            'metadata': {
                'episode': ep_id,
                'step': step_idx,
                'reward': reward,
                'is_attack': log['is_attack'],
                'attack_type': log['attack_type'],
                'source_ip': log.get('source_ip', ''),
            },
        })

    return samples


def run_one_episode(env: WAFSimEnv, max_decisions: int = 12) -> List[dict]:
    """运行一轮完整 episode，记录每一步的决策日志。"""
    env.reset()
    episode_log = []
    decision_count = 0
    prev_reward = 0.0

    # ── 获取第一个决策点 ──
    obs, _, done, info = env._advance_to_next_decision()

    while not env.done and decision_count < max_decisions:
        if not env.waiting_for_decision:
            break

        # ── 使用启发式最优 WAF 动作作为专家标签 ──
        optimal_action = _heuristic_waf_decision(env, info)
        strategy = _strategy_for_action(optimal_action)

        # 存储决策前状态
        state_before = env.state.values.copy()
        req = info['request']

        # 执行动作
        _, step_reward, done, _ = env.step(optimal_action)
        state_after = env.state.values.copy()

        attack_type = req.attack_type if req.is_attack else ''
        state_desc = env.state.describe()

        episode_log.append({
            'state': state_before,
            'state_after': state_after,
            'features': req.feature_vector,  # 32维原始特征
            'action': optimal_action,
            'strategy': strategy,
            'reward': step_reward,
            'prev_reward': prev_reward,
            'is_attack': req.is_attack,
            'attack_type': attack_type,
            'source_ip': req.source_ip,
            'state_description': state_desc,
            'trigger_reason': info['trigger_reason'],
        })
        decision_count += 1
        prev_reward = step_reward

        # ── 推进到下一个决策点 ──
        if not env.done:
            obs, _, done, info = env._advance_to_next_decision()

    return episode_log


def _heuristic_waf_decision(env: WAFSimEnv, info: dict) -> int:
    """
    启发式最优 WAF 决策（充当专家标注）。
    覆盖全部 8 种动作：
      0=ALLOW  1=LOG_ONLY  2=RATE_LIMIT  3=CHALLENGE
      4=BLOCK_IP  5=BLOCK_SESSION  6=HONEYPOT  7=ESCALATE
    """
    req = info['request']
    state = env.state
    sev = state[0]; persistence = state[5]; fp_risk = state[10]
    alert = state[12]; rate = state[13]; atk = req.attack_type
    SEVERE = ('CmdInjection','SSRF','BruteForce')
    MODERATE = ('SQLi','XSS','FileUpload')
    MILD = ('ScannerBot','PathTraversal')

    # 15% 探索偏向少见动作，保证8种均有训练样本
    if random.random() < 0.15 and req.is_attack:
        return random.choices([1,2,3,4,5,6,7],[0.05,0.15,0.15,0.05,0.2,0.2,0.2])[0]

    if not req.is_attack:
        return random.choices([0,1,3],[0.8,0.1,0.1])[0]

    if atk in SEVERE or sev > 0.6:
        if atk == 'BruteForce': return random.choices([4,5],[0.3,0.7])[0]
        return random.choices([4,5,6,7],[0.4,0.2,0.2,0.2])[0]

    mid = (atk in MODERATE or sev > 0.3)
    if mid:
        if persistence > 0.7: return 4
        if persistence > 0.35: return random.choices([2,3],[0.6,0.4])[0]
        if persistence > 0.15: return random.choices([1,3],[0.5,0.5])[0]
        return random.choices([1,2,3],[0.4,0.3,0.3])[0]

    if atk in MILD:
        if persistence > 0.5: return random.choices([1,2,3],[0.3,0.4,0.3])[0]
        return random.choices([0,1],[0.4,0.6])[0]

    return random.choices([0,1,2,3],[0.2,0.4,0.2,0.2])[0]


def _strategy_for_action(action_idx: int) -> int:
    """根据动作索引推断采用的策略。"""
    for strat, actions in STRATEGY_ACTION_MAP.items():
        if action_idx in actions:
            return strat
    return 1  # default balanced


def generate_waf_data(num_episodes: int = 200,
                      max_decisions: int = 12,
                      seed: int = 42) -> List[dict]:
    """主生成函数：运行 episodes 并返回训练样本。"""
    random.seed(seed)
    np.random.seed(seed)

    all_samples = []
    for ep in range(num_episodes):
        if ep % 20 == 0:
            print(f'  Episode {ep}/{num_episodes}...')

        env = WAFSimEnv(seed + ep)
        episode_log = run_one_episode(env, max_decisions)
        samples = episode_to_samples(env, episode_log, ep)
        all_samples.extend(samples)

    print(f'生成 {len(all_samples)} 个决策样本，来自 {num_episodes} 轮 episode')
    return all_samples


# ═══════════════════════════════════════════════════════════
# 保存 + 主入口
# ═══════════════════════════════════════════════════════════

def save_jsonl(samples: List[dict], path: str):
    """保存为 JSONL 格式。"""
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'已保存 {len(samples)} 个样本到 {path}')

    # 同时保存纯结构化标签
    labels_path = path.replace('.jsonl', '_labels.json')
    structures = [s['structured'] for s in samples]
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(structures, f, indent=2, ensure_ascii=False)
    print(f'标签已保存到 {labels_path}')


def print_stats(samples: List[dict]):
    """打印数据分布统计。"""
    strat_dist = {s: 0 for s in STRATEGY_NAMES}
    action_dist = {a[0]: 0 for a in WAF_ACTIONS}
    attack_count = 0
    normal_count = 0
    attack_types = {}
    rewards = []

    for s in samples:
        st = s['structured']
        strat_dist[STRATEGY_NAMES[st['strategy']]] += 1
        action_dist[st['action_name']] += 1
        meta = s['metadata']
        if meta['is_attack']:
            attack_count += 1
            at = meta['attack_type']
            attack_types[at] = attack_types.get(at, 0) + 1
        else:
            normal_count += 1
        rewards.append(meta['reward'])

    print(f'\n=== 数据统计 ===')
    print(f'总样本数: {len(samples)}')
    print(f'攻击请求: {attack_count} ({attack_count/max(len(samples),1)*100:.0f}%)')
    print(f'正常请求: {normal_count} ({normal_count/max(len(samples),1)*100:.0f}%)')
    print(f'\n策略分布: {strat_dist}')
    print(f'\n动作分布 (前5):')
    for name, count in sorted(action_dist.items(), key=lambda x: -x[1])[:5]:
        print(f'  {name}: {count}')
    print(f'\n攻击类型分布:')
    for at, count in sorted(attack_types.items(), key=lambda x: -x[1]):
        print(f'  {at}: {count}')
    print(f'\nReward 范围: [{min(rewards):.2f}, {max(rewards):.2f}], 均值={np.mean(rewards):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WAF 模拟器数据生成')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Episodes 数量')
    parser.add_argument('--max-decisions', type=int, default=12,
                        help='每轮最多决策次数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'),
                        help='输出目录')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f'运行 {args.episodes} 轮，每轮 {args.max_decisions} 步决策...\n')

    samples = generate_waf_data(args.episodes, args.max_decisions, args.seed)
    print_stats(samples)

    base = f'waf_trajectories_{args.episodes}ep'
    save_jsonl(samples, os.path.join(args.output, base + '.jsonl'))

    print('\n完成。')
