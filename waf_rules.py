"""
waf_rules.py — 静态规则引擎。

作为 ML 模型的前置过滤层：
  L1 规则引擎:  快速拒绝已知攻击 / 放行安全请求
  L2 ML 模型:   处理模糊地带（规则判定 UNCERTAIN 的请求）

规则设计原则:
  - 规则优先匹配，低开销
  - 明确恶意 → BLOCK
  - 明确安全 → ALLOW
  - 不确定 → UNCERTAIN（交给 ML）
"""
import re, time, json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field

RULERESULT_ALLOW = 0
RULERESULT_BLOCK = 1
RULERESULT_UNCERTAIN = 2  # 不确定，交给 ML

RULE_LEVELS = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}


@dataclass
class RuleMatch:
    """单条规则匹配结果。"""
    rule_id: str
    rule_name: str
    result: int          # ALLOW / BLOCK / UNCERTAIN
    severity: int = 1
    detail: str = ''
    timestamp: float = 0.0


@dataclass
class Rule:
    """一条静态规则。"""
    id: str
    name: str
    severity: str        # low / medium / high / critical
    enabled: bool = True
    description: str = ''

    def match(self, method: str, path: str, query: str, body: str,
              headers: dict, source_ip: str) -> Optional[RuleMatch]:
        """子类重写此方法。"""
        raise NotImplementedError


class RegexRule(Rule):
    """正则匹配规则。"""
    def __init__(self, id: str, name: str, severity: str,
                 patterns: List[str], target: str = 'all',
                 description: str = '', enabled: bool = True):
        super().__init__(id, name, severity, enabled, description)
        self.compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
        # target: 'path', 'query', 'body', 'all'
        self.target = target

    def match(self, method, path, query, body, headers, source_ip):
        for i, regex in enumerate(self.compiled):
            if self.target in ('path', 'all') and regex.search(path):
                return RuleMatch(self.id, self.name, RULERESULT_BLOCK,
                                 RULE_LEVELS[self.severity],
                                 f'规则匹配 [{self.name}] path 命中模式 #{i}')
            if self.target in ('query', 'all') and regex.search(query):
                return RuleMatch(self.id, self.name, RULERESULT_BLOCK,
                                 RULE_LEVELS[self.severity],
                                 f'规则匹配 [{self.name}] query 命中模式 #{i}')
            if self.target in ('body', 'all') and regex.search(body):
                return RuleMatch(self.id, self.name, RULERESULT_BLOCK,
                                 RULE_LEVELS[self.severity],
                                 f'规则匹配 [{self.name}] body 命中模式 #{i}')
        return None


class HeaderRule(Rule):
    """HTTP 头部检查规则。"""
    def __init__(self, id: str, name: str, severity: str,
                 check: str, pattern: str = None, description: str = '',
                 enabled: bool = True):
        super().__init__(id, name, severity, enabled, description)
        self.check = check   # 'missing_ua', 'bad_ua', 'missing_referer', etc.
        self.pattern = re.compile(pattern, re.IGNORECASE) if pattern else None

    def match(self, method, path, query, body, headers, source_ip):
        if self.check == 'missing_ua':
            ua = headers.get('user-agent', headers.get('User-Agent', ''))
            if not ua or len(ua) < 5:
                return RuleMatch(self.id, self.name, RULERESULT_UNCERTAIN,
                                 1, '缺少 User-Agent')
        return None


class RateLimitRule(Rule):
    """IP 速率限制规则。"""
    def __init__(self, id: str, name: str, severity: str,
                 max_requests: int = 100, window_seconds: int = 60,
                 block_seconds: int = 300, description: str = '',
                 enabled: bool = True):
        super().__init__(id, name, severity, enabled, description)
        self.max_requests = max_requests
        self.window = window_seconds
        self.block_duration = block_seconds
        self._history: Dict[str, List[float]] = defaultdict(list)

    def match(self, method, path, query, body, headers, source_ip):
        now = time.time()
        # 清理过期记录
        cutoff = now - self.window
        self._history[source_ip] = [t for t in self._history[source_ip] if t > cutoff]

        # 计数
        if len(self._history[source_ip]) >= self.max_requests:
            wait = cutoff - self._history[source_ip][0] + self.window
            return RuleMatch(self.id, self.name, RULERESULT_BLOCK,
                             RULE_LEVELS[self.severity],
                             f'速率限制: {len(self._history[source_ip])}次/{self.window}s')

        self._history[source_ip].append(now)
        return None


# ═══════════════════════════════════════════════════════════════════
# 规则库
# ═══════════════════════════════════════════════════════════════════

DEFAULT_RULES = [
    # ═══ SQL 注入 ═══
    RegexRule('sqli_01', 'SQLi UNION/SELECT', 'critical', [
        r'(\bUNION\b.{0,20}\bSELECT\b)', r'(\bSELECT\b.{0,20}\bFROM\b)',
    ], 'all', '检测 UNION SELECT 注入'),
    RegexRule('sqli_02', 'SQLi 布尔盲注', 'high', [
        r"'(\s*--|\s*#)", r'"(\s*--|\s*#)', r"'(\s*OR\s+\d+\s*=\s*\d+)",
        r"'(\s*AND\s+\d+\s*=\s*\d+)", r"'(\s*OR\s+')", r"'\s*=\s*'",
        r"'(\s*OR\s+\w+\s*=\s*\w+)",
        r"'?\s*\bor\b\s*\d+\s*=\s*\d+\s*'?", r"'?\s*\band\b\s*\d+\s*=\s*\d+\s*'?",
        r"\bOR\b.*\b1\b.*\b=\b.*\b1\b",
    ], 'all', '检测 SQL 布尔/单引号注入'),
    RegexRule('sqli_03', 'SQLi 时间盲注', 'critical', [
        r'SLEEP\s*\(\s*\d+\s*\)', r'WAITFOR\s+DELAY', r'pg_sleep\s*\(',
        r'BENCHMARK\s*\(', r'\bSLEEP\b',
    ], 'all', '检测 SQL 时间盲注'),
    RegexRule('sqli_04', 'SQLi 系统表/存储过程', 'high', [
        r'information_schema\.', r'mysql\.user', r'sys\.objects',
        r'sys\.tables', r'sys\.columns',
        r'\bexec\s+(sp_|master)', r'\bxp_cmdshell\b', r'\bxp_regread\b',
    ], 'all', '检测系统表/存储过程查询'),
    RegexRule('sqli_05', 'SQLi DML/DDL', 'critical', [
        r'\bDROP\s+TABLE\b', r'\bTRUNCATE\s+TABLE\b', r'\bALTER\s+TABLE\b',
        r'\bINSERT\s+INTO\b', r'\bDELETE\s+FROM\b',
        r'\bINTO\s+OUTFILE\b', r'\bLOAD_FILE\s*\(', r'\bINTO\s+DUMPFILE\b',
    ], 'all', '检测 SQL DDL/DML 操作'),
    RegexRule('sqli_06', 'SQLi 编码/函数', 'high', [
        r'CONVERT\s*\(', r'CHAR\s*\(\d+', r'UNHEX\s*\(',
        r'HEX\s*\(', r'0x[0-9a-fA-F]{6,}',
    ], 'all', '检测 SQL 编码绕过'),
    RegexRule('sqli_07', 'NoSQL 注入', 'high', [
        r'\b\$ne\b', r'\b\$gt\b', r'\b\$regex\b', r'\b\$where\b',
    ], 'all', '检测 NoSQL 操作符注入'),

    # ═══ XSS ═══
    RegexRule('xss_01', 'XSS script 标签', 'high', [
        r'<script\b[^>]*>.*?</script>', r'<script\b[^>]*>', r'</script>',
    ], 'all', '检测 script 标签注入'),
    RegexRule('xss_02', 'XSS 事件处理器', 'high', [
        r'onerror\s*=', r'onload\s*=', r'onclick\s*=', r'onfocus\s*=',
        r'onmouseover\s*=', r'onchange\s*=', r'<svg\b.*onload\s*=',
        r'<img\b[^>]*\bonerror\b', r'<input[^>]+onfocus',
        r'onblur\s*=', r'onscroll\s*=', r'<body[^>]+onload',
        r'<video[^>]+onerror', r'<audio[^>]+onerror',
        r'<marquee[^>]+onstart', r'<details[^>]+ontoggle',
    ], 'all', '检测 HTML 事件注入'),
    RegexRule('xss_03', 'XSS 伪协议', 'high', [
        r'javascript\s*:', r'<iframe[^>]*src', r'<meta[^>]+http-equiv',
        r'<form[^>]+action\s*=\s*"javascript',
        r'alert\s*\([^)]*\)', r'prompt\s*\([^)]*\)',
    ], 'all', '检测 XSS 伪协议/函数'),
    RegexRule('xss_04', 'XSS 编码绕过', 'medium', [
        r'&#\d{2,};', r'%3Cscript%3E', r'%3Csvg%3E',
        r'\\x3C', r'\\u003C',
        r'&lt;script', r'&#60;script',
    ], 'all', '检测 XSS 编码绕过'),

    # ═══ 路径遍历 / LFI ═══
    RegexRule('pt_01', '路径遍历 ..', 'high', [
        r'\.\./', r'\.\.\\', r'%2e%2e%2f', r'%2E%2E%2F',
        r'\.\.%5c', r'\.\.%255c',
        r'(\.\./|\.\.\\){2,}',
    ], 'all', '检测路径遍历'),
    RegexRule('pt_02', 'LFI 敏感文件', 'critical', [
        r'/etc/passwd', r'/etc/shadow', r'/etc/hosts', r'/proc/self/environ',
        r'c:\\windows\\win\.ini', r'boot\.ini',
        r'WEB-INF', r'META-INF',
    ], 'all', '检测敏感文件读取'),
    RegexRule('pt_03', 'LFI PHP wrapper', 'critical', [
        r'php://filter', r'php://input', r'data://.*base64',
        r'expect://', r'phar://',
    ], 'all', '检测 PHP wrapper LFI'),
    RegexRule('pt_04', 'LFI include 包含', 'high', [
        r'include\s*\(.*\.\./', r'require\s*\(.*\.\./',
    ], 'all', '检测文件包含函数'),

    # ═══ 命令注入 / RCE ═══
    RegexRule('cmdi_01', 'RCE 管道/重定向', 'critical', [
        r';\s*(id|whoami|cat|ls|dir|ping|nslookup|wget|curl|bash|sh|cmd|nc|nmap)',
        r'\|(ping|nslookup|id|whoami|cat|ls|dir)',
        r'`[^`]+`', r'\$\([^)]+\)',
    ], 'all', '检测命令注入'),
    RegexRule('cmdi_02', 'RCE 远程下载', 'critical', [
        r'\bwget\s+http', r'\bcurl\s+http',
        r'\bpython\s+-c\s+\"', r'\bperl\s+-e\s+\"',
        r'\bruby\s+-e\s+\"', r'\bphp\s+-r\s+\"',
        r'\bpowershell\s+-[ec]\s+',
    ], 'all', '检测远程命令执行'),
    RegexRule('cmdi_03', 'RCE 信息收集', 'high', [
        r'\bnetstat\s+-an', r'\bifconfig\b', r'\bip\s+addr',
        r'\bps\s+-aux', r'\btasklist\b',
    ], 'all', '检测主机信息收集'),

    # ═══ SSRF ═══
    RegexRule('ssrf_01', 'SSRF 内网/云元数据', 'critical', [
        r'http://127\.0\.0\.1', r'http://localhost', r'http://0\.0\.0\.0',
        r'http://169\.254\.', r'http://10\.\d+\.\d+\.\d+',
        r'http://172\.(1[6-9]|2\d|3[01])\.', r'http://192\.168\.',
        r'file:///', r'gopher://', r'dict://',
        r'metadata\.google\.internal', r'100\.100\.100\.200',
        r'instance-data',
    ], 'all', '检测 SSRF 内网/云服务探测'),
    RegexRule('ssrf_02', 'SSRF IP端口探测', 'medium', [
        r'https?://\d+\.\d+\.\d+\.\d+:\d+',
    ], 'all', '检测 SSRF IP+端口扫描'),

    # ═══ SSTI ═══
    RegexRule('ssti_01', 'SSTI 模板注入', 'critical', [
        r'\{\{.*\}\}', r'#\{.*\}', r'\$\{.*\}',
        r'\{\{.*\bconfig\b.*\}\}', r'\{\{.*\bself\b.*\}\}',
        r'\{\{.*__class__.*\}\}', r'\{\{.*__subclasses__.*\}\}',
        r'\$\{.*class\.forName.*\}',
    ], 'all', '检测 SSTI 模板注入'),

    # ═══ XXE ═══
    RegexRule('xxe_01', 'XXE XML外部实体', 'critical', [
        r'<!ENTITY\s+', r'<!DOCTYPE\s+', r'<!ELEMENT\s+',
        r'\bENTITY\b.*SYSTEM\b', r'xinclude',
    ], 'all', '检测 XXE XML 外部实体注入'),

    # ═══ 反序列化 ═══
    RegexRule('deser_01', '反序列化攻击', 'critical', [
        r'O:\d+:"[^"]+":\d+:{', r'__PHP_Incomplete_Class',
        r'pickle\.loads', r'pickle\.load\b', r'yaml\.load\s*\(',
    ], 'all', '检测 PHP/Python 反序列化'),

    # ═══ Log4j / JNDI ═══
    RegexRule('log4j_01', 'Log4j JNDI注入', 'critical', [
        r'\$\{jndi:', r'jndi:ldap://', r'jndi:rmi://', r'jndi:dns://',
    ], 'all', '检测 Log4j JNDI 注入'),

    # ═══ PHP 代码执行 ═══
    RegexRule('php_01', 'PHP 代码执行', 'critical', [
        r'base64_decode\s*\(', r'eval\s*\(.*\$', r'assert\s*\(.*\'',
        r'system\s*\(|exec\s*\(|passthru\s*\(',
    ], 'all', '检测 PHP 代码执行函数'),

    # ═══ 文件上传绕过 ═══
    RegexRule('up_01', '文件上传 双扩展名', 'high', [
        r'\.php\.(jpg|png|gif|jpeg)', r'\.asp\.(jpg|png)',
        r'\.php\d{0,2}\.', r'\.shtml\.', r'\.phtml\.',
    ], 'all', '检测文件上传双扩展名绕过'),

    # ═══ 开放重定向 ═══
    RegexRule('redirect_01', '开放重定向', 'medium', [
        r'redirect\s*=\s*https?://', r'url\s*=\s*https?://',
        r'next\s*=\s*https?://',
    ], 'all', '检测开放重定向参数'),

    # ═══ CRLF 注入 ═══
    RegexRule('crlf_01', 'CRLF 注入', 'high', [
        r'%0d%0a', r'%0D%0A', r'%0d%0a%0d%0a',
    ], 'all', '检测 CRLF 头部注入'),

    # ═══ 敏感路径/文件 ═══
    RegexRule('path_01', '敏感路径', 'medium', [
        r'/\.env', r'/admin\.php', r'/wp-admin, /post.php', r'/backup',
        r'/robots\.txt', r'/\.git', r'/\.svn',
        r'/phpinfo\.php', r'/server-status', r'/console',
        r'/actuator', r'/swagger', r'/api-docs', r'/graphql',
        r'/actuator/health', r'/actuator/env', r'/actuator/dump',
        r'/phpmyadmin', r'/manager/html',
        r'/debug', r'/trace',
        r'/config\.json', r'/config\.yaml', r'/database\.yml',
        r'/credentials\.json',
    ], 'path', '检测敏感路径扫描'),

    # ═══ 扫描器/爬虫 ═══
    RegexRule('scan_01', '扫描器识别', 'medium', [
        r'(nikto|nmap|sqlmap|acunetix|nessus|openvas|Burp\s*Suite|ZAP)',
        r'(masscan|zmap|unicornscan|grendel-scan|havij|appscan|netsparker)',
    ], 'all', '检测已知扫描器'),

    # ═══ 速率限制 ═══
    RateLimitRule('rate_01', '速率限制: 60秒300次', 'medium',
                  max_requests=300, window_seconds=60),
    RateLimitRule('rate_02', '速率限制: 10秒50次', 'high',
                  max_requests=50, window_seconds=10),

    # ═══ 头部检查 ═══
    HeaderRule('hdr_01', '缺少 User-Agent', 'low', 'missing_ua'),
]


# ═══════════════════════════════════════════════════════════════════
# 规则引擎
# ═══════════════════════════════════════════════════════════════════

class RuleEngine:
    """
    静态规则引擎。

    用法:
        engine = RuleEngine()
        result = engine.evaluate(method, path, qs, body, headers, ip)
        # → (ruleresult, [RuleMatch, ...])
    """

    def __init__(self, rules: List[Rule] = None):
        self.rules = rules or [r for r in DEFAULT_RULES]
        self._stats = Counter()

    def evaluate(self, method: str, path: str, query_string: str = '',
                 body: str = '', headers: dict = None,
                 source_ip: str = '') -> Tuple[int, List[RuleMatch]]:
        """
        执行所有已启用的规则。

        Returns:
            (result, matches)
            result: RULERESULT_ALLOW / BLOCK / UNCERTAIN
            matches: 命中的规则列表
        """
        headers = headers or {}
        matches = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            try:
                m = rule.match(method, path, query_string, body, headers, source_ip)
                if m is not None:
                    m.timestamp = time.time()
                    matches.append(m)
                    self._stats[rule.id] += 1

                    # 一旦有规则判定 BLOCK，立即返回
                    if m.result == RULERESULT_BLOCK:
                        return RULERESULT_BLOCK, matches

            except Exception as e:
                # 规则执行出错不计入，继续执行
                pass

        # 没有 BLOCK 判定 → 检查是否有 ALLOW 判定
        if any(m.result == RULERESULT_ALLOW for m in matches):
            return RULERESULT_ALLOW, matches

        # 有命中但不能确定（UNCERTAIN）或未命中 → 交给 ML
        return RULERESULT_UNCERTAIN, matches

    def get_rule_stats(self) -> dict:
        """获取规则命中统计。"""
        total = sum(self._stats.values())
        return {
            'total_hits': total,
            'by_rule': dict(self._stats.most_common(20)),
        }

    def reset_stats(self):
        self._stats.clear()


# 快捷入口
def quick_check(method, path, query='', body='', headers=None, ip=''):
    """快速规则检查，返回 (result_code, reason, matches)。"""
    engine = RuleEngine()
    result, matches = engine.evaluate(method, path, query, body, headers, ip)
    codes = {0: 'ALLOW', 1: 'BLOCK', 2: 'UNCERTAIN'}
    reasons = [m.detail for m in matches[:3]]
    return result, codes[result], reasons


if __name__ == '__main__':
    # 自测
    engine = RuleEngine()
    test_cases = [
        ('GET', '/search', "id=1' OR '1'='1", '', {}, '1.2.3.4'),
        ('GET', '/', '', '', {}, '1.2.3.4'),
        ('POST', '/comment', '', '<script>alert(1)</script>', {}, '1.2.3.4'),
        ('GET', '/admin.php', '', '', {}, '1.2.3.4'),
        ('GET', '/download', 'file=../../../etc/passwd', '', {}, '1.2.3.4'),
    ]
    for m, p, q, b, h, ip in test_cases:
        r, c, reasons = quick_check(m, p, q, b, h, ip)
        print(f'  {c:<10s} {m:<5s} {p:<20s} {"|".join(reasons[:2])}')
