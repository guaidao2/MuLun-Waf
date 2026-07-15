"""
generate_normal.py — 生成正常流量训练数据。

在跑完攻击脚本后运行，补充 ALLOW 样本。
用法:
    python scripts/generate_normal.py --count 500
"""
import urllib.request, json, random, sys

TARGET = 'http://localhost:5800/api/decide'

PATHS = [
    '/', '/api/user/profile', '/api/products', '/api/product/1',
    '/api/product/2', '/api/product/3', '/api/search?q=phone',
    '/api/search?q=laptop', '/api/cart', '/api/checkout',
    '/api/order', '/static/styles.css', '/static/app.js',
    '/static/logo.png', '/about', '/contact', '/help',
    '/api/user/login', '/api/user/logout', '/api/categories',
    '/api/recommendations', '/api/product/4', '/api/product/5',
]
METHODS = ['GET'] * 8 + ['POST'] * 2

def send(method, path, body=''):
    payload = {
        'method': method,
        'path': path,
        'body': body,
        'ip': '127.0.0.1',
        'headers': {'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'},
    }
    try:
        req = urllib.request.Request(TARGET,
            data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=10).read()
        return True
    except:
        return False

if __name__ == '__main__':
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    ok = 0
    for i in range(count):
        method = random.choice(METHODS)
        path = random.choice(PATHS)
        body = 'username=user&password=pass' if method == 'POST' and random.random() < 0.3 else ''
        if send(method, path, body):
            ok += 1
        if (i+1) % 100 == 0:
            print(f'  [{i+1}/{count}] {ok} OK')

    print(f'\n完成: {ok}/{count} 条正常请求已发送')
    print('用以下命令检查: python -c "import json; from collections import Counter;'
          ' open(\\\"data/collected/buffer.jsonl\\\") as f: samples=[json.loads(l) for l in f if l.strip()];'
          ' print(Counter(s[\\\"action\\\"] for s in samples))"')
