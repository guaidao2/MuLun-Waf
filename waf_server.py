"""
waf_server.py — WAF 管理服务器。

提供:
  - Web 仪表盘 (http://localhost:5800)
  - REST API (/api/decide, /api/stats, /api/rules, /api/ips)
  - 静态规则引擎 + ML 模型级联
  - 实时请求日志 (SSE / Server-Sent Events)

启动:
    python waf_server.py
    # 打开 http://localhost:5800

生产部署:
    python waf_server.py --port 5800 --host 0.0.0.0 --model ./out_waf_3act/standalone_waf_final.pth
"""
import os, sys, json, time, queue, threading, datetime, socket
from typing import Dict, List, Optional
from collections import deque
import urllib.parse, urllib.request

from http.server import HTTPServer, BaseHTTPRequestHandler

import torch

# 确保能找到 same dir 的模块
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from flask import Flask, request, jsonify, render_template_string, Response

from waf_engine import WAFEngine, ACTION_NAMES, extract_features_from_request, inject_detection_signals, ip_in_network
from waf_rules import RuleEngine, RULERESULT_ALLOW, RULERESULT_BLOCK, RULERESULT_UNCERTAIN


# ─── 配置 ──────────────────────────────────────────────────────────

DEFAULT_MODEL = os.path.join(_here, 'out_waf_3act', 'standalone_waf_final.pth')

app = Flask(__name__)

# 全局实例
waf_engine: WAFEngine = None
rule_engine: RuleEngine = None

# 实时日志队列
log_queue = queue.Queue(maxsize=1000)
request_log = deque(maxlen=500)


@app.route('/')
def index():
    """Web 仪表盘首页。"""
    template_path = os.path.join(_here, 'templates', 'dashboard.html')
    if os.path.exists(template_path):
        with open(template_path, encoding='utf-8') as f:
            return render_template_string(f.read())
    return '<h1>Mulun WAF</h1><p>模板文件未找到</p>'


# ── API ──────────────────────────────────────────────────────────

@app.route('/api/stats')
def api_stats():
    """获取统计数据。"""
    stats = waf_engine.get_stats() if waf_engine else {}
    rule_stats = rule_engine.get_rule_stats() if rule_engine else {}
    # 计算时间序列（用于图表）
    now = time.time()
    timeline = {'allow': 0, 'monitor': 0, 'block': 0, 'total': 0}
    for entry in request_log:
        ts = entry.get('timestamp', 0)
        if now - ts < 300:  # 过去 5 分钟
            act = entry.get('action', '')
            if act == 'ALLOW': timeline['allow'] += 1
            elif act == 'MONITOR': timeline['monitor'] += 1
            elif act == 'BLOCK': timeline['block'] += 1
            timeline['total'] += 1

    return jsonify({
        'stats': stats,
        'rule_stats': rule_stats,
        'timeline': timeline,
        'recent': list(request_log)[-50:],
        'action_names': {
            '0': 'ALLOW', '1': 'MONITOR', '2': 'BLOCK',
        },
    })


@app.route('/api/decide', methods=['POST'])
def api_decide():
    """模拟一个请求的 WAF 决策（供前端测试用）。"""
    data = request.get_json(silent=True) or {}
    method = data.get('method', 'GET')
    path = data.get('path', '/')
    query_string = data.get('query', '')
    body = data.get('body', '')
    headers = data.get('headers', {})
    source_ip = data.get('ip', f'10.0.0.{hash(path) % 255}')
    detections = data.get('detections', {})

    result = run_waf(method, path, query_string, body, headers, source_ip, detections)
    return jsonify(result)


@app.route('/api/ips', methods=['GET', 'POST', 'DELETE'])
def api_ips():
    """IP 管理。"""
    if request.method == 'GET':
        return jsonify({
            'allowlist': waf_engine.allowlist if waf_engine else [],
            'blocklist': waf_engine.blocklist if waf_engine else [],
        })
    data = request.get_json(silent=True) or {}
    cidr = data.get('cidr', '')
    list_type = data.get('type', 'allow')  # allow / block
    action = data.get('action', 'add')     # add / remove

    if not waf_engine:
        return jsonify({'error': '引擎未初始化'}), 500

    if action == 'add':
        if list_type == 'allow':
            waf_engine.allowlist_add(cidr)
        else:
            waf_engine.blocklist_add(cidr)
    elif action == 'remove':
        if list_type == 'allow':
            waf_engine.allowlist_remove(cidr)
        else:
            waf_engine.blocklist_remove(cidr)

    return jsonify({'ok': True})


@app.route('/api/rules', methods=['GET', 'POST'])
def api_rules():
    """规则管理。"""
    if request.method == 'GET':
        rules_data = []
        for r in rule_engine.rules:
            rules_data.append({
                'id': r.id,
                'name': r.name,
                'severity': r.severity,
                'enabled': r.enabled,
                'description': r.description,
            })
        return jsonify({'rules': rules_data})

    data = request.get_json(silent=True) or {}
    rule_id = data.get('id', '')
    enabled = data.get('enabled', True)

    for r in rule_engine.rules:
        if r.id == rule_id:
            r.enabled = enabled
            return jsonify({'ok': True, 'id': rule_id, 'enabled': enabled})

    return jsonify({'error': f'规则 {rule_id} 未找到'}), 404


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """重置统计和 IP 上下文。"""
    if waf_engine:
        waf_engine.reset_context()
    if rule_engine:
        rule_engine.reset_stats()
    request_log.clear()
    return jsonify({'ok': True})


# ── 核心决策流 ──────────────────────────────────────────────────

def run_waf(method: str, path: str, query_string: str = '',
            body: str = '', headers: dict = None, source_ip: str = '',
            detections: dict = None) -> dict:
    """
    级联决策:
      L1 静态规则 (快)
      L2 WAF 引擎 (ML)
    """
    headers = headers or {}
    detections = detections or {}

    if not waf_engine:
        return {'error': '引擎未初始化'}

    start = time.time()

    # ── 特征提取 ──
    feats = extract_features_from_request(method, path,
        dict(urllib.parse.parse_qsl(query_string)), body, headers, source_ip)

    # L0: IP 临时封禁检查
    if waf_engine:
        ctx = waf_engine._get_context(source_ip)
        if ctx.blocked_until > time.time() and not _is_whitelisted(source_ip):
            remaining = ctx.blocked_until - time.time()
            entry = {"action": "BLOCK", "source": "rule", "confidence": 0.95,
                     "reason": f"IP 已被临时封禁 (剩余 {remaining:.0f}s)",
                     "method": method, "path": path, "ip": source_ip, "time_ms": 0}
            request_log.appendleft(entry)
            _push_log(entry)
            return entry

    # ═══ L1: 静态规则引擎 ═══
    rule_result, rule_matches = rule_engine.evaluate(
        method, path, query_string, body, headers, source_ip
    )

    if rule_result == RULERESULT_BLOCK:
        reasons = [m.detail for m in rule_matches]
        atk_type = rule_matches[0].rule_id.split('_')[0] if rule_matches else ''

        # 白名单 IP：只记录数据不拦截
        if _is_whitelisted(source_ip):
            _save_training_sample(feats, method, path, source_ip, 2, 'rule', atk_type)
        else:
            # 正常拦截
            entry = {
                'action': 'BLOCK',
                'source': 'rule',
                'reason': ' | '.join(reasons[:2]),
                'method': method, 'path': path, 'ip': source_ip,
                'time_ms': round((time.time() - start) * 1000, 2),
                'timestamp': time.time(),
            }
            request_log.appendleft(entry)
            _push_log(entry)
            if waf_engine:
                waf_engine.temp_block(source_ip, 300)
            _save_training_sample(feats, method, path, source_ip, 2, 'rule', atk_type)
            return {'action': 'BLOCK', 'confidence': 1.0, 'reason': entry['reason'],
                    'method': method, 'path': path, 'ip': source_ip,
                    'source': 'rule', 'time_ms': entry['time_ms']}

    if rule_result == RULERESULT_ALLOW:
        # 规则确认安全
        entry = {
            'action': 'ALLOW', 'source': 'rule',
            'reason': '规则确认安全', 'method': method, 'path': path,
            'ip': source_ip, 'time_ms': round((time.time() - start) * 1000, 2),
            'timestamp': time.time(),
        }
        request_log.appendleft(entry)
        _push_log(entry)
        return {'action': 'ALLOW', 'confidence': 0.95, 'reason': '规则确认安全',
                'method': method, 'path': path, 'ip': source_ip,
                'source': 'rule', 'time_ms': entry['time_ms']}

    # ═══ L1.5: 特征异常评分（不依赖路径规则） ═══
    anomaly_score = 0
    # [0-7]: 攻击类型信号
    for i in range(8):
        if feats[i] > 0.3:
            anomaly_score += 2
    # [9]: 特殊字符比率
    if feats[9] > 0.2: anomaly_score += 2
    if feats[9] > 0.4: anomaly_score += 3
    # [11]: 参数数量异常
    if feats[11] > 0.3: anomaly_score += 2
    # [13]: 缺少 UA
    if feats[13] > 0.3: anomaly_score += 1
    # [14]: 缺少 Cookie
    if feats[14] > 0.5: anomaly_score += 1
    # 高异常分 → 直接拦截
    if anomaly_score >= 5:
        entry = {
            'action': 'BLOCK', 'source': 'rule',
            'reason': f'请求特征异常 (评分 {anomaly_score})',
            'method': method, 'path': path, 'ip': source_ip,
            'time_ms': round((time.time() - start) * 1000, 2), 'timestamp': time.time(),
        }
        request_log.appendleft(entry)
        _push_log(entry)
        if waf_engine: waf_engine.temp_block(source_ip, 300)
        return entry

    # ═══ L2: ML 模型决策 ═══
    if detections:
        feats = inject_detection_signals(feats, detections)

    ml_result = waf_engine.decide(
        source_ip=source_ip,
        features=feats,
        detections=detections,
    )

    # ML BLOCK -> 封禁 + 收集训练数据
    atk_from_reason = ''
    reason_lower = ml_result.get('reason', '').lower()
    for kw, at in [('sqli','sqli'),('xss','xss'),('path','traversal'),('lfi','traversal'),
                   ('cmdi','cmdi'),('rce','cmdi'),('ssrf','ssrf'),('ssti','other'),
                   ('xxe','other'),('log4j','other'),('jndi','other'),('scan','scanner')]:
        if kw in reason_lower:
            atk_from_reason = at
            break
    # 白名单 IP 的训练标签强制为 ALLOW (模型输出可能偏保守)
    train_act = ml_result['action']
    if _is_whitelisted(source_ip):
        train_act = 0

    if ml_result['action'] == 'BLOCK' and waf_engine:
        waf_engine.temp_block(source_ip, 600)
        _save_training_sample(feats, method, path, source_ip, 2, 'ml', atk_from_reason)
    elif ml_result['action'] == 'MONITOR':
        raw_act = ml_result.get('raw_action', 1)
        _save_training_sample(feats, method, path, source_ip, train_act, 'ml')
    else:
        raw_act = ml_result.get('raw_action', 0)
        _save_training_sample(feats, method, path, source_ip, train_act, 'ml')

    entry = {
        'action': ml_result['action'],
        'source': 'ml',
        'reason': ml_result.get('reason', ''),
        'confidence': ml_result['confidence'],
        'uncertainty': ml_result['uncertainty'],
        'method': method, 'path': path, 'ip': source_ip,
        'time_ms': ml_result.get('total_time_ms', 0),
        'timestamp': time.time(),
    }
    if ml_result['action'] == 'BLOCK' and waf_engine:
        waf_engine.temp_block(source_ip, 600)
    request_log.appendleft(entry)
    _push_log(entry)

    ml_result['method'] = method
    ml_result['path'] = path
    ml_result['source'] = 'ml'
    ml_result['time_ms'] = entry['time_ms']
    return ml_result


# ── SSE 实时推送 ────────────────────────────────────────────────

def _is_whitelisted(ip):
    """检查 IP 是否在白名单中。"""
    if not waf_engine:
        return False
    for cidr in waf_engine.allowlist:
        if ip_in_network(ip, cidr):
            return True
    return False


def _save_training_sample(feats, method, path, ip, action, source, attack_type=None):
    """保存一条决策样本到学习缓存。"""
    try:
        buf_path = os.path.join(os.path.dirname(__file__), 'data', 'collected', 'buffer.jsonl')
        os.makedirs(os.path.dirname(buf_path), exist_ok=True)
        features = list(feats)[:32]
        # 注入攻击类型信号到特征 [0-7]
        if attack_type:
            atk_map = {'sqli':0, 'xss':1, 'traversal':2, 'pt':2, 'path':2, 'lfi':2,
                       'cmdi':3, 'cmd':3, 'php':3, 'rce':3, 'deser':3,
                       'ssrf':4, 'xxe':4, 'log4j':4,
                       'upload':5, 'up':5, 'redirect':5, 'crlf':5,
                       'brute':6, 'rate':6,
                       'scan':7, 'scanner':7, 'hdr':1, 'ssti':0}
            idx = atk_map.get(attack_type.lower(), 0)
            if 0 <= idx < 8:
                features[idx] = 0.9
        sample = {
            '_id': f'{ip}_{time.time()}',
            'features': features,
            'action': action,
            'strategy': 0 if action == 2 else 1,
            'value': 0.9 if action == 2 else 0.3,
            'source': source,
            'attack_type': attack_type or '',
            'timestamp': time.time(),
            'ip': ip, 'method': method, 'path': path,
        }
        with open(buf_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    except Exception:
        pass


def _push_log(entry: dict):
    """推送日志到 SSE 队列。"""
    try:
        log_queue.put_nowait(entry)
    except queue.Full:
        pass


@app.route('/api/events')
def sse_events():
    """SSE (Server-Sent Events) 实时日志流。"""
    def generate():
        while True:
            try:
                entry = log_queue.get(timeout=30)
                yield f'data: {json.dumps(entry)}\n\n'
            except queue.Empty:
                yield f'data: {{"keepalive": true}}\n\n'
    return Response(generate(), mimetype='text/event-stream')


# ── 启动 ────────────────────────────────────────────────────────

def start(host='127.0.0.1', port=5800, model_path=DEFAULT_MODEL,
          proxy_enabled=False, proxy_port=8080, backend=''):
    """启动 WAF 服务器。"""
    global waf_engine, rule_engine

    print(f'[WAF] 初始化...')
    print(f'[WAF] 模型: {model_path}')

    if not os.path.exists(model_path):
        print(f'[WAF] !!! 模型文件不存在: {model_path}')
        print(f'[WAF]     请先训练模型: python standalone_waf.py --data-path ...')
        return

    # 初始化引擎
    waf_engine = WAFEngine(model_path=model_path)
    rule_engine = RuleEngine()

    print(f'[WAF] 规则加载: {len(rule_engine.rules)} 条')
    print(f'[WAF] 管理面板: http://{host}:{port}')

    # 启动反向代理（在 Flask 之前，独立线程）
    proxy_server = None
    if proxy_enabled:
        proxy_server = ProxyServer('0.0.0.0', proxy_port, backend)
        proxy_server.start()
        print(f'[WAF] 反向代理: http://0.0.0.0:{proxy_port} -> {backend}')

    print(f'[WAF] Ctrl+C 停止')
    print()

    app.run(host=host, port=port, debug=False, threaded=True)


# ═══════════════════════════════════════════════════════════════════
# 反向代理模式
# ═══════════════════════════════════════════════════════════════════

class WAFProxyHandler(BaseHTTPRequestHandler):
    """
    HTTP 反向代理处理器。

    工作流程:
      1. 接收客户端请求
      2. 提取特征 → 规则引擎 → ML 模型
      3. 如果 BLOCK → 返回 403
      4. 如果 ALLOW → 转发到后端服务器 → 返回后端响应
    """

    # 类级共享：由 ProxyServer 设置
    backend_url = 'http://localhost:3000'
    waf_engine_ref = None
    rule_engine_ref = None

    def do_GET(self):
        self._proxy_request('GET')

    def do_POST(self):
        self._proxy_request('POST')

    def do_PUT(self):
        self._proxy_request('PUT')

    def do_DELETE(self):
        self._proxy_request('DELETE')

    def do_PATCH(self):
        self._proxy_request('PATCH')

    def do_OPTIONS(self):
        self._proxy_request('OPTIONS')

    def _proxy_request(self, method):
        start = time.time()

        # 读取请求体
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len).decode('utf-8', errors='replace') if content_len > 0 else ''

        source_ip = self.client_address[0]
        path = urllib.parse.urlparse(self.path).path
        query_string = urllib.parse.urlparse(self.path).query

        # 提取请求头
        headers = {k.lower(): v for k, v in self.headers.items()}

        # ── 指定检测信号（可从前端透传） ──
        detections_str = headers.get('x-waf-detections', '')
        detections = {}
        if detections_str:
            try:
                detections = json.loads(detections_str)
            except:
                pass

        # ── 检查 IP 是否已被临时封禁 ──
        if self.waf_engine_ref:
            ctx = self.waf_engine_ref._get_context(source_ip)
            if ctx.blocked_until > time.time() and not _is_whitelisted(source_ip):
                remaining = ctx.blocked_until - time.time()
                _log_entry('BLOCK', 'rule', f'IP 临时封禁中 ({remaining:.0f}s 剩余)',
                           method, path, source_ip, 0)
                self._send_block(f'IP 已被临时封禁 (剩余 {remaining:.0f}s)')
                return

        # ── 执行 WAF 决策 ──
        feats = extract_features_from_request(method, path,
            dict(urllib.parse.parse_qsl(query_string)), body, headers, source_ip)

        eng = self.waf_engine_ref
        rul = self.rule_engine_ref

        if eng and rul:
            # L1 规则引擎
            rule_result, matches = rul.evaluate(
                method, path, query_string, body, headers, source_ip
            )
            if rule_result == RULERESULT_BLOCK:
                # 规则拦截
                reasons = [m.detail for m in matches]
                if eng:
                    eng.temp_block(source_ip, 300)
                _log_entry('BLOCK', 'rule', ' | '.join(reasons[:2]),
                           method, path, source_ip, (time.time()-start)*1000)
                self._send_block(reasons[0] if reasons else '规则拦截')
                return

            if rule_result == RULERESULT_ALLOW:
                _log_entry('ALLOW', 'rule', '规则确认安全',
                           method, path, source_ip, (time.time()-start)*1000)
                # 规则确认安全，转发
                self._forward(method, body)
                return

        # L2 ML 模型
        if eng:
            result = eng.decide(source_ip, feats, detections)
            _log_entry(result['action'], 'ml', result.get('reason', ''),
                       method, path, source_ip, result.get('total_time_ms', 0))

            if result['action'] == 'BLOCK':
                if eng:
                    eng.temp_block(source_ip, 600)
                _log_entry(result['action'], 'ml', result.get('reason', ''),
                           method, path, source_ip, result.get('total_time_ms', 0))
                if not _is_whitelisted(source_ip):
                    self._send_block(result.get('reason', 'ML 模型拦截'))
                    return
            else:
                _log_entry(result['action'], 'ml', result.get('reason', ''),
                           method, path, source_ip, result.get('total_time_ms', 0))

        # ALLOW / MONITOR → 转发
        self._forward(method, body)

    def _forward(self, method, body):
        """转发请求到后端服务器。"""
        target = self.backend_url.rstrip('/') + self.path

        try:
            # 构建转发请求
            data = body.encode('utf-8') if body else None
            req = urllib.request.Request(
                target, data=data, method=method,
                headers={k: v for k, v in self.headers.items()
                         if k.lower() not in ('host', 'x-waf-detections', 'content-encoding')},
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                # 返回后端响应给客户端
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ('transfer-encoding', 'content-encoding', 'content-length'):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp.read())

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(f'502 Bad Gateway: {e}'.encode('utf-8'))

    def _send_block(self, reason):
        """返回 403 拦截页面。"""
        html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>请求被拦截</title>
<style>
body{{font-family:sans-serif;background:#0f172a;color:#e2e8f0;display:flex;
justify-content:center;align-items:center;height:100vh;margin:0}}
.card{{background:#1e293b;padding:40px;border-radius:12px;text-align:center;
border:1px solid #334155;max-width:500px}}
h1{{color:#ef4444;font-size:24px;margin-bottom:16px}}
p{{color:#94a3b8;font-size:14px}}
code{{color:#38bdf8}}
</style></head><body>
<div class="card">
<h1>🛡 请求被拦截</h1>
<p>Mulun WAF 已拦截此请求</p>
<p><code>{reason}</code></p>
</div></body></html>'''
        self.send_response(403)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('X-WAF-Status', 'BLOCKED')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def log_message(self, format, *args):
        pass  # 安静模式


class ProxyServer:
    """反向代理服务器。"""

    def __init__(self, listen_host='0.0.0.0', listen_port=8080,
                 backend='http://localhost:3000'):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.backend = backend
        self.server = None
        self.thread = None

    def start(self):
        """在后台线程启动代理服务器。"""
        WAFProxyHandler.backend_url = self.backend
        WAFProxyHandler.waf_engine_ref = waf_engine
        WAFProxyHandler.rule_engine_ref = rule_engine

        self.server = HTTPServer((self.listen_host, self.listen_port), WAFProxyHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f'[WAF] 反向代理: http://{self.listen_host}:{self.listen_port} → {self.backend}')

    def stop(self):
        if self.server:
            self.server.shutdown()


def _log_entry(action, source, reason, method, path, ip, time_ms):
    """记录日志条目（供 proxy handler 调用）。"""
    entry = {
        'action': action, 'source': source, 'reason': reason,
        'method': method, 'path': path, 'ip': ip,
        'time_ms': round(time_ms, 2), 'timestamp': time.time(),
    }
    request_log.appendleft(entry)
    try:
        log_queue.put_nowait(entry)
    except queue.Full:
        pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mulun WAF 服务器')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='管理面板监听地址')
    parser.add_argument('--port', type=int, default=5800,
                        help='管理面板监听端口')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='模型权重路径')
    parser.add_argument('--proxy', action='store_true',
                        help='启用反向代理模式')
    parser.add_argument('--listen', type=int, default=8080,
                        help='反向代理监听端口 (默认 8080)')
    parser.add_argument('--backend', type=str, default='',
                        help='后端服务地址 (如 http://localhost:3000)')
    args = parser.parse_args()

    backend = args.backend
    if args.proxy and not backend:
        print('[WAF] !!! 反向代理模式需要指定 --backend')
        print('[WAF]     例: --proxy --listen 8080 --backend http://localhost:3000')
        sys.exit(1)

    start(args.host, args.port, args.model, args.proxy, args.listen, backend)
