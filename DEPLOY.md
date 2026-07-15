# Mulun WAF 部署指南

轻量 WAF 引擎，基于 GameNN 侧枝决策架构。
纯 CPU 运行，单次决策 ~10μs，权重 264KB。

---

## 快速开始

### 方式一: 管理面板（推荐）

```bash
# 启动（默认 localhost:5800）
python waf_server.py
# 打开 http://localhost:5800
```

管理面板提供: 实时日志 · 统计 · IP黑白名单 · 规则管理 · 请求测试控制台

### 方式二: 反向代理（像雷池 WAF 一样用）

```bash
# 启动反向代理，挡在真实服务前面
python waf_server.py --proxy --listen 8080 --backend http://localhost:3000
#                      ↑  WAF监听端口    ↑ 你的真实服务地址

# 浏览器访问 http://localhost:8080
# 所有流量先经过 WAF 检查，再转发到 localhost:3000
# 管理面板仍在 http://localhost:5800
```

**架构：**
```
用户 → http://your-site.com:8080
       ↓
   WAF 反向代理 (端口 8080)
       ├─ 规则拦截 → 403 Forbidden
       ├─ ML 拦截  → 403 Forbidden  
       └─ 放行     → 转发到后端 (localhost:3000)
                        ↓
                   你的真实 Web 服务
```

**Nginx 集成：**
```nginx
# 如果你已经有 Nginx，把 WAF 放在 Nginx 后面
upstream waf_backend {
    server 127.0.0.1:8080;  # WAF 的 --listen 端口
}
server {
    listen 80;
    location / {
        proxy_pass http://waf_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 方式三: API 集成（在你的代码里调）

```python
import requests
r = requests.post('http://localhost:5800/api/decide', json={
    'method': 'GET', 'path': '/api/login',
    'query': "id=1' OR '1'='1",
    'ip': request.remote_addr,
}).json()
if r['action'] == 'BLOCK':
    return '403 Forbidden', 403
```

---

## 架构

```
请求 → L1 规则引擎 (18条静态规则) 
       ├─ 明确攻击(BLOCK) → 直接拦截 (不经过 ML)
       ├─ 明确安全(ALLOW) → 直接放行 (不经过 ML)  
       └─ 不确定(UNCERTAIN) → L2 ML 决策
                              ├─ WAFEngine (64K 参数)
                              ├─ 置信度门控降级
                              └─ 不确定性高时保守放行
```

---

## API 文档

| 端点 | 方法 | 说明 |
|:----|:----|:------|
| `/` | GET | Web 管理面板 |
| `/api/stats` | GET | 统计与实时日志 |
| `/api/decide` | POST | 执行 WAF 决策 |
| `/api/ips` | GET/POST | IP 黑白名单管理 |
| `/api/rules` | GET/POST | 规则启用/禁用 |
| `/api/reset` | POST | 重置统计 |
| `/api/events` | GET | SSE 实时日志流 |

### `/api/decide` 请求示例

```json
{
  "method": "POST",
  "path": "/api/upload",
  "query": "",
  "body": "filename=shell.php",
  "ip": "203.0.113.5",
  "headers": {"User-Agent": "curl/7.68"},
  "detections": {"webshell": 0.85}
}
```

### `/api/decide` 响应示例

```json
{
  "action": "BLOCK",
  "confidence": 1.0,
  "reason": "规则匹配 [文件上传双扩展名] path 命中模式 #0",
  "source": "rule",
  "method": "POST",
  "path": "/api/upload",
  "ip": "203.0.113.5",
  "time_ms": 0.12
}
```

---

## 生产部署

### 方式一: 直接运行

```bash
python waf_server.py --port 5800 --host 0.0.0.0 --model ./out_waf_3act/standalone_waf_final.pth
```

### 方式二: Systemd 服务

```ini
[Unit]
Description=Mulun WAF
After=network.target

[Service]
Type=simple
User=waf
WorkingDirectory=/opt/mulun
ExecStart=/usr/bin/python3 /opt/mulun/waf_server.py --port 5800 --host 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 方式三: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5800
CMD ["python", "waf_server.py", "--host", "0.0.0.0"]
```

---

## 文件说明

| 文件 | 说明 |
|:----|:------|
| `standalone_waf.py` | 核心模型定义 (65K 参数) |
| `waf_engine.py` | WAF 引擎 (RNN 状态跟踪 + 置信度门控) |
| `waf_rules.py` | 静态规则引擎 (18 条规则) |
| `waf_server.py` | Flask 管理服务器 |
| `templates/dashboard.html` | Web 管理面板 |
| `scripts/waf_simulator.py` | 训练数据生成器 |
| `out_waf_3act/standalone_waf_final.pth` | 训练好的模型权重 |

---

## 性能指标

| 场景 | 延迟 | 说明 |
|:----|:----|:------|
| 规则引擎 (L1) | ~1-5μs | 正则匹配，无模型调用 |
| ML 模型 (L2) | ~10-30μs | 65K 参数推理 |
| 完整决策 | ~15-50μs | L1 + L2 |
| 吞吐量 (单核) | ~20,000 req/s | 纯 CPU |

---

## 自定义模型训练

```bash
# 1. 生成训练数据
python scripts/waf_simulator.py --episodes 1000 --output ./data

# 2. 训练新模型
python standalone_waf.py --data-path ./data/waf_trajectories_1000ep.jsonl \
  --epochs 80 --save-dir ./my_model

# 3. 使用新模型启动
python waf_server.py --model ./my_model/standalone_waf_final.pth
```
