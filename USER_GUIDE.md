# Mulun-WAF 使用说明书

玄幕安全团队 — guaidao2

---

## 目录

1. [快速开始](#1-快速开始)
2. [架构说明](#2-架构说明)
3. [管理面板](#3-管理面板)
4. [API 文档](#4-api-文档)
5. [反向代理部署](#5-反向代理部署)
6. [规则引擎](#6-规则引擎)
7. [模型训练与进化](#7-模型训练与进化)
8. [数据收集与主动学习](#8-数据收集与主动学习)
9. [生产部署](#9-生产部署)
10. [常见问题](#10-常见问题)

---

## 1. 快速开始

### 环境要求

- Python 3.10+
- pip 依赖：`pip install torch flask numpy tqdm`

### 启动管理面板

```bash
cd mulun-waf
python waf_server.py
# 浏览器打开 http://localhost:5800
```

管理面板提供：实时请求日志、统计卡片、IP 黑白名单管理、规则开关、请求测试控制台。

### 测试 WAF 决策

```bash
# 测试 SQL 注入拦截
curl -X POST http://localhost:5800/api/decide \
  -H "Content-Type: application/json" \
  -d '{"method":"GET","path":"/api/login","query":"id=1%27%20OR%20%271%27%3D%271","ip":"10.0.0.1"}'

# 测试正常请求
curl -X POST http://localhost:5800/api/decide \
  -H "Content-Type: application/json" \
  -d '{"method":"GET","path":"/api/products","ip":"10.0.0.1"}'
```

---

## 2. 架构说明

### 级联检测架构

```
请求 → L0: IP 管理（白名单/黑名单/临时封禁）
       ↓
       L1: 静态规则引擎（33 条规则，1-5μs）
       ├─ 匹配 → 403 + 记录数据
       └─ 未匹配 → 进入 L2
              ↓
       L2: 侧枝 ML 模型（65K 参数，10-30μs）
       ├─ BLOCK  → 403 + 临时封禁 + 记录数据
       ├─ MONITOR → 放行 + 记录数据
       └─ ALLOW  → 放行 + 记录数据
              ↓
       L3: 置信度门控（防止误判）
       └─ 不确定时自动降级
```

### 文件说明

| 文件 | 说明 |
|:----|:------|
| `standalone_waf.py` | 核心模型（65K 参数、3 动作） |
| `waf_engine.py` | WAF 引擎（IP 管理、置信度门控、白名单训练） |
| `waf_rules.py` | 33 条静态规则 |
| `waf_server.py` | Flask 管理服务器 + 反向代理 |
| `scripts/waf_simulator.py` | 训练数据生成器 |
| `scripts/payload_generator.py` | 攻击 payload 批量生成器 |
| `scripts/generate_normal.py` | 正常流量生成器 |
| `scripts/collect_and_retrain.py` | 主动学习循环 |
| `templates/dashboard.html` | Web 管理面板 |
| `out_waf_v6/standalone_waf_final.pth` | 训练好的模型权重（264KB） |

---

## 3. 管理面板

启动后访问 `http://localhost:5800`。

### 统计卡片

面板顶部显示实时统计：
- **放行**：ALLOW 决策总数
- **观察**：MONITOR 决策总数（可疑但未拦截）
- **封禁**：BLOCK 决策总数
- **ML 决策**：由 ML 模型处理的请求数
- **规则拦截**：由规则引擎拦截的请求数

### 实时请求日志

显示最近 500 条请求的决策记录，包含：动作、来源（规则/ML）、方法、路径、IP、原因、耗时。

### 请求测试

底部的测试控制台可以手动构造 HTTP 请求测试 WAF 决策：

```
方法: GET/POST/PUT/DELETE
路径: /api/login
查询参数: id=1' OR '1'='1
请求体: <script>alert(1)</script>
源 IP: 203.0.113.5
检测引擎信号: {"sqli": 0.9}
```

点击发送即返回 WAF 决策结果。

### IP 管理

- **白名单**：CIDR 格式，白名单 IP 的请求经过完整检测但不会被拦截，数据被记录用于训练
- **黑名单**：CIDR 格式，黑名单 IP 的请求直接 403

### 规则管理

每条规则可单独启用/禁用，包含规则名称、严重级别、描述。

---

## 4. API 文档

### `/api/decide` — 执行 WAF 决策

**请求：**
```json
{
  "method": "GET",
  "path": "/api/login",
  "query": "id=1' OR '1'='1",
  "body": "",
  "ip": "203.0.113.5",
  "headers": {"User-Agent": "curl/7.68"},
  "detections": {"sqli": 0.9}
}
```

**响应（规则拦截）：**
```json
{
  "action": "BLOCK",
  "confidence": 1.0,
  "source": "rule",
  "reason": "规则匹配 [SQLi 布尔盲注] query 命中模式 #4",
  "method": "GET",
  "path": "/api/login",
  "ip": "203.0.113.5",
  "time_ms": 0.12
}
```

**响应（ML 决策）：**
```json
{
  "action": "MONITOR",
  "confidence": 0.55,
  "uncertainty": 0.47,
  "source": "ml",
  "reason": "ALLOW 置信度不足 (0.47<0.55)，升级 MONITOR",
  "raw_action": 0,
  "is_whitelisted": false,
  "method": "GET",
  "path": "/",
  "ip": "10.0.0.1",
  "time_ms": 15.3
}
```

### `/api/stats` — 获取统计

返回实时统计和最近请求日志。

### `/api/ips` — IP 管理

```bash
# 查看列表
GET /api/ips

# 添加白名单
POST /api/ips
{"cidr": "192.168.1.0/24", "type": "allow", "action": "add"}

# 删除白名单
POST /api/ips
{"cidr": "192.168.1.0/24", "type": "allow", "action": "remove"}
```

### `/api/rules` — 规则管理

```bash
# 查看规则列表
GET /api/rules

# 启用/禁用规则
POST /api/rules
{"id": "xss_01", "enabled": false}
```

### `/api/events` — SSE 实时日志流

```javascript
const evtSource = new EventSource('/api/events');
evtSource.onmessage = (event) => {
  console.log(JSON.parse(event.data));
};
```

---

## 5. 反向代理部署

### 标准模式

```bash
python waf_server.py --proxy --listen 8080 --backend http://localhost:3000
```

用户 → WAF :8080 → 后端 :3000

### 与 Nginx 集成

```nginx
upstream waf_upstream {
    server 127.0.0.1:8080;  # WAF 监听端口
}

server {
    listen 80;
    
    location / {
        proxy_pass http://waf_upstream;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### 自定义端口

```bash
# 管理面板 :5800，代理 :8888
python waf_server.py --port 5800 --proxy --listen 8888 --backend http://localhost:8000
```

---

## 6. 规则引擎

Mulun-WAF 内置 33 条静态规则，覆盖 OWASP Top 10 主要攻击类型。

### 规则分类

| 类别 | 数量 | 说明 |
|:----|:----:|:------|
| SQL 注入 | 7 | UNION SELECT、布尔盲注、时间盲注、系统表查询、DML/DDL、编码绕过、NoSQL |
| XSS | 4 | Script 标签、事件处理器、伪协议、编码绕过 |
| 路径遍历/LFI | 4 | 路径遍历、敏感文件读取、PHP Wrapper、文件包含 |
| 命令注入/RCE | 3 | 管道重定向、远程下载执行、信息收集 |
| SSRF | 2 | 内网地址探测、云元数据访问 |
| SSTI | 1 | 模板引擎注入 |
| XXE | 1 | XML 外部实体注入 |
| 反序列化 | 1 | PHP/Python 反序列化 |
| Log4j/JNDI | 1 | JNDI 注入 |
| PHP 代码执行 | 1 | 危险函数调用 |
| 文件上传 | 1 | 双扩展名绕过 |
| 开放重定向 | 1 | URL 跳转参数 |
| CRLF 注入 | 1 | 头部注入 |
| 敏感路径扫描 | 1 | 20+ 种敏感路径 |
| 扫描器检测 | 1 | 已知扫描器指纹 |
| 速率限制 | 2 | 60 秒 60 次 / 10 秒 15 次 |
| 头部检查 | 1 | User-Agent 缺失 |

### 自定义规则

规则在 `waf_rules.py` 的 `DEFAULT_RULES` 列表中定义，格式：

```python
RegexRule(
    id='规则ID',      # 如 'sqli_01'
    name='规则名称',    # 如 'SQLi UNION/SELECT'
    severity='严重级别', # 'low' / 'medium' / 'high' / 'critical'
    patterns=[          # 正则模式列表
        r'(\bUNION\b.{0,20}\bSELECT\b)',
    ],
    target='all',       # 检查位置: 'path' / 'query' / 'body' / 'all'
    description='规则说明',
)
```

---

## 7. 模型训练与进化

### 从零训练

```bash
# 1. 生成训练数据
python scripts/waf_simulator.py --episodes 500 --output ./data

# 2. 训练模型
python standalone_waf.py \
  --data-path ./data/waf_trajectories_500ep.jsonl \
  --epochs 60 \
  --batch-size 64 \
  --save-dir ./out_waf_custom
```

### 增量训练（模型进化）

```bash
# 1. 生成攻击数据
python scripts/payload_generator.py --count 2000

# 2. 生成正常流量
python scripts/generate_normal.py 500

# 3. 收集 + 转换 + 重训练
python scripts/collect_and_retrain.py --retrain

# 4. 用新模型启动
python waf_server.py --model ./out_waf_evolved/standalone_waf_final.pth
```

### 白名单训练模式

```bash
# 1. 把扫描器 IP 加入白名单
curl -X POST http://localhost:5800/api/ips \
  -H "Content-Type: application/json" \
  -d '{"cidr":"127.0.0.1/32","type":"allow","action":"add"}'

# 2. 用 AWVS / SQLMap / Burp Suite 扫描
#    所有攻击 payload 被 WAF 记录但不会拦截

# 3. 扫描结束后重训练
python scripts/collect_and_retrain.py --retrain
```

---

## 8. 数据收集与主动学习

### 自动收集

WAF 运行期间，所有决策（规则拦截 + ML 决策）自动写入 `data/collected/buffer.jsonl`。

### 查看收集进度

```bash
python -c "
import json
from collections import Counter
with open('data/collected/buffer.jsonl', encoding='utf-8') as f:
    samples = [json.loads(l) for l in f if l.strip()]
acts = Counter(s['action'] for s in samples)
print(f'BLOCK={acts[2]}  ALLOW={acts.get(0,0)}  总计={len(samples)}')
"
```

### 训练数据格式

每条训练样本包含：

| 字段 | 说明 |
|:----|:------|
| `features[32]` | 32 维请求特征向量 |
| `action` | 0=ALLOW, 2=BLOCK |
| `attack_type` | 攻击类型（规则自动注入） |
| `source` | 'rule' 或 'ml' |
| `ip` | 源 IP |
| `method` | HTTP 方法 |
| `path` | 请求路径 |

---

## 9. 生产部署

### 系统服务（Linux）

```ini
[Unit]
Description=Mulun WAF
After=network.target

[Service]
Type=simple
User=waf
WorkingDirectory=/opt/mulun-waf
ExecStart=/usr/bin/python3 /opt/mulun-waf/waf_server.py --port 5800 --host 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 性能指标

| 场景 | 延迟 | 吞吐量 |
|:----|:----|:------|
| 规则引擎 | 1-5μs/req | >200,000 req/s |
| ML 模型 | 10-30μs/req | ~30,000 req/s |
| 完整决策 | 15-50μs/req | ~20,000 req/s |

### 注意事项

1. **首次部署**：先用白名单模式运行一段时间积累数据，再正式启用拦截
2. **日志监控**：定期检查 MONITOR 类型的请求，手动标注漏报/误报
3. **模型更新**：收集 500+ 条新样本后建议重训练一次
4. **规则更新**：新攻击出现时先在规则引擎中加正则，数据积累后模型会自动学会

---

## 10. 常见问题

### 为什么 ML 模型总是放行？

初始模型的置信度较低（~0.47），置信度门控会自动将其从 ALLOW 升级为 MONITOR。这是正常行为——模型需要足够数据才能建立高确信度。

### 如何避免误封？

1. 将已知正常的 IP 范围加入白名单
2. 置信度门控自动防止低确信的 BLOCK 决策
3. 不确定性 > 0.65 时强制 ALLOW

### 规则引擎和 ML 模型谁先处理？

规则引擎（L1）在前，ML 模型（L2）在后。规则明确拦截的直接 403，不经过 ML；规则不确定的进入 ML 判断。

### 训练数据不平衡怎么办？

用 `generate_normal.py` 补充正常流量样本。建议 BLOCK:ALLOW 比例在 3:1 到 5:1 之间。

### 如何迁移模型到新环境？

模型文件只有 `out_waf_v6/standalone_waf_final.pth`（264KB），复制到新环境即可：

```bash
python waf_server.py --model ./path/to/standalone_waf_final.pth
```

---

*Mulun-WAF 由玄幕安全团队开发维护。项目地址：https://github.com/guaidao2/MuLun-Waf*
