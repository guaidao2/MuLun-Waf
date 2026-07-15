"""
主动学习循环：收集 WAF 决策数据 → 增量训练模型。

流程:
  1. WAF 运行中规则引擎拦截的攻击自动保存为训练样本
  2. 收集到足够新样本后重训练模型
  3. 模型逐步学会规则引擎的检测模式，最终能检测规则漏掉的攻击

用法:
    python scripts/collect_and_retrain.py --collect   # 从 WAF 日志提取训练数据
    python scripts/collect_and_retrain.py --retrain   # 重训练模型
    python scripts/collect_and_retrain.py --auto      # 自动: 收集+重训练
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_here)
sys.path.insert(0, _ROOT)

# 收集目录
COLLECT_DIR = os.path.join(_ROOT, 'data', 'collected')
os.makedirs(COLLECT_DIR, exist_ok=True)

BUFFER_PATH = os.path.join(COLLECT_DIR, 'buffer.jsonl')  # 运行时累积
DATASET_PATH = os.path.join(COLLECT_DIR, 'training_data.jsonl')  # 整理后的训练集


def collect_from_waf(api_url='http://localhost:5800'):
    """从 WAF API 提取最近的决策日志转为训练样本。"""
    import urllib.request

    try:
        r = json.loads(urllib.request.urlopen(f'{api_url}/api/stats', timeout=5).read())
        logs = r.get('recent', [])
    except:
        print('[Collect] WAF API 不可用，跳过收集')
        return 0

    # 读已有缓存，去重
    existing = set()
    if os.path.exists(BUFFER_PATH):
        with open(BUFFER_PATH, encoding='utf-8') as f:
            for line in f:
                s = json.loads(line)
                existing.add(s.get('_id', ''))

    new_count = 0
    with open(BUFFER_PATH, 'a', encoding='utf-8') as f:
        for log in logs:
            # 跳过重复
            log_id = f"{log.get('ip','')}_{log.get('timestamp','')}"
            if log_id in existing:
                continue
            existing.add(log_id)

            # 构造训练样本
            sample = {
                '_id': log_id,
                'features': extract_features_from_log(log),
                'action': 2 if log['action'] == 'BLOCK' else (1 if log['action'] == 'MONITOR' else 0),
                'strategy': 0 if log['action'] == 'BLOCK' else 1,
                'value': 0.9 if log['action'] == 'BLOCK' else 0.3,
                'source': log.get('source', 'ml'),
                'timestamp': log.get('timestamp', time.time()),
                'ip': log.get('ip', ''),
                'method': log.get('method', ''),
                'path': log.get('path', ''),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            new_count += 1

    print(f'[Collect] 新增 {new_count} 条, 缓存共 {len(existing)} 条')
    return new_count


def extract_features_from_log(log):
    """从日志条目提取特征向量（32维，全零占位，WAF 完整版应存实际特征）。"""
    # 简化版本：基于日志信息构造一个简单特征
    feats = [0.0] * 32
    action = log.get('action', '')
    if action == 'BLOCK':
        feats[0] = 0.8   # 威胁特征
    # 从 reason 中提取一些特征信号
    reason = log.get('reason', '')
    if 'SQL' in reason or 'sqli' in reason:
        feats[0] = 0.9
    elif 'XSS' in reason or 'xss' in reason:
        feats[1] = 0.9
    elif '路径' in reason or '遍历' in reason:
        feats[2] = 0.9
    elif '命令' in reason or 'RCE' in reason:
        feats[3] = 0.9
    elif 'SSRF' in reason:
        feats[4] = 0.9
    elif 'SSTI' in reason or '模板' in reason:
        feats[0] = 0.85
    elif 'JNDI' in reason or 'Log4j' in reason:
        feats[0] = 0.95
    return feats


def prepare_training_set(min_samples=50):
    """将缓存数据整理为训练集。"""
    buffer = []
    if os.path.exists(BUFFER_PATH):
        with open(BUFFER_PATH, encoding='utf-8') as f:
            for line in f:
                buffer.append(json.loads(line))

    # 按 source 分组统计
    rule_samples = [s for s in buffer if s.get('source') == 'rule' and s['action'] == 2]
    ml_samples = [s for s in buffer if s.get('source') == 'ml']

    print(f'[Prepare] 缓存共 {len(buffer)} 条: 规则拦截 {len(rule_samples)} 条, ML 决策 {len(ml_samples)} 条')

    if len(buffer) < min_samples:
        print(f'[Prepare] 样本不足 {min_samples} 条，跳过训练集生成')
        return None

    # 合并：所有规则拦截作为正样本 + ML 决策样本
    train_set = rule_samples + [s for s in ml_samples if s['action'] != 2]
    print(f'[Prepare] 训练集: {len(train_set)} 条')

    # 保存为训练集
    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        for s in train_set:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    return DATASET_PATH


def retrain():
    """用收集的训练数据重训练 standalone_waf 模型。"""
    dataset_path = prepare_training_set()
    if not dataset_path:
        return

    print(f'[Retrain] 开始重训练...')
    # 转换数据格式为 standalone_waf 可用的 jsonl
    # （当前是简化版，完整版需要把 features 对齐到模型的输入格式）
    converted_path = os.path.join(COLLECT_DIR, 'converted.jsonl')
    with open(dataset_path, encoding='utf-8') as f, \
         open(converted_path, 'w', encoding='utf-8') as out:
        for line in f:
            s = json.loads(line)
            out.write(json.dumps({
                'structured': {
                    'state': s['features'][:16],  # 16 维状态
                    'features': s['features'][:32],
                    'strategy': s.get('strategy', 1),
                    'action': s.get('action', 0),
                    'value': s.get('value', 0.5),
                },
                'metadata': {
                    'episode': 0,
                    'step': 0,
                    'is_attack': s['action'] == 2,
                    'attack_type': s.get('source', ''),
                    'source_ip': s.get('ip', ''),
                },
                'conversations': [
                    {'role': 'system', 'content': 'WAF 安全引擎'},
                    {'role': 'user', 'content': f'WAF 日志: {s.get("method","")} {s.get("path","")}'},
                    {'role': 'assistant', 'content': f'决策: {s.get("action","")}'},
                ],
            }, ensure_ascii=False) + '\n')

    print(f'[Retrain] 转换完成: {converted_path}')
    print(f'[Retrain] 用以下命令重训练:')
    print(f'  python standalone_waf.py --data-path {converted_path}')
    print(f'    --epochs 30 --batch-size 32 --save-dir ./out_waf_evolved')
    print(f'[Retrain] 然后用新模型启动 WAF:')
    print(f'  python waf_server.py --model ./out_waf_evolved/standalone_waf_final.pth')


def auto():
    """自动收集+重训练。"""
    count = collect_from_waf()
    if count > 0:
        retrain()
    else:
        print('[Auto] 无新数据')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WAF 主动学习循环')
    parser.add_argument('--collect', action='store_true', help='从 WAF 收集训练数据')
    parser.add_argument('--retrain', action='store_true', help='重训练模型')
    parser.add_argument('--auto', action='store_true', help='自动收集+重训练')
    args = parser.parse_args()

    if args.collect:
        collect_from_waf()
    elif args.retrain:
        retrain()
    elif args.auto:
        auto()
    else:
        parser.print_help()
