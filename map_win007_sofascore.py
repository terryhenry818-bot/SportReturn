"""
Win007 vs Sofascore 比赛映射脚本 (优化版)
1. 加载 all_sofascore.csv.zip 和 extracted_win007.g1-g12.csv
2. 基于球队名称映射、比赛时间、比分等条件进行比赛映射
3. 输出映射成功的记录
"""

import pandas as pd
import numpy as np
import re
import zipfile
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
SOFASCORE_ZIP = 'data/matches/all_sofascore.csv.zip'
TEAM_MAPPING_FILE = 'a0_sofascore_and_win007_teams.csv'
WIN007_FILES = [f'data/matches/extracted_win007.g{i}.csv' for i in range(1, 13)]
OUTPUT_DIR = 'data/matches'
TIME_TOLERANCE_HOURS = 30

# ============ 辅助函数 ============
def clean_win007_team_name(team_name):
    """清理win007队名，去掉[xx]前缀和后缀"""
    if pd.isna(team_name):
        return ''
    team_name = str(team_name)
    # 去掉开头的[...]前缀
    team_name = re.sub(r'^\[[^\]]*\]', '', team_name)
    # 去掉结尾的[...]后缀
    team_name = re.sub(r'\[[^\]]*\]$', '', team_name)
    return team_name.strip()


def parse_win007_score(score_str):
    """解析win007比分，如[1:3]"""
    if pd.isna(score_str):
        return None, None
    match = re.search(r'\[(\d+):(\d+)\]', str(score_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


# ============ 加载数据 ============
print("=" * 70)
print("Win007 vs Sofascore 比赛映射")
print("=" * 70)

# 加载sofascore数据
print("\n[1] 加载 all_sofascore.csv.zip...")
with zipfile.ZipFile(SOFASCORE_ZIP, 'r') as z:
    with z.open('all_sofascore.csv') as f:
        df_sofascore = pd.read_csv(f, encoding='utf-8-sig')

# 过滤掉重复的header行
df_sofascore = df_sofascore[df_sofascore['date'] != 'date'].copy()
print(f"    Sofascore比赛数: {len(df_sofascore)}")

# 解析sofascore时间
df_sofascore['datetime'] = pd.to_datetime(df_sofascore['date'] + ' ' + df_sofascore['time'], errors='coerce')
df_sofascore['date_key'] = pd.to_datetime(df_sofascore['date'], errors='coerce').dt.strftime('%Y-%m-%d')

# 确保比分是数值类型
df_sofascore['home_goals'] = pd.to_numeric(df_sofascore['home_goals'], errors='coerce')
df_sofascore['away_goals'] = pd.to_numeric(df_sofascore['away_goals'], errors='coerce')
df_sofascore['home_team_id'] = pd.to_numeric(df_sofascore['home_team_id'], errors='coerce')
df_sofascore['away_team_id'] = pd.to_numeric(df_sofascore['away_team_id'], errors='coerce')

# 加载球队映射
print("\n[2] 加载球队映射...")
df_team_mapping = pd.read_csv(TEAM_MAPPING_FILE, encoding='utf-8-sig')
team_cn_to_en = dict(zip(df_team_mapping['win007_team_name'], df_team_mapping['sofascore_team_name']))
team_cn_to_id = dict(zip(df_team_mapping['win007_team_name'], df_team_mapping['sofascore_team_id']))
print(f"    已有映射数: {len(team_cn_to_en)}")

# 加载win007数据
print("\n[3] 加载 extracted_win007.g1-g12.csv...")
win007_dfs = {}
for i, filepath in enumerate(WIN007_FILES, 1):
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        # 解析时间
        df['datetime'] = pd.to_datetime(df['full_start_time'], errors='coerce')
        df['home_clean'] = df['主场球队'].apply(clean_win007_team_name)
        df['away_clean'] = df['客场球队'].apply(clean_win007_team_name)
        # 解析比分
        scores = df['比分'].apply(lambda x: parse_win007_score(x))
        df['home_goals'] = scores.apply(lambda x: x[0])
        df['away_goals'] = scores.apply(lambda x: x[1])
        win007_dfs[i] = df
        print(f"    g{i}: {len(df)} 条记录")
    except FileNotFoundError:
        print(f"    g{i}: 文件不存在，跳过")
        win007_dfs[i] = pd.DataFrame()

# ============ 构建Sofascore索引 ============
print("\n[4] 构建Sofascore比赛索引...")

# 按日期建立索引
sofascore_by_date = defaultdict(list)
for idx, row in df_sofascore.iterrows():
    if pd.notna(row['date_key']):
        sofascore_by_date[row['date_key']].append(idx)

# 按主队ID+客队ID建立索引
sofascore_by_teams = defaultdict(list)
for idx, row in df_sofascore.iterrows():
    key = (row['home_team_id'], row['away_team_id'])
    sofascore_by_teams[key].append(idx)

# 按队名建立索引
sofascore_by_names = defaultdict(list)
for idx, row in df_sofascore.iterrows():
    key = (str(row['home_team']).lower().strip(), str(row['away_team']).lower().strip())
    sofascore_by_names[key].append(idx)

print(f"    按日期索引: {len(sofascore_by_date)} 天")
print(f"    按队ID索引: {len(sofascore_by_teams)} 组合")
print(f"    按队名索引: {len(sofascore_by_names)} 组合")


# ============ 挖掘新的球队映射 (优化版) ============
print("\n[5] 挖掘潜在球队映射...")

# 收集所有未映射的队名
all_win007_teams = set()
for df in win007_dfs.values():
    if len(df) == 0:
        continue
    all_win007_teams.update(df['home_clean'].dropna().unique())
    all_win007_teams.update(df['away_clean'].dropna().unique())

unmapped_teams = all_win007_teams - set(team_cn_to_en.keys())
print(f"    未映射球队数: {len(unmapped_teams)}")

# 构建 (日期, 比分) -> 索引列表 的快速查找表
print("    构建比分-日期索引...")
sofascore_by_score_date = defaultdict(list)
for idx, row in df_sofascore.iterrows():
    date_key = row['date_key']
    home_goals = row['home_goals']
    away_goals = row['away_goals']
    if pd.notna(date_key) and pd.notna(home_goals) and pd.notna(away_goals):
        key = (date_key, int(home_goals), int(away_goals))
        sofascore_by_score_date[key].append(idx)

# 通过比赛时间和比分匹配来挖掘映射
match_candidates = defaultdict(lambda: defaultdict(int))
processed = 0

for group_id, df in win007_dfs.items():
    if len(df) == 0:
        continue

    for _, row in df.iterrows():
        win007_home = row['home_clean']
        win007_away = row['away_clean']

        # 只处理有未映射球队的比赛
        home_unmapped = win007_home not in team_cn_to_en
        away_unmapped = win007_away not in team_cn_to_en

        if not home_unmapped and not away_unmapped:
            continue

        win007_dt = row['datetime']
        home_goals = row['home_goals']
        away_goals = row['away_goals']

        if pd.isna(win007_dt) or pd.isna(home_goals):
            continue

        home_goals = int(home_goals)
        away_goals = int(away_goals)

        # 只搜索±2天范围内的sofascore比赛，且比分相同
        candidate_indices = []
        for delta in range(-2, 3):
            d = (win007_dt + timedelta(days=delta)).strftime('%Y-%m-%d')
            key = (d, home_goals, away_goals)
            candidate_indices.extend(sofascore_by_score_date.get(key, []))

        for sf_idx in candidate_indices:
            sf_row = df_sofascore.loc[sf_idx]
            sf_dt = sf_row['datetime']

            if pd.isna(sf_dt):
                continue

            # 时间差检查
            time_diff = abs((win007_dt - sf_dt).total_seconds() / 3600)
            if time_diff > TIME_TOLERANCE_HOURS:
                continue

            # 找到潜在匹配
            sf_home = str(sf_row['home_team'])
            sf_away = str(sf_row['away_team'])
            sf_home_id = sf_row['home_team_id']
            sf_away_id = sf_row['away_team_id']

            if home_unmapped:
                match_candidates[win007_home][(sf_home, sf_home_id)] += 1
            if away_unmapped:
                match_candidates[win007_away][(sf_away, sf_away_id)] += 1

        processed += 1

print(f"    处理了 {processed} 场比赛")

# 选择最高频的映射
new_mappings = {}
for cn_name, candidates in match_candidates.items():
    if not candidates:
        continue
    best_match = max(candidates.items(), key=lambda x: x[1])
    (en_name, team_id), count = best_match
    if count >= 2:  # 至少2次匹配才确认
        new_mappings[cn_name] = (en_name, team_id, count)

print(f"    新挖掘映射数: {len(new_mappings)}")

# 合并映射
for cn_name, (en_name, team_id, count) in new_mappings.items():
    team_cn_to_en[cn_name] = en_name
    team_cn_to_id[cn_name] = team_id

print(f"    合并后总映射数: {len(team_cn_to_en)}")


# ============ 比赛映射 ============
def map_matches(df_win007, df_sofascore, group_id):
    """映射win007比赛到sofascore"""

    matched_pairs = []
    sofascore_used = set()

    for win007_idx, row in df_win007.iterrows():
        win007_home = row['home_clean']
        win007_away = row['away_clean']
        win007_dt = row['datetime']
        home_goals = row['home_goals']
        away_goals = row['away_goals']

        if pd.isna(win007_dt) or pd.isna(home_goals):
            continue

        # 尝试通过球队ID匹配
        candidates = []

        if win007_home in team_cn_to_id and win007_away in team_cn_to_id:
            home_id = team_cn_to_id[win007_home]
            away_id = team_cn_to_id[win007_away]

            for sf_idx in sofascore_by_teams.get((home_id, away_id), []):
                if sf_idx in sofascore_used:
                    continue
                candidates.append(sf_idx)

        # 如果通过ID没找到，尝试通过队名匹配
        if not candidates and win007_home in team_cn_to_en and win007_away in team_cn_to_en:
            en_home = team_cn_to_en[win007_home].lower().strip()
            en_away = team_cn_to_en[win007_away].lower().strip()

            for sf_idx in sofascore_by_names.get((en_home, en_away), []):
                if sf_idx in sofascore_used:
                    continue
                candidates.append(sf_idx)

        # 在候选中找最佳匹配
        best_match = None
        best_score = -1

        for sf_idx in candidates:
            sf_row = df_sofascore.loc[sf_idx]
            sf_dt = sf_row['datetime']

            if pd.isna(sf_dt):
                continue

            # 时间差检查
            time_diff = abs((win007_dt - sf_dt).total_seconds() / 3600)
            if time_diff > TIME_TOLERANCE_HOURS:
                continue

            # 比分检查
            if sf_row['home_goals'] != home_goals or sf_row['away_goals'] != away_goals:
                continue

            # 计算匹配分数（时间越近分数越高）
            score = 100 - time_diff
            if score > best_score:
                best_score = score
                best_match = sf_idx

        if best_match is not None:
            matched_pairs.append((win007_idx, best_match))
            sofascore_used.add(best_match)

    return matched_pairs


print("\n[6] 执行比赛映射...")
all_matched = []
group_results = {}

for group_id, df_win007 in win007_dfs.items():
    if len(df_win007) == 0:
        group_results[group_id] = (0, 0, [])
        continue

    matched_pairs = map_matches(df_win007, df_sofascore, group_id)
    group_results[group_id] = (len(df_win007), len(matched_pairs), matched_pairs)

    for win007_idx, sf_idx in matched_pairs:
        all_matched.append({
            'group_id': group_id,
            'win007_idx': win007_idx,
            'sofascore_idx': sf_idx
        })

    match_rate = len(matched_pairs) / len(df_win007) * 100 if len(df_win007) > 0 else 0
    print(f"    g{group_id}: {len(matched_pairs)}/{len(df_win007)} 匹配成功 ({match_rate:.1f}%)")

total_win007 = sum(r[0] for r in group_results.values())
total_matched = sum(r[1] for r in group_results.values())
print(f"\n    总计: {total_matched}/{total_win007} 匹配成功 ({total_matched/total_win007*100:.1f}%)")


# ============ 输出结果 ============
print("\n[7] 输出映射结果...")

# 输出每个group的sofascore记录
for group_id, (total, matched_count, matched_pairs) in group_results.items():
    if matched_count == 0:
        continue

    sf_indices = [pair[1] for pair in matched_pairs]
    df_sf_matched = df_sofascore.loc[sf_indices].copy()

    # 移除辅助列
    drop_cols = ['datetime', 'date_key']
    df_sf_matched = df_sf_matched.drop(columns=[c for c in drop_cols if c in df_sf_matched.columns])

    output_path = f"{OUTPUT_DIR}/extracted_sofascore.g{group_id}.csv"
    df_sf_matched.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"    已输出: extracted_sofascore.g{group_id}.csv ({len(df_sf_matched)} 条)")


# 输出合并的mapped_matches文件
print("\n[8] 输出 mapped_matches_win007_sofascore.csv...")

mapped_records = []

for match_info in all_matched:
    group_id = match_info['group_id']
    win007_idx = match_info['win007_idx']
    sf_idx = match_info['sofascore_idx']

    sf_row = df_sofascore.loc[sf_idx]
    win007_row = win007_dfs[group_id].loc[win007_idx]

    record = {
        # Sofascore列
        'sofascore_match_id': sf_row['match_id'],
        'sofascore_date': sf_row['date'],
        'sofascore_time': sf_row['time'],
        'sofascore_weekday': sf_row['weekday'],
        'sofascore_competition': sf_row['competition'],
        'sofascore_season': sf_row['season'],
        'sofascore_round': sf_row['round'],
        'sofascore_venue': sf_row.get('venue', ''),
        'sofascore_opponent': sf_row.get('opponent', ''),
        'sofascore_home_team': sf_row['home_team'],
        'sofascore_away_team': sf_row['away_team'],
        'sofascore_home_team_id': sf_row['home_team_id'],
        'sofascore_away_team_id': sf_row['away_team_id'],
        'sofascore_home_goals': sf_row['home_goals'],
        'sofascore_away_goals': sf_row['away_goals'],
        'sofascore_home_ht': sf_row['home_ht'],
        'sofascore_away_ht': sf_row['away_ht'],
        'sofascore_team_goals': sf_row.get('team_goals', ''),
        'sofascore_opponent_goals': sf_row.get('opponent_goals', ''),
        'sofascore_result': sf_row.get('result', ''),
        'sofascore_status': sf_row['status'],
        'sofascore_match_url': sf_row['match_url'],
        # Win007列
        'win007_date_str': win007_row['date_str'],
        'win007_full_start_time': win007_row['full_start_time'],
        'win007_match_id': win007_row['match_id'],
        'win007_联赛': win007_row['联赛'],
        'win007_赛事时间': win007_row['赛事时间'],
        'win007_状态': win007_row['状态'],
        'win007_主场球队': win007_row['主场球队'],
        'win007_比分': win007_row['比分'],
        'win007_客场球队': win007_row['客场球队'],
        'win007_半场': win007_row['半场'],
        'win007_亚让': win007_row['亚让'],
        'win007_进球数': win007_row['进球数'],
        'win007_数据': win007_row['数据'],
    }
    mapped_records.append(record)

df_mapped = pd.DataFrame(mapped_records)
output_path = f"{OUTPUT_DIR}/mapped_matches_win007_sofascore.csv"
df_mapped.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"    已输出: mapped_matches_win007_sofascore.csv ({len(df_mapped)} 条)")

# ============ 统计汇总 ============
print("\n" + "=" * 70)
print("映射统计汇总")
print("=" * 70)
print(f"\nSofascore总比赛数: {len(df_sofascore)}")
print(f"Win007总比赛数: {total_win007}")
print(f"成功映射数: {total_matched}")
print(f"映射成功率: {total_matched/total_win007*100:.1f}%")
print(f"\n球队映射:")
print(f"  原有映射: {len(df_team_mapping)}")
print(f"  新挖掘映射: {len(new_mappings)}")
print(f"  合并后总计: {len(team_cn_to_en)}")

print("\n各组映射详情:")
print(f"{'Group':<8} {'Win007':<10} {'已映射':<10} {'成功率':<10}")
print("-" * 40)
for group_id in range(1, 13):
    total, matched, _ = group_results.get(group_id, (0, 0, []))
    rate = matched / total * 100 if total > 0 else 0
    print(f"g{group_id:<7} {total:<10} {matched:<10} {rate:.1f}%")

print("\n" + "=" * 70)
print("完成!")
print("=" * 70)
