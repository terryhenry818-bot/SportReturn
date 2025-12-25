#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比赛映射工具 - 将SofaScore和Win007的比赛数据进行匹配

功能：
1. 读取五大联赛球队映射表
2. 读取SofaScore和Win007的比赛CSV
3. 根据球队名称和比赛日期进行匹配
4. 输出合并后的CSV（包含两个来源的所有字段）
"""

import os
import re
import csv
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Set, Tuple, Optional


def load_team_mapping(mapping_file: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    加载球队映射表

    Args:
        mapping_file: 映射文件路径 (a0_sofascore_and_win007_teams.csv)

    Returns:
        win007_to_sofascore: win007中文名 -> sofascore team_id
        sofascore_to_win007: sofascore team_id -> win007中文名
    """
    win007_to_sofascore = {}
    sofascore_to_win007 = {}

    try:
        with open(mapping_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                win007_name = row.get('win007_team_name', '').strip()
                sofascore_id = row.get('sofascore_team_id', '').strip()

                if win007_name and sofascore_id:
                    try:
                        sofascore_id = int(sofascore_id)
                        win007_to_sofascore[win007_name] = sofascore_id
                        sofascore_to_win007[sofascore_id] = win007_name
                    except ValueError:
                        pass

        print(f"✓ 加载了 {len(win007_to_sofascore)} 个球队映射")

    except FileNotFoundError:
        print(f"✗ 映射文件不存在: {mapping_file}")
    except Exception as e:
        print(f"✗ 加载映射文件失败: {e}")

    return win007_to_sofascore, sofascore_to_win007


def clean_win007_team_name(team_name: str) -> str:
    """
    清洗Win007球队名称，去掉排名前缀/后缀

    例如:
        "[英超1]阿森纳" -> "阿森纳"
        "水晶宫[英超8]" -> "水晶宫"
    """
    if not team_name:
        return team_name
    # 去掉所有 [...] 形式的标记
    cleaned = re.sub(r'\[.*?\]', '', team_name)
    return cleaned.strip()


def parse_date(date_str: str) -> Optional[datetime]:
    """
    解析日期字符串，支持多种格式
    """
    if not date_str or pd.isna(date_str):
        return None

    date_str = str(date_str).strip()

    formats = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%Y%m%d',
        '%d/%m/%Y',
        '%d-%m-%Y',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def load_sofascore_matches(csv_file: str) -> pd.DataFrame:
    """
    加载SofaScore比赛CSV
    """
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        print(f"✓ SofaScore: 加载 {len(df)} 场比赛")
        return df
    except Exception as e:
        print(f"✗ 加载SofaScore CSV失败: {e}")
        return pd.DataFrame()


def load_win007_matches(csv_file: str) -> pd.DataFrame:
    """
    加载Win007比赛CSV
    """
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        print(f"✓ Win007: 加载 {len(df)} 场比赛")
        return df
    except Exception as e:
        print(f"✗ 加载Win007 CSV失败: {e}")
        return pd.DataFrame()


def match_by_teams_and_date(
    sofascore_df: pd.DataFrame,
    win007_df: pd.DataFrame,
    win007_to_sofascore: Dict[str, int]
) -> pd.DataFrame:
    """
    根据球队和日期匹配比赛

    匹配逻辑：
    1. Win007球队名清洗后通过映射表找到SofaScore team_id
    2. 日期匹配：允许±1天的误差（考虑跨日比赛）
    3. 双方球队都匹配成功才算匹配
    """
    matched_rows = []
    matched_win007_indices = set()
    matched_sofascore_indices = set()

    # 为Win007数据添加清洗后的球队名和sofascore team_id
    win007_df = win007_df.copy()
    win007_df['home_team_clean'] = win007_df['主场球队'].apply(clean_win007_team_name)
    win007_df['away_team_clean'] = win007_df['客场球队'].apply(clean_win007_team_name)
    win007_df['home_sofascore_id'] = win007_df['home_team_clean'].map(win007_to_sofascore)
    win007_df['away_sofascore_id'] = win007_df['away_team_clean'].map(win007_to_sofascore)

    # 为Win007数据解析日期
    if 'date_str' in win007_df.columns:
        win007_df['parsed_date'] = win007_df['date_str'].apply(parse_date)
    elif 'full_start_time' in win007_df.columns:
        # 从full_start_time提取日期
        win007_df['parsed_date'] = win007_df['full_start_time'].apply(
            lambda x: parse_date(str(x)[:10]) if pd.notna(x) else None
        )
    else:
        print("⚠️ Win007 CSV中找不到日期字段")
        win007_df['parsed_date'] = None

    # 为SofaScore数据解析日期
    if 'date' in sofascore_df.columns:
        sofascore_df = sofascore_df.copy()
        sofascore_df['parsed_date'] = sofascore_df['date'].apply(parse_date)
    else:
        print("⚠️ SofaScore CSV中找不到date字段")
        sofascore_df['parsed_date'] = None

    # 遍历Win007比赛进行匹配
    for w_idx, w_row in win007_df.iterrows():
        home_sf_id = w_row.get('home_sofascore_id')
        away_sf_id = w_row.get('away_sofascore_id')
        w_date = w_row.get('parsed_date')

        # 需要双方球队都有映射
        if pd.isna(home_sf_id) or pd.isna(away_sf_id):
            continue

        home_sf_id = int(home_sf_id)
        away_sf_id = int(away_sf_id)

        # 在SofaScore中查找匹配
        for s_idx, s_row in sofascore_df.iterrows():
            if s_idx in matched_sofascore_indices:
                continue

            s_home_id = s_row.get('home_team_id')
            s_away_id = s_row.get('away_team_id')
            s_date = s_row.get('parsed_date')

            if pd.isna(s_home_id) or pd.isna(s_away_id):
                continue

            try:
                s_home_id = int(s_home_id)
                s_away_id = int(s_away_id)
            except (ValueError, TypeError):
                continue

            # 球队ID匹配
            if home_sf_id != s_home_id or away_sf_id != s_away_id:
                continue

            # 日期匹配（允许±1天误差）
            date_match = False
            if w_date and s_date:
                date_diff = abs((w_date - s_date).days)
                date_match = date_diff <= 1
            elif not w_date and not s_date:
                # 如果都没有日期，只靠球队匹配
                date_match = True

            if date_match:
                # 匹配成功，合并数据
                matched_win007_indices.add(w_idx)
                matched_sofascore_indices.add(s_idx)

                # 创建合并行
                merged_row = {}

                # 添加SofaScore字段（加前缀）
                for col in sofascore_df.columns:
                    if col not in ['parsed_date']:
                        merged_row[f'sofascore_{col}'] = s_row[col]

                # 添加Win007字段（加前缀）
                for col in win007_df.columns:
                    if col not in ['parsed_date', 'home_team_clean', 'away_team_clean',
                                   'home_sofascore_id', 'away_sofascore_id']:
                        merged_row[f'win007_{col}'] = w_row[col]

                matched_rows.append(merged_row)
                break

    print(f"✓ 匹配成功: {len(matched_rows)} 场比赛")
    print(f"  - Win007 未匹配: {len(win007_df) - len(matched_win007_indices)}")
    print(f"  - SofaScore 未匹配: {len(sofascore_df) - len(matched_sofascore_indices)}")

    return pd.DataFrame(matched_rows)


def main():
    parser = argparse.ArgumentParser(
        description='比赛映射工具 - 匹配SofaScore和Win007比赛数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python a0_zmapping_matches.py \\
    --sofascore-csv data/matches/sofascore_top5.csv \\
    --win007-csv data/matches/win007_top5.csv \\
    --mapping-csv a0_sofascore_and_win007_teams.csv \\
    --output data/matches/mapped_matches.csv
        """
    )

    parser.add_argument(
        '--sofascore-csv', '-s',
        required=True,
        help='SofaScore五大联赛比赛CSV文件路径'
    )

    parser.add_argument(
        '--win007-csv', '-w',
        required=True,
        help='Win007五大联赛比赛CSV文件路径'
    )

    parser.add_argument(
        '--mapping-csv', '-m',
        default='a0_sofascore_and_win007_teams.csv',
        help='球队映射CSV文件路径 (默认: a0_sofascore_and_win007_teams.csv)'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出合并后的CSV文件路径'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("比赛映射工具")
    print("="*60)
    print(f"SofaScore CSV: {args.sofascore_csv}")
    print(f"Win007 CSV: {args.win007_csv}")
    print(f"映射表: {args.mapping_csv}")
    print(f"输出文件: {args.output}")
    print("="*60 + "\n")

    # 加载映射表
    win007_to_sofascore, sofascore_to_win007 = load_team_mapping(args.mapping_csv)

    if not win007_to_sofascore:
        print("✗ 无法加载球队映射，退出")
        return

    # 加载比赛数据
    sofascore_df = load_sofascore_matches(args.sofascore_csv)
    win007_df = load_win007_matches(args.win007_csv)

    if sofascore_df.empty or win007_df.empty:
        print("✗ 比赛数据为空，退出")
        return

    # 执行匹配
    print("\n开始匹配...")
    merged_df = match_by_teams_and_date(sofascore_df, win007_df, win007_to_sofascore)

    if merged_df.empty:
        print("✗ 没有匹配到任何比赛")
        return

    # 保存结果
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    merged_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"\n✓ 已保存合并结果: {args.output}")
    print(f"  总列数: {len(merged_df.columns)}")
    print(f"  总行数: {len(merged_df)}")

    # 显示列名
    print("\n合并后的字段:")
    sofascore_cols = [c for c in merged_df.columns if c.startswith('sofascore_')]
    win007_cols = [c for c in merged_df.columns if c.startswith('win007_')]
    print(f"  SofaScore字段: {len(sofascore_cols)}")
    print(f"  Win007字段: {len(win007_cols)}")


if __name__ == '__main__':
    main()
