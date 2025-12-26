#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据配置文件提取指定联赛的比赛记录

用法: python extract_league_matches.py --config win007.g1.conf
      python extract_league_matches.py --config win007.g5.conf
"""

import argparse
import configparser
import os
import re
import zipfile
from datetime import datetime, timedelta

import pandas as pd


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='提取指定联赛的比赛记录')
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='配置文件名 (如 win007.g1.conf)'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/matches/win007_20220601_20251225.csv.zip',
        help='输入的zip文件路径'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/matches',
        help='输出目录'
    )
    parser.add_argument(
        '--months', '-m',
        type=int,
        default=2,
        help='集合2中比赛日期距离阈值(月)'
    )
    return parser.parse_args()


def load_config(config_path):
    """
    加载配置文件，提取联赛列表

    配置文件格式:
    [matches]
    selected = 意乙,法乙,西乙,德乙,德乙升,德乙降
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')

    if 'matches' not in config:
        raise ValueError(f"配置文件 {config_path} 缺少 [matches] 节")

    selected = config.get('matches', 'selected', fallback='')
    leagues = [league.strip() for league in selected.split(',') if league.strip()]

    return leagues


def load_csv_from_zip(zip_path):
    """从zip文件中加载CSV"""
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
    return df


def clean_team_name(team_name):
    """
    清理球队名称，去除排名前缀和后缀

    示例:
    - "[14]FC江原B队" -> "FC江原B队"
    - "大邱FCB队[15]" -> "大邱FCB队"
    - "[7]全州市民" -> "全州市民"
    - "晋州市民[16]" -> "晋州市民"
    """
    if pd.isna(team_name):
        return team_name

    team_name = str(team_name)

    # 去除前缀 [数字]
    team_name = re.sub(r'^\[\d+\]', '', team_name)

    # 去除后缀 [数字]
    team_name = re.sub(r'\[\d+\]$', '', team_name)

    return team_name.strip()


def extract_matches(df, config_leagues, months_threshold=2):
    """
    提取比赛记录

    集合1: 亚让和进球数都不为空，且联赛在配置内
    集合2: 集合1中的球队参加配置外联赛的比赛，日期在2个月内
    """
    # 过滤掉日期格式错误的行（数据偏移问题）
    valid_date_mask = df['full_start_time'].str.match(r'^\d{4}-\d{2}-\d{2}', na=False)
    df = df[valid_date_mask].copy()
    print(f"过滤无效日期后: {len(df)} 条")

    # 确保日期列是datetime类型
    df['full_start_time'] = pd.to_datetime(df['full_start_time'])

    # ========== 集合1: 目标联赛的有效比赛 ==========
    # 条件1: 亚让和进球数都不为空
    has_odds = df['亚让'].notna() & df['进球数'].notna()

    # 条件2: 联赛在配置内
    in_config = df['联赛'].isin(config_leagues)

    set1 = df[has_odds & in_config].copy()
    print(f"集合1 (目标联赛有效比赛): {len(set1)} 条")

    # 清理球队名称
    set1['主场球队'] = set1['主场球队'].apply(clean_team_name)
    set1['客场球队'] = set1['客场球队'].apply(clean_team_name)

    # ========== 集合2: 集合1球队的其他联赛比赛 ==========
    # 提取集合1中所有球队
    home_teams = set(set1['主场球队'].dropna().unique())
    away_teams = set(set1['客场球队'].dropna().unique())
    all_teams = home_teams | away_teams
    print(f"集合1中球队数量: {len(all_teams)}")

    # 获取每个球队在集合1中的参赛日期
    team_dates = {}
    for _, row in set1.iterrows():
        match_date = row['full_start_time']
        home = row['主场球队']
        away = row['客场球队']

        if home and pd.notna(home):
            if home not in team_dates:
                team_dates[home] = []
            team_dates[home].append(match_date)

        if away and pd.notna(away):
            if away not in team_dates:
                team_dates[away] = []
            team_dates[away].append(match_date)

    # 先清理所有球队名称用于匹配
    df_cleaned = df.copy()
    df_cleaned['主场球队_clean'] = df_cleaned['主场球队'].apply(clean_team_name)
    df_cleaned['客场球队_clean'] = df_cleaned['客场球队'].apply(clean_team_name)

    # 筛选配置外联赛
    not_in_config = ~df_cleaned['联赛'].isin(config_leagues)
    df_other_leagues = df_cleaned[not_in_config]

    # 日期阈值
    days_threshold = months_threshold * 30  # 约2个月

    set2_indices = []

    for idx, row in df_other_leagues.iterrows():
        match_date = row['full_start_time']
        home = row['主场球队_clean']
        away = row['客场球队_clean']

        # 检查主队是否在集合1的球队中
        if home in team_dates:
            for ref_date in team_dates[home]:
                if abs((match_date - ref_date).days) <= days_threshold:
                    set2_indices.append(idx)
                    break
            continue  # 如果已添加，跳过客队检查

        # 检查客队是否在集合1的球队中
        if away in team_dates:
            for ref_date in team_dates[away]:
                if abs((match_date - ref_date).days) <= days_threshold:
                    set2_indices.append(idx)
                    break

    set2 = df.loc[set2_indices].copy()
    print(f"集合2 (球队其他联赛比赛): {len(set2)} 条")

    # 清理集合2球队名称
    set2['主场球队'] = set2['主场球队'].apply(clean_team_name)
    set2['客场球队'] = set2['客场球队'].apply(clean_team_name)

    # ========== 合并集合1 + 集合2 ==========
    merged = pd.concat([set1, set2], ignore_index=True)

    # 按match_id去重
    merged = merged.drop_duplicates(subset=['match_id'], keep='first')
    print(f"合并去重后: {len(merged)} 条")

    # 按日期排序
    merged = merged.sort_values('full_start_time').reset_index(drop=True)

    return merged


def main():
    args = parse_args()

    print("=" * 60)
    print("提取指定联赛比赛记录")
    print("=" * 60)

    # 1. 加载配置文件
    config_path = os.path.join('conf/win007', args.config)
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        return

    leagues = load_config(config_path)
    print(f"\n配置文件: {args.config}")
    print(f"目标联赛: {', '.join(leagues)}")

    # 2. 加载CSV数据
    print(f"\n加载数据: {args.input}")
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return

    df = load_csv_from_zip(args.input)
    print(f"原始数据: {len(df)} 条记录")

    # 3. 提取比赛记录
    print(f"\n提取比赛记录 (日期阈值: {args.months}个月)...")
    result = extract_matches(df, leagues, args.months)

    # 4. 输出结果
    config_name = os.path.splitext(args.config)[0]  # 去掉.conf后缀
    output_file = os.path.join(args.output_dir, f'extracted_{config_name}.csv')

    os.makedirs(args.output_dir, exist_ok=True)
    result.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n输出文件: {output_file}")
    print(f"记录数: {len(result)}")

    # 5. 统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)

    print(f"\n联赛分布 (Top 10):")
    league_counts = result['联赛'].value_counts().head(10)
    for league, count in league_counts.items():
        in_config = "✓" if league in leagues else ""
        print(f"  {league}: {count} {in_config}")

    print(f"\n日期范围:")
    print(f"  最早: {result['full_start_time'].min()}")
    print(f"  最晚: {result['full_start_time'].max()}")

    print("\n完成!")


if __name__ == '__main__':
    main()
