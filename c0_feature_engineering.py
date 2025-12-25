#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Football Match Feature Engineering Pipeline - 超高性能版本
优化策略：
1. 预扫描目录，建立文件索引（避免重复glob）
2. 批量读取+内存缓存
3. 多进程并行处理
4. 减少pandas开销
"""

import os
import re
import pandas as pd
import numpy as np
from glob import glob
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict
import time

warnings.filterwarnings('ignore')

# ============== 全局解析函数 ==============

def parse_percentage(val):
    if pd.isna(val) or val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val)
    if '%' in val:
        try:
            return float(val.replace('%', ''))
        except:
            return np.nan
    return np.nan

def parse_ratio(val):
    if pd.isna(val) or val is None:
        return np.nan, np.nan, np.nan
    val = str(val)
    match = re.match(r'(\d+)/(\d+)\s*\(([0-9.]+)%\)(?:/(\d+))?', val)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    try:
        return float(val), np.nan, np.nan
    except:
        return np.nan, np.nan, np.nan

HANDICAP_MAP = {
    '平手': 0, '平': 0, '平手/半球': 0.25, '平/半': 0.25,
    '半球': 0.5, '半': 0.5, '半球/一球': 0.75, '半/一': 0.75,
    '一球': 1.0, '一': 1.0, '一球/球半': 1.25, '一/球半': 1.25,
    '球半': 1.5, '球半/两球': 1.75, '两球': 2.0, '二球': 2.0,
    '两球/两球半': 2.25, '两球半': 2.5, '二球半': 2.5,
    '两球半/三球': 2.75, '三球': 3.0, '三球/三球半': 3.25,
    '三球半': 3.5, '三球半/四球': 3.75, '四球': 4.0,
    '四球/四球半': 4.25, '四球半': 4.5,
    '一球': 1, '三球': 3, '两球': 2, '半球': 0.5, '四球': 4, '平手': 0, '球半': 1.5, '一球/球半': 1.25,
    '半球/一球': 0.75, '平手/半球': 0.25, '球半/两球': 1.75, '三球/三球半': 3.25, '两球/两球半': 2.25,
    '四球/四球半': 4.25, '三球半': 3.5, '两球半': 2.5, '四球半': 4.5, '三球半/四球': 3.75, 
      '两球半/三球': 2.75, '受让一球': -1, '受让七球': -7, '受让三球': -3, '受让两球': -2, '受让五球': -5, 
      '受让六球': -6, '受让半球': -0.5, '受让四球': -4, '受让球半': -1.5, '受让一球/球半': -1.25, 
      '受让半球/一球': -0.75, '受让平手/半球': -0.25, '受让球半/两球': -1.75, '受让七球/七球半': -7.25, 
      '受让三球/三球半': -3.25, '受让两球/两球半': -2.25, '受让五球/五球半': -5.25, '受让六球/六球半': -6.25, 
      '受让四球/四球半': -4.25, '受让七球半': -7.5, '受让三球半': -3.5, '受让两球半': -2.5, '受让五球半': -5.5, 
      '受让六球半': -6.5, '受让四球半': -4.5, '受让七球半/八球': -7.75, '受让三球半/四球': -3.75, 
      '受让两球半/三球': -2.75, '受让五球半/六球': -5.75, '受让六球半/七球': -6.75, 
      '受让四球半/五球': -4.75
}

def parse_handicap(val):
    if pd.isna(val) or val is None:
        return np.nan
    return HANDICAP_MAP.get(str(val).strip(), np.nan)

def parse_overunder_line(val):
    if pd.isna(val) or val is None:
        return np.nan
    val = str(val).strip()
    try:
        return float(val)
    except:
        pass
    if '/' in val:
        parts = val.split('/')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            pass
    return np.nan

def safe_float(val):
    """快速安全转换为float"""
    if pd.isna(val) or val is None:
        return np.nan
    try:
        return float(val)
    except:
        return np.nan

# ============== 文件索引构建 ==============

def build_file_index(directory, pattern_func):
    """构建文件索引，避免重复glob"""
    index = {}
    if not os.path.exists(directory):
        return index
    for f in os.listdir(directory):
        key = pattern_func(f)
        if key:
            index[key] = os.path.join(directory, f)
    return index

def sofascore_team_pattern(filename):
    """解析sofascore team stats文件名: {match_id}_{team_type}_{team_id}_stats.csv"""
    if not filename.endswith('_stats.csv') or 'all_players' in filename:
        return None
    parts = filename.replace('_stats.csv', '').split('_')
    if len(parts) >= 3:
        try:
            match_id = int(parts[0])
            team_type = parts[1]
            team_id = int(parts[2])
            return (match_id, team_type, team_id)
        except:
            pass
    return None

def sofascore_players_pattern(filename):
    """解析sofascore all_players文件名"""
    if '_all_players_stats.csv' in filename:
        try:
            match_id = int(filename.split('_')[0])
            return match_id
        except:
            pass
    return None

def win007_pattern(filename):
    """解析win007文件名"""
    # 匹配实际的输出文件名格式
    for suffix in ['_handicap_s2d_data.csv', '_overunder_s2d_data.csv', '_euro1x2_s2d_data.csv']:
        if filename.endswith(suffix):
            try:
                match_id = int(filename.replace(suffix, ''))
                file_type = suffix.replace('.csv', '').replace('_s2d_data', '')[1:]
                return (match_id, file_type)
            except:
                pass
    return None


# ============== 批量数据预加载 ==============

def preload_win007_data(win007_dir, match_ids):
    """预加载win007数据到内存"""
    data = {'handicap': {}, 'overunder': {}, 'euro1x2': {}}
    
    match_ids_set = set(match_ids)
    
    for filename in os.listdir(win007_dir):
        parsed = win007_pattern(filename)
        if parsed and parsed[0] in match_ids_set:
            match_id, file_type = parsed
            filepath = os.path.join(win007_dir, filename)
            try:
                df = pd.read_csv(filepath, encoding='utf-8-sig')
                if len(df) > 0:
                    data[file_type][match_id] = df
            except:
                pass
    
    return data

def preload_sofascore_data(sofascore_dir, match_ids):
    """预加载sofascore数据到内存"""
    team_stats = {}  # (match_id, team_type) -> (team_id, df)
    player_stats = {}  # match_id -> df
    
    match_ids_set = set(match_ids)
    
    for filename in os.listdir(sofascore_dir):
        # Team stats
        parsed = sofascore_team_pattern(filename)
        if parsed and parsed[0] in match_ids_set:
            match_id, team_type, team_id = parsed
            filepath = os.path.join(sofascore_dir, filename)
            try:
                df = pd.read_csv(filepath, encoding='utf-8-sig')
                if len(df) > 0:
                    team_stats[(match_id, team_type)] = (team_id, df)
            except:
                pass
            continue
        
        # Player stats
        parsed = sofascore_players_pattern(filename)
        if parsed and parsed in match_ids_set:
            match_id = parsed
            filepath = os.path.join(sofascore_dir, filename)
            try:
                df = pd.read_csv(filepath, encoding='utf-8-sig')
                if len(df) > 0:
                    player_stats[match_id] = df
            except:
                pass
    
    return team_stats, player_stats


# ============== 特征提取函数（使用预加载数据） ==============

def extract_win007_handicap(df, lost_files, match_id, team_type):
    """
    提取让球盘特征

    盘口规则：
    - 原始盘口数据是以主队为视角的让球数（正数=主队让球）
    - 对于主队(home): line取负值（让球为负，受让为正）
    - 对于客队(away): line取正值（让球为负，受让为正）

    例如：主队 让球半(1.5) 客队
    - 主队记录为: -1.5 (主队让1.5球)
    - 客队记录为: +1.5 (客队受让1.5球)

    文件格式 (b2_win007_hdlive_scraper.py 输出):
    - match_id, company, 初盘-主队, 初盘盘口, 初盘-客队, 终盘-主队, 终盘盘口, 终盘-客队
    """
    features = {}
    if df is None:
        lost_files.append(('win007', match_id, f'{match_id}_handicap_s2d_data.csv'))
        return features
    try:
        # 主队视角的让球数转换为该队视角
        sign = -1 if team_type == 'home' else 1

        # 优先选择澳门等主流公司
        priority_companies = ['澳门', '皇冠', 'Pinnacle', '立博', '威廉希尔', 'Bet365']
        selected = None
        for company in priority_companies:
            mask = df['company'].str.contains(company, case=False, na=False)
            if mask.any():
                selected = df[mask].iloc[0]
                break
        if selected is None and len(df) > 0:
            selected = df.iloc[0]

        if selected is not None:
            # 初盘 (Early)
            raw_line = parse_handicap(selected.get('初盘盘口'))
            features['handicap_early_line'] = raw_line * sign if pd.notna(raw_line) else np.nan
            if team_type == 'home':
                features['handicap_early_odds'] = safe_float(selected.get('初盘-主队'))
                features['handicap_early_odds_opponent'] = safe_float(selected.get('初盘-客队'))
            else:
                features['handicap_early_odds'] = safe_float(selected.get('初盘-客队'))
                features['handicap_early_odds_opponent'] = safe_float(selected.get('初盘-主队'))

            # 终盘 (Final)
            raw_line = parse_handicap(selected.get('终盘盘口'))
            features['handicap_final_line'] = raw_line * sign if pd.notna(raw_line) else np.nan
            if team_type == 'home':
                features['handicap_final_odds'] = safe_float(selected.get('终盘-主队'))
                features['handicap_final_odds_opponent'] = safe_float(selected.get('终盘-客队'))
            else:
                features['handicap_final_odds'] = safe_float(selected.get('终盘-客队'))
                features['handicap_final_odds_opponent'] = safe_float(selected.get('终盘-主队'))

            # kickoff使用终盘数据（实际相同）
            features['handicap_kickoff_line'] = features.get('handicap_final_line')
            features['handicap_kickoff_odds'] = features.get('handicap_final_odds')
            features['handicap_kickoff_odds_opponent'] = features.get('handicap_final_odds_opponent')

            # 盘口变化
            el = features.get('handicap_early_line')
            fl = features.get('handicap_final_line')
            if pd.notna(el) and pd.notna(fl):
                features['handicap_line_change'] = fl - el
    except:
        pass
    return features

def extract_win007_overunder(df, lost_files, match_id):
    """
    提取大小球盘特征

    文件格式 (b3_win007_ovlive_scraper.py 输出):
    - match_id, 公司名, 初盘-大球, 初盘-盘, 初盘-小球, 终盘-大球, 终盘-盘, 终盘-小球
    """
    features = {}
    if df is None:
        lost_files.append(('win007', match_id, f'{match_id}_overunder_s2d_data.csv'))
        return features
    try:
        # 优先选择主流公司
        priority_companies = ['澳门', '皇冠', 'Pinnacle', '立博', '威廉希尔', 'Bet365']
        selected = None
        for company in priority_companies:
            mask = df['公司名'].str.contains(company, case=False, na=False)
            if mask.any():
                selected = df[mask].iloc[0]
                break
        if selected is None and len(df) > 0:
            selected = df.iloc[0]

        if selected is not None:
            # 初盘 (Early)
            features['overunder_early_line'] = parse_overunder_line(selected.get('初盘-盘'))
            features['overunder_early_over_odds'] = safe_float(selected.get('初盘-大球'))
            features['overunder_early_under_odds'] = safe_float(selected.get('初盘-小球'))

            # 终盘 (Final)
            features['overunder_final_line'] = parse_overunder_line(selected.get('终盘-盘'))
            features['overunder_final_over_odds'] = safe_float(selected.get('终盘-大球'))
            features['overunder_final_under_odds'] = safe_float(selected.get('终盘-小球'))

            # kickoff使用终盘数据（实际相同）
            features['overunder_kickoff_line'] = features.get('overunder_final_line')
            features['overunder_kickoff_over_odds'] = features.get('overunder_final_over_odds')
            features['overunder_kickoff_under_odds'] = features.get('overunder_final_under_odds')

            # 盘口变化
            el = features.get('overunder_early_line')
            fl = features.get('overunder_final_line')
            if pd.notna(el) and pd.notna(fl):
                features['overunder_line_change'] = fl - el
    except:
        pass
    return features

def extract_win007_euro(df, lost_files, match_id):
    features = {}
    if df is None:
        lost_files.append(('win007', match_id, f'{match_id}_euro1x2_s2d_data.csv'))
        return features
    try:
        priority = ['Pinnacle', 'Bet 365', 'William Hill', 'Bwin', 'Ladbrokes']
        selected = None
        for company in priority:
            mask = df['eu_cp_name'].str.contains(company, case=False, na=False)
            if mask.any():
                selected = df[mask].iloc[0]
                break
        if selected is None and len(df) > 0:
            selected = df.iloc[0]
        
        if selected is not None:
            features['euro_early_home_odds'] = safe_float(selected['home_win_odd0'])
            features['euro_early_draw_odds'] = safe_float(selected['draw_odd0'])
            features['euro_early_away_odds'] = safe_float(selected['away_win_odd0'])
            features['euro_early_return_rate'] = safe_float(selected['return_rate0'])
            features['euro_final_home_odds'] = safe_float(selected['home_win_odd1'])
            features['euro_final_draw_odds'] = safe_float(selected['draw_odd1'])
            features['euro_final_away_odds'] = safe_float(selected['away_win_odd1'])
            features['euro_final_return_rate'] = safe_float(selected['return_rate1'])
            features['euro_early_home_prob'] = safe_float(selected['home_win_p0'])
            features['euro_early_draw_prob'] = safe_float(selected['draw_p0'])
            features['euro_early_away_prob'] = safe_float(selected['away_win_p0'])
            features['euro_final_home_prob'] = safe_float(selected['home_win_p1'])
            features['euro_final_draw_prob'] = safe_float(selected['draw_p1'])
            features['euro_final_away_prob'] = safe_float(selected['away_win_p1'])
            features['euro_kelly_home'] = safe_float(selected['kelly_home'])
            features['euro_kelly_draw'] = safe_float(selected['kelly_draw'])
            features['euro_kelly_away'] = safe_float(selected['kelly_away'])
            
            e_h = features.get('euro_early_home_odds')
            f_h = features.get('euro_final_home_odds')
            if pd.notna(e_h) and pd.notna(f_h):
                features['euro_home_odds_change'] = f_h - e_h
                features['euro_draw_odds_change'] = features['euro_final_draw_odds'] - features['euro_early_draw_odds']
                features['euro_away_odds_change'] = features['euro_final_away_odds'] - features['euro_early_away_odds']
    except:
        pass
    return features

def extract_sofascore_team(team_id, df, lost_files, match_id, team_type):
    features = {}
    if df is None:
        lost_files.append(('sofascore', match_id, f'{match_id}_{team_type}_{team_id}_stats.csv'))
        return features
    try:
        row = df.iloc[0]
        features['xG'] = safe_float(row.get('Match_overview_Expected_goals'))
        features['total_shots'] = safe_float(row.get('Shots_Total_shots'))
        features['shots_on_target'] = safe_float(row.get('Shots_Shots_on_target'))
        features['shots_inside_box'] = safe_float(row.get('Shots_Shots_inside_box'))
        features['shots_outside_box'] = safe_float(row.get('Shots_Shots_outside_box'))
        features['blocked_shots'] = safe_float(row.get('Shots_Blocked_shots'))
        features['big_chances'] = safe_float(row.get('Match_overview_Big_chances'))
        features['big_chances_scored'] = safe_float(row.get('Attack_Big_chances_scored'))
        features['big_chances_missed'] = safe_float(row.get('Attack_Big_chances_missed'))
        features['ball_possession'] = parse_percentage(row.get('Match_overview_Ball_possession'))
        features['total_passes'] = safe_float(row.get('Match_overview_Passes'))
        
        ap = row.get('Passes_Accurate_passes')
        if pd.notna(ap):
            num, _, pct = parse_ratio(str(ap))
            features['accurate_passes'] = num
            features['pass_accuracy'] = pct
        
        features['touches_in_box'] = safe_float(row.get('Attack_Touches_in_penalty_area'))
        features['through_balls'] = safe_float(row.get('Attack_Through_balls'))
        features['final_third_entries'] = safe_float(row.get('Passes_Final_third_entries'))
        features['corner_kicks'] = safe_float(row.get('Match_overview_Corner_kicks'))
        features['goalkeeper_saves'] = safe_float(row.get('Match_overview_Goalkeeper_saves'))
        features['total_saves'] = safe_float(row.get('Goalkeeping_Total_saves'))
        features['goals_prevented'] = safe_float(row.get('Goalkeeping_Goals_prevented'))
        features['tackles'] = safe_float(row.get('Match_overview_Tackles'))
        features['total_tackles'] = safe_float(row.get('Defending_Total_tackles'))
        
        tw = row.get('Defending_Tackles_won')
        if pd.notna(tw):
            s = str(tw)
            features['tackles_won_pct'] = parse_percentage(s.split('/')[0] if '/' in s else s)
        
        features['interceptions'] = safe_float(row.get('Defending_Interceptions'))
        features['clearances'] = safe_float(row.get('Defending_Clearances'))
        features['recoveries'] = safe_float(row.get('Defending_Recoveries'))
        features['errors_to_shot'] = safe_float(row.get('Defending_Errors_lead_to_a_shot'))
        features['duels_won_pct'] = parse_percentage(row.get('Duels_Duels'))
        
        gd = row.get('Duels_Ground_duels')
        if pd.notna(gd):
            num, _, pct = parse_ratio(str(gd))
            features['ground_duels_won'] = num
            features['ground_duels_pct'] = pct
        
        ad = row.get('Duels_Aerial_duels')
        if pd.notna(ad):
            num, _, pct = parse_ratio(str(ad))
            features['aerial_duels_won'] = num
            features['aerial_duels_pct'] = pct
        
        dr = row.get('Duels_Dribbles')
        if pd.notna(dr):
            s = str(dr)
            if '(' in s:
                num, _, pct = parse_ratio(s)
                features['dribbles_successful'] = num
                features['dribbles_pct'] = pct
            elif '/' in s:
                features['dribbles_pct'] = parse_percentage(s.split('/')[0])
        
        features['fouls'] = safe_float(row.get('Match_overview_Fouls'))
        features['yellow_cards'] = safe_float(row.get('Match_overview_Yellow_cards'))
        
        lb = row.get('Passes_Long_balls')
        if pd.notna(lb):
            num, _, pct = parse_ratio(str(lb))
            features['long_balls_accurate'] = num
            features['long_balls_pct'] = pct
    except:
        pass
    return features

def extract_sofascore_players(df, team_type, lost_files, match_id):
    features = {}
    if df is None:
        lost_files.append(('sofascore', match_id, f'{match_id}_all_players_stats.csv'))
        return features
    try:
        team_df = df[df['team'] == team_type].copy()
        if len(team_df) == 0:
            return features
        
        for col in ['Rating', 'Goals', 'Expected_goals_xG', 'Total_shots', 'Shots_on_target',
                   'Key_passes', 'Pass_accuracy_pct', 'Duels_won', 'Duels_lost',
                   'Tackles_total', 'Interceptions', 'Clearances']:
            if col in team_df.columns:
                team_df[col] = pd.to_numeric(team_df[col], errors='coerce')
        
        features['team_avg_rating'] = team_df['Rating'].mean()
        features['team_max_rating'] = team_df['Rating'].max()
        features['team_min_rating'] = team_df['Rating'].min()
        features['team_total_shots'] = team_df['Total_shots'].sum()
        features['team_total_xG'] = team_df['Expected_goals_xG'].sum()
        features['team_shots_on_target'] = team_df['Shots_on_target'].sum()
        features['team_total_key_passes'] = team_df['Key_passes'].sum()
        features['team_avg_pass_accuracy'] = team_df['Pass_accuracy_pct'].mean()
        features['team_total_tackles'] = team_df['Tackles_total'].sum()
        features['team_total_interceptions'] = team_df['Interceptions'].sum()
        features['team_total_clearances'] = team_df['Clearances'].sum()
        features['team_total_duels_won'] = team_df['Duels_won'].sum()
        features['team_total_duels_lost'] = team_df['Duels_lost'].sum()
        
        total_duels = features['team_total_duels_won'] + features['team_total_duels_lost']
        if pd.notna(total_duels) and total_duels > 0:
            features['team_duels_win_rate'] = features['team_total_duels_won'] / total_duels
        
        # Key players
        if 'Expected_goals_xG' in team_df.columns:
            valid = team_df[(team_df['Expected_goals_xG'].notna()) & (team_df['Expected_goals_xG'] > 0)]
            if len(valid) > 0:
                top = valid.nlargest(1, 'Expected_goals_xG').iloc[0]
                features['key_attacker_name'] = top['player_name']
                features['key_attacker_xG'] = top['Expected_goals_xG']
                features['key_attacker_shots'] = top['Total_shots']
                features['key_attacker_rating'] = top['Rating']
        
        if 'Rating' in team_df.columns:
            valid = team_df[team_df['Rating'].notna()]
            if len(valid) > 0:
                mvp = valid.nlargest(1, 'Rating').iloc[0]
                features['mvp_name'] = mvp['player_name']
                features['mvp_rating'] = mvp['Rating']
                features['mvp_goals'] = mvp['Goals']
        
        if 'Key_passes' in team_df.columns:
            valid = team_df[(team_df['Key_passes'].notna()) & (team_df['Key_passes'] > 0)]
            if len(valid) > 0:
                top = valid.nlargest(1, 'Key_passes').iloc[0]
                features['key_passer_name'] = top['player_name']
                features['key_passer_key_passes'] = top['Key_passes']
                features['key_passer_pass_accuracy'] = top['Pass_accuracy_pct']
        
        if 'Tackles_total' in team_df.columns and 'Interceptions' in team_df.columns:
            team_df['def_actions'] = team_df['Tackles_total'].fillna(0) + team_df['Interceptions'].fillna(0)
            valid = team_df[team_df['def_actions'] > 0]
            if len(valid) > 0:
                top = valid.nlargest(1, 'def_actions').iloc[0]
                features['key_defender_name'] = top['player_name']
                features['key_defender_tackles'] = top['Tackles_total']
                features['key_defender_interceptions'] = top['Interceptions']
                features['key_defender_rating'] = top['Rating']
        
        if 'Duels_won' in team_df.columns:
            valid = team_df[(team_df['Duels_won'].notna()) & (team_df['Duels_won'] > 0)]
            if len(valid) > 0:
                top = valid.nlargest(1, 'Duels_won').iloc[0]
                features['key_dueler_name'] = top['player_name']
                features['key_dueler_won'] = top['Duels_won']
                features['key_dueler_lost'] = top['Duels_lost']
                total = top['Duels_won'] + top['Duels_lost']
                if pd.notna(total) and total > 0:
                    features['key_dueler_win_rate'] = top['Duels_won'] / total
    except:
        pass
    return features


# ============== 批量处理函数 ==============

def process_batch(batch_data):
    """处理一批比赛数据"""
    matches, win007_data, sofascore_team_data, sofascore_player_data = batch_data

    results = []
    lost_files = []

    for match_row in matches:
        # 解析match_id - 使用带前缀的列名
        sofascore_match_id_raw = match_row.get('sofascore_match_id', '')
        if pd.isna(sofascore_match_id_raw) or sofascore_match_id_raw == '':
            continue
        sofascore_match_id = int(float(str(sofascore_match_id_raw).replace('\ufeff', '').strip()))

        # Win007 match_id 也是带前缀的
        win007_match_id = match_row.get('win007_match_id')

        for team_type in ['home', 'away']:
            features = {}

            features['sofascore_match_id'] = sofascore_match_id
            features['win007_match_id'] = int(win007_match_id) if pd.notna(win007_match_id) else 0

            # Team ID and info - 使用带前缀的列名
            team_data = sofascore_team_data.get((sofascore_match_id, team_type))
            team_id = team_data[0] if team_data else None
            features['team_id'] = team_id if team_id else 0

            if team_type == 'home':
                features['team_name'] = match_row.get('sofascore_home_team', '')
                features['is_home'] = 1
            else:
                features['team_name'] = match_row.get('sofascore_away_team', '')
                features['is_home'] = 0

            features['date'] = match_row.get('sofascore_date', '')
            features['competition'] = match_row.get('sofascore_competition', '')
            features['season'] = match_row.get('sofascore_season', '')

            # Goals - 使用带前缀的列名
            home_goals = safe_float(match_row.get('sofascore_home_goals'))
            away_goals = safe_float(match_row.get('sofascore_away_goals'))
            if team_type == 'home':
                features['goals_scored'] = home_goals
                features['goals_conceded'] = away_goals
                if pd.notna(home_goals) and pd.notna(away_goals):
                    features['goal_diff'] = home_goals - away_goals
            else:
                features['goals_scored'] = away_goals
                features['goals_conceded'] = home_goals
                if pd.notna(home_goals) and pd.notna(away_goals):
                    features['goal_diff'] = away_goals - home_goals
            if pd.notna(home_goals) and pd.notna(away_goals):
                features['total_goals'] = home_goals + away_goals

            # Win007 features
            if pd.notna(win007_match_id):
                w_id = int(win007_match_id)

                hf = extract_win007_handicap(win007_data['handicap'].get(w_id), lost_files, w_id, team_type)
                for k, v in hf.items():
                    features[f'win007_{k}'] = v

                of = extract_win007_overunder(win007_data['overunder'].get(w_id), lost_files, w_id)
                for k, v in of.items():
                    features[f'win007_{k}'] = v

                ef = extract_win007_euro(win007_data['euro1x2'].get(w_id), lost_files, w_id)
                for k, v in ef.items():
                    features[f'win007_{k}'] = v

                # Handicap result
                hl = features.get('win007_handicap_final_line', features.get('win007_handicap_kickoff_line'))
                gd = features.get('goal_diff')
                if pd.notna(hl) and pd.notna(gd):
                    actual = gd + hl
                    features['handicap_result'] = 1 if actual > 0.25 else (-1 if actual < -0.25 else 0)

                # Over/under result
                ol = features.get('win007_overunder_final_line', features.get('win007_overunder_kickoff_line'))
                tg = features.get('total_goals')
                if pd.notna(ol) and pd.notna(tg):
                    diff = tg - ol
                    features['overunder_result'] = 1 if diff > 0.25 else (-1 if diff < -0.25 else 0)

            # Sofascore team features
            if team_data:
                tf = extract_sofascore_team(team_id, team_data[1], lost_files, sofascore_match_id, team_type)
                for k, v in tf.items():
                    features[f'sofascore_{k}'] = v

            # Sofascore player features
            player_df = sofascore_player_data.get(sofascore_match_id)
            pf = extract_sofascore_players(player_df, team_type, lost_files, sofascore_match_id)
            for k, v in pf.items():
                features[f'sofascore_{k}'] = v

            results.append(features)

    return results, lost_files


def main():
    parser = argparse.ArgumentParser(
        description='Football Match Feature Engineering (SofaScore + Win007)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python c0_feature_engineering.py \\
    --sofascore-dir data/sofascore \\
    --win007-dir data/win007 \\
    --matches-file data/matches/mapped_matches.csv \\
    --output-dir output
        """
    )
    parser.add_argument('--sofascore-dir', type=str, required=True, help='SofaScore数据目录')
    parser.add_argument('--win007-dir', type=str, required=True, help='Win007数据目录')
    parser.add_argument('--matches-file', type=str, required=True, help='比赛映射CSV文件')
    parser.add_argument('--output-dir', type=str, default='./output', help='输出目录 (默认: ./output)')
    parser.add_argument('--batch-size', type=int, default=500, help='批处理大小 (默认: 500)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    # 读取matches
    print("Loading matches file...")
    matches_df = pd.read_csv(args.matches_file, encoding='utf-8-sig')
    matches_df.columns = matches_df.columns.str.strip().str.replace('\ufeff', '')
    print(f"Total matches: {len(matches_df)}")

    # 收集所有需要的match_id - 使用带前缀的列名
    print("Collecting match IDs...")
    sofascore_ids = set()
    win007_ids = set()

    for _, row in matches_df.iterrows():
        # SofaScore match_id
        sf_mid = row.get('sofascore_match_id')
        if pd.notna(sf_mid):
            mid_raw = str(sf_mid).replace('\ufeff', '').strip()
            sofascore_ids.add(int(float(mid_raw)))
        # Win007 match_id
        w_mid = row.get('win007_match_id')
        if pd.notna(w_mid):
            win007_ids.add(int(w_mid))

    # 预加载所有数据到内存
    print(f"Preloading win007 data ({len(win007_ids)} matches)...")
    t0 = time.time()
    win007_data = preload_win007_data(args.win007_dir, win007_ids)
    print(f"  Loaded in {time.time()-t0:.1f}s - handicap:{len(win007_data['handicap'])}, overunder:{len(win007_data['overunder'])}, euro:{len(win007_data['euro1x2'])}")

    print(f"Preloading sofascore data ({len(sofascore_ids)} matches)...")
    t0 = time.time()
    sofascore_team_data, sofascore_player_data = preload_sofascore_data(args.sofascore_dir, sofascore_ids)
    print(f"  Loaded in {time.time()-t0:.1f}s - team_stats:{len(sofascore_team_data)}, player_stats:{len(sofascore_player_data)}")

    # 转换为字典列表（便于处理）
    matches_list = matches_df.to_dict('records')

    # 单进程处理（避免多进程序列化开销）
    print("\nProcessing matches...")
    all_features = []
    all_lost_files = []

    batch_size = args.batch_size
    total_batches = (len(matches_list) + batch_size - 1) // batch_size

    for i in range(0, len(matches_list), batch_size):
        batch = matches_list[i:i+batch_size]
        batch_num = i // batch_size + 1

        results, lost = process_batch((batch, win007_data, sofascore_team_data, sofascore_player_data))
        all_features.extend(results)
        all_lost_files.extend(lost)

        if batch_num % 5 == 0 or batch_num == total_batches:
            elapsed = time.time() - start_time
            pct = batch_num / total_batches * 100
            print(f"  Batch {batch_num}/{total_batches} ({pct:.1f}%) - {elapsed:.1f}s elapsed")

    print(f"\nCreating output DataFrame...")
    wide_table = pd.DataFrame(all_features)

    # 定义完整的列顺序（与wide_table.csv参考一致，去除understat_match_id）
    expected_columns = [
        'sofascore_match_id', 'win007_match_id', 'team_id', 'team_name', 'is_home',
        'date', 'competition', 'season', 'goals_scored', 'goals_conceded', 'goal_diff', 'total_goals',
        # Win007 handicap
        'win007_handicap_early_line', 'win007_handicap_early_odds', 'win007_handicap_early_odds_opponent',
        'win007_handicap_final_line', 'win007_handicap_final_odds', 'win007_handicap_final_odds_opponent',
        'win007_handicap_kickoff_line', 'win007_handicap_kickoff_odds', 'win007_handicap_kickoff_odds_opponent',
        'win007_handicap_line_change',
        # Win007 overunder
        'win007_overunder_early_line', 'win007_overunder_early_over_odds', 'win007_overunder_early_under_odds',
        'win007_overunder_final_line', 'win007_overunder_final_over_odds', 'win007_overunder_final_under_odds',
        'win007_overunder_kickoff_line', 'win007_overunder_kickoff_over_odds', 'win007_overunder_kickoff_under_odds',
        'win007_overunder_line_change',
        # Win007 euro
        'win007_euro_early_home_odds', 'win007_euro_early_draw_odds', 'win007_euro_early_away_odds', 'win007_euro_early_return_rate',
        'win007_euro_final_home_odds', 'win007_euro_final_draw_odds', 'win007_euro_final_away_odds', 'win007_euro_final_return_rate',
        'win007_euro_early_home_prob', 'win007_euro_early_draw_prob', 'win007_euro_early_away_prob',
        'win007_euro_final_home_prob', 'win007_euro_final_draw_prob', 'win007_euro_final_away_prob',
        'win007_euro_kelly_home', 'win007_euro_kelly_draw', 'win007_euro_kelly_away',
        'win007_euro_home_odds_change', 'win007_euro_draw_odds_change', 'win007_euro_away_odds_change',
        # Result
        'handicap_result', 'overunder_result',
        # Sofascore team stats
        'sofascore_xG', 'sofascore_total_shots', 'sofascore_shots_on_target', 'sofascore_shots_inside_box',
        'sofascore_shots_outside_box', 'sofascore_blocked_shots', 'sofascore_big_chances', 'sofascore_big_chances_scored',
        'sofascore_big_chances_missed', 'sofascore_ball_possession', 'sofascore_total_passes', 'sofascore_accurate_passes',
        'sofascore_pass_accuracy', 'sofascore_touches_in_box', 'sofascore_through_balls', 'sofascore_final_third_entries',
        'sofascore_corner_kicks', 'sofascore_goalkeeper_saves', 'sofascore_total_saves', 'sofascore_goals_prevented',
        'sofascore_tackles', 'sofascore_total_tackles', 'sofascore_tackles_won_pct', 'sofascore_interceptions',
        'sofascore_clearances', 'sofascore_recoveries', 'sofascore_errors_to_shot', 'sofascore_duels_won_pct',
        'sofascore_ground_duels_won', 'sofascore_ground_duels_pct', 'sofascore_aerial_duels_won', 'sofascore_aerial_duels_pct',
        'sofascore_dribbles_successful', 'sofascore_dribbles_pct', 'sofascore_fouls', 'sofascore_yellow_cards',
        'sofascore_long_balls_accurate', 'sofascore_long_balls_pct',
        # Sofascore player aggregates
        'sofascore_team_avg_rating', 'sofascore_team_max_rating', 'sofascore_team_min_rating',
        'sofascore_team_total_shots', 'sofascore_team_total_xG', 'sofascore_team_shots_on_target',
        'sofascore_team_total_key_passes', 'sofascore_team_avg_pass_accuracy', 'sofascore_team_total_tackles',
        'sofascore_team_total_interceptions', 'sofascore_team_total_clearances', 'sofascore_team_total_duels_won',
        'sofascore_team_total_duels_lost', 'sofascore_team_duels_win_rate',
        # Sofascore key players
        'sofascore_key_attacker_name', 'sofascore_key_attacker_xG', 'sofascore_key_attacker_shots', 'sofascore_key_attacker_rating',
        'sofascore_mvp_name', 'sofascore_mvp_rating', 'sofascore_mvp_goals',
        'sofascore_key_passer_name', 'sofascore_key_passer_key_passes', 'sofascore_key_passer_pass_accuracy',
        'sofascore_key_defender_name', 'sofascore_key_defender_tackles', 'sofascore_key_defender_interceptions', 'sofascore_key_defender_rating',
        'sofascore_key_dueler_name', 'sofascore_key_dueler_won', 'sofascore_key_dueler_lost', 'sofascore_key_dueler_win_rate',
    ]

    # 确保所有期望的列都存在
    for col in expected_columns:
        if col not in wide_table.columns:
            wide_table[col] = np.nan

    # 按预期顺序重排列（额外列保留在末尾）
    extra_cols = [c for c in wide_table.columns if c not in expected_columns]
    wide_table = wide_table[expected_columns + extra_cols]

    for col in ['sofascore_match_id', 'win007_match_id', 'team_id']:
        if col in wide_table.columns:
            wide_table[col] = wide_table[col].fillna(0).astype(int)

    # 保存结果 - 输出为 inc_wide_table.csv
    wide_table_path = os.path.join(args.output_dir, 'inc_wide_table.csv')
    wide_table.to_csv(wide_table_path, index=False, encoding='utf-8-sig')
    print(f"Inc wide table saved: {wide_table_path}")
    print(f"Shape: {wide_table.shape}")
    print(f"Expected columns: {len(expected_columns)}, Total columns: {len(wide_table.columns)}")

    # Lost files
    if all_lost_files:
        lost_df = pd.DataFrame(all_lost_files, columns=['site', 'match_id', 'lost_file']).drop_duplicates()
        lost_df['match_id'] = lost_df['match_id'].astype(int)
    else:
        lost_df = pd.DataFrame(columns=['site', 'match_id', 'lost_file'])

    lost_path = os.path.join(args.output_dir, 'lost.csv')
    lost_df.to_csv(lost_path, index=False, encoding='utf-8-sig')
    print(f"Lost files: {lost_path} ({len(lost_df)} records)")

    total_time = time.time() - start_time
    print(f"\n✓ Completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")


if __name__ == '__main__':
    main()
