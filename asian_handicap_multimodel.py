"""
亚洲让球盘投注模型 - 多盘口专项模型版
针对6类盘口范围分别构建专项模型：0, ±0.25, ±0.5, ±0.75, ±1, ±1.25
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("亚洲让球盘投注模型 - 6类盘口专项模型")
print("=" * 70)

# 1. 数据加载
print("\n" + "=" * 70)
print("1. 加载和预处理数据")
print("=" * 70)

df = pd.read_csv('wide_table.csv')
print(f"原始数据形状: {df.shape}")

df['date'] = pd.to_datetime(df['date'])

win007_required_cols = [
    'win007_handicap_kickoff_line',
    'win007_handicap_kickoff_odds',
    'win007_handicap_kickoff_odds_opponent',
]

df_clean = df.dropna(subset=win007_required_cols)
print(f"过滤win007让球盘缺失后: {df_clean.shape}")

df_clean = df_clean.dropna(subset=['handicap_result'])
print(f"过滤handicap_result缺失后: {df_clean.shape}")

# 定义6类盘口范围
HANDICAP_RANGES = {
    '0': [0.0],                    # 平手盘
    '0.25': [0.25, -0.25],         # 平半盘
    '0.5': [0.5, -0.5],            # 半球盘
    '0.75': [0.75, -0.75],         # 半一盘
    '1': [1.0, -1.0],              # 一球盘
    '1.25': [1.25, -1.25],         # 球半盘
}

# 筛选指定盘口
valid_lines = []
for lines in HANDICAP_RANGES.values():
    valid_lines.extend(lines)

def is_valid_line(line):
    """判断是否为有效盘口"""
    for valid in valid_lines:
        if abs(line - valid) < 0.001:
            return True
    return False

df_filtered = df_clean[df_clean['win007_handicap_kickoff_line'].apply(is_valid_line)].copy()
print(f"\n筛选6类盘口后: {df_filtered.shape}")

# 盘口分布
print("\n盘口分布:")
line_counts = df_filtered['win007_handicap_kickoff_line'].value_counts().sort_index()
for line, count in line_counts.items():
    print(f"  {line:+.2f}: {count}")

# 排除表现差的联赛
BAD_LEAGUES = [
    'Championship',
    '2. Bundesliga',
    'Club Friendly Games',
    'Ligue 1',
    'Ligue 2',
    'LaLiga 2',
]

df_filtered = df_filtered[~df_filtered['competition'].isin(BAD_LEAGUES)]
print(f"\n过滤表现差联赛后: {df_filtered.shape}")

df_filtered = df_filtered.sort_values(['date', 'sofascore_match_id'])

train_start = datetime(2023, 6, 1)
train_cutoff = datetime(2025, 3, 1)
train_df = df_filtered[(df_filtered['date'] >= train_start) & (df_filtered['date'] <= train_cutoff)].copy()
test_df = df_filtered[df_filtered['date'] > train_cutoff].copy()

print(f"\n训练集: {len(train_df)} 条记录")
print(f"测试集: {len(test_df)} 条记录")

# 2. 特征工程
print("\n" + "=" * 70)
print("2. 特征工程")
print("=" * 70)

key_stats = [
    'sofascore_xG', 'sofascore_total_shots', 'sofascore_shots_on_target',
    'sofascore_big_chances', 'sofascore_ball_possession',
    'sofascore_pass_accuracy', 'sofascore_corner_kicks',
    'sofascore_goalkeeper_saves', 'sofascore_tackles',
    'sofascore_duels_won_pct', 'sofascore_team_avg_rating',
    'goals_scored', 'goals_conceded', 'goal_diff',
]


def build_team_features(df_all, target_date, team_id, is_home, n_matches=7):
    """构建球队历史特征"""
    team_matches = df_all[
        (df_all['team_id'] == team_id) &
        (df_all['date'] < target_date)
    ].sort_values('date', ascending=False).head(n_matches)

    if len(team_matches) < 3:
        return None

    features = {}
    weights = np.exp(-np.arange(len(team_matches)) * 0.15)
    weights = weights / weights.sum()

    for col in key_stats:
        if col in team_matches.columns:
            values = team_matches[col].values
            valid_mask = ~pd.isna(values)
            if valid_mask.sum() >= 2:
                valid_values = values[valid_mask].astype(float)
                valid_weights = weights[valid_mask]
                valid_weights = valid_weights / valid_weights.sum()
                features[f'{col}_wavg'] = np.average(valid_values, weights=valid_weights)
                features[f'{col}_std'] = np.std(valid_values) if len(valid_values) > 1 else 0

    features['handicap_win_rate'] = (team_matches['handicap_result'] == 1).mean()
    features['handicap_draw_rate'] = (team_matches['handicap_result'] == 0).mean()

    if is_home == 1:
        home_matches = team_matches[team_matches['is_home'] == 1]
        if len(home_matches) >= 2:
            features['venue_handicap_win_rate'] = (home_matches['handicap_result'] == 1).mean()
            features['venue_goals_avg'] = home_matches['goals_scored'].mean()
    else:
        away_matches = team_matches[team_matches['is_home'] == 0]
        if len(away_matches) >= 2:
            features['venue_handicap_win_rate'] = (away_matches['handicap_result'] == 1).mean()
            features['venue_goals_avg'] = away_matches['goals_scored'].mean()

    features['n_matches'] = len(team_matches)
    return features


def build_match_features(row, opponent_row, df_all):
    """构建比赛特征"""
    features = {}

    features['handicap_line'] = row['win007_handicap_kickoff_line']
    features['handicap_odds'] = row['win007_handicap_kickoff_odds']
    features['handicap_odds_opponent'] = row['win007_handicap_kickoff_odds_opponent']

    odds1 = row['win007_handicap_kickoff_odds']
    odds2 = row['win007_handicap_kickoff_odds_opponent']
    features['implied_prob'] = 1 / (1 + odds1)
    features['odds_ratio'] = odds1 / odds2 if odds2 > 0 else 1

    if pd.notna(row.get('win007_handicap_early_odds')):
        features['early_odds'] = row['win007_handicap_early_odds']
        features['odds_drift'] = row['win007_handicap_kickoff_odds'] - row['win007_handicap_early_odds']

    # ========== 盘口时序特征 (Line Movement Features) ==========
    # 盘口变化: kickoff_line - early_line
    early_line = row.get('win007_handicap_early_line')
    kickoff_line = row.get('win007_handicap_kickoff_line')
    if pd.notna(early_line) and pd.notna(kickoff_line):
        features['line_movement'] = kickoff_line - early_line
        # 盘口变化方向: 1=升盘(让球增加), -1=降盘, 0=不变
        if abs(features['line_movement']) < 0.001:
            features['line_move_direction'] = 0
        elif features['line_movement'] > 0:
            features['line_move_direction'] = 1
        else:
            features['line_move_direction'] = -1

    # 赔率变化: kickoff_odds - early_odds
    early_odds = row.get('win007_handicap_early_odds')
    kickoff_odds = row.get('win007_handicap_kickoff_odds')
    if pd.notna(early_odds) and pd.notna(kickoff_odds):
        features['odds_movement'] = kickoff_odds - early_odds
        # 赔率变化率
        if early_odds > 0:
            features['odds_movement_pct'] = (kickoff_odds - early_odds) / early_odds

    # 对手赔率变化
    early_odds_opp = row.get('win007_handicap_early_odds_opponent')
    kickoff_odds_opp = row.get('win007_handicap_kickoff_odds_opponent')
    if pd.notna(early_odds_opp) and pd.notna(kickoff_odds_opp):
        features['odds_movement_opponent'] = kickoff_odds_opp - early_odds_opp

    # 使用已有的盘口变化字段
    if pd.notna(row.get('win007_handicap_line_change')):
        features['line_change_official'] = row['win007_handicap_line_change']

    # 盘口-赔率背离特征: 盘口升但赔率降(或反之)可能暗示市场分歧
    if 'line_movement' in features and 'odds_movement' in features:
        line_up = features['line_movement'] > 0.001
        odds_down = features['odds_movement'] < -0.01
        line_down = features['line_movement'] < -0.001
        odds_up = features['odds_movement'] > 0.01
        # 背离信号: 盘口升但赔率降, 或盘口降但赔率升
        features['line_odds_divergence'] = 1 if (line_up and odds_down) or (line_down and odds_up) else 0

    if pd.notna(row.get('win007_euro_final_home_prob')):
        if row['is_home'] == 1:
            features['euro_win_prob'] = row['win007_euro_final_home_prob']
            features['euro_draw_prob'] = row['win007_euro_final_draw_prob']
        else:
            features['euro_win_prob'] = row['win007_euro_final_away_prob']
            features['euro_draw_prob'] = row['win007_euro_final_draw_prob']

    for col in ['win007_euro_kelly_home', 'win007_euro_kelly_draw', 'win007_euro_kelly_away']:
        if pd.notna(row.get(col)):
            features[col] = row[col]

    if pd.notna(row.get('win007_overunder_kickoff_line')):
        features['overunder_line'] = row['win007_overunder_kickoff_line']

    features['is_home'] = row['is_home']

    team_hist = build_team_features(df_all, row['date'], row['team_id'], row['is_home'])
    if team_hist is None:
        return None

    for k, v in team_hist.items():
        features[f'team_{k}'] = v

    if opponent_row is not None:
        opp_hist = build_team_features(df_all, row['date'], opponent_row['team_id'], opponent_row['is_home'])
        if opp_hist:
            for k, v in opp_hist.items():
                features[f'opp_{k}'] = v

            for stat in key_stats:
                team_feat = f'team_{stat}_wavg'
                opp_feat = f'opp_{stat}_wavg'
                if team_feat in features and opp_feat in features:
                    features[f'diff_{stat}'] = features[team_feat] - features[opp_feat]

    return features


def get_handicap_range(line):
    """获取盘口所属范围"""
    abs_line = abs(line)
    if abs_line < 0.001:
        return '0'
    elif abs(abs_line - 0.25) < 0.001:
        return '0.25'
    elif abs(abs_line - 0.5) < 0.001:
        return '0.5'
    elif abs(abs_line - 0.75) < 0.001:
        return '0.75'
    elif abs(abs_line - 1.0) < 0.001:
        return '1'
    elif abs(abs_line - 1.25) < 0.001:
        return '1.25'
    return None


def build_dataset(df_source, df_all_for_history, dataset_name):
    """构建数据集"""
    features_list = []
    labels_list = []
    info_list = []

    for match_id in df_source['sofascore_match_id'].unique():
        match_records = df_source[df_source['sofascore_match_id'] == match_id]
        if len(match_records) < 2:
            continue

        for idx, row in match_records.iterrows():
            opponent = match_records[match_records['team_id'] != row['team_id']]
            opponent_row = opponent.iloc[0] if len(opponent) > 0 else None

            feat = build_match_features(row, opponent_row, df_all_for_history)
            if feat is not None:
                features_list.append(feat)
                labels_list.append(row['handicap_result'])
                info_list.append({
                    'date': row['date'],
                    'match_id': match_id,
                    'team_id': row['team_id'],
                    'team_name': row['team_name'],
                    'is_home': row['is_home'],
                    'handicap_line': row['win007_handicap_kickoff_line'],
                    'handicap_odds': row['win007_handicap_kickoff_odds'],
                    'handicap_odds_opponent': row['win007_handicap_kickoff_odds_opponent'],
                    'handicap_result': row['handicap_result'],
                    'competition': row['competition'],
                    'goal_diff': row['goal_diff'],
                    'handicap_range': get_handicap_range(row['win007_handicap_kickoff_line']),
                })

    print(f"  {dataset_name}: {len(features_list)} 样本")
    return features_list, labels_list, info_list


print("\n构建数据集...")
train_features, train_labels, train_info = build_dataset(train_df, df_clean, "训练集")
test_features, test_labels, test_info = build_dataset(test_df, df_clean, "测试集")

# 3. 模型训练
print("\n" + "=" * 70)
print("3. 分盘口模型训练")
print("=" * 70)

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ODDS_MARKUP = 1.015

# 各盘口专用参数 (优化版 - 聚焦受让方)
# 分析发现：受让方(正盘口)ROI明显更高，让球方(负盘口)表现差
RANGE_PARAMS = {
    '0': {'min_edge': 0.10, 'max_edge': 0.18, 'min_odds': 0.85, 'max_odds': 1.12, 'vt': 0.10, 'only_positive': False},
    '0.25': {'min_edge': 0.12, 'max_edge': 0.20, 'min_odds': 0.82, 'max_odds': 1.15, 'vt': 0.12, 'only_positive': True},
    '0.5': {'min_edge': 0.08, 'max_edge': 0.22, 'min_odds': 0.80, 'max_odds': 1.18, 'vt': 0.08, 'only_positive': True},
    '0.75': {'min_edge': 0.10, 'max_edge': 0.18, 'min_odds': 0.85, 'max_odds': 1.12, 'vt': 0.10, 'only_positive': True},
    '1': {'min_edge': 0.10, 'max_edge': 0.18, 'min_odds': 0.85, 'max_odds': 1.12, 'vt': 0.10, 'only_positive': True},
    '1.25': {'min_edge': 0.10, 'max_edge': 0.18, 'min_odds': 0.85, 'max_odds': 1.12, 'vt': 0.10, 'only_positive': True},
}


def calculate_handicap_outcome(goal_diff, handicap_line, bet_direction):
    """计算亚洲让球盘的实际结果"""
    if bet_direction == 'win':
        result = goal_diff + handicap_line
    else:
        result = -(goal_diff + handicap_line)

    remainder = abs(handicap_line) % 0.5
    is_compound = abs(remainder - 0.25) < 0.001

    if is_compound:
        if result > 0.5:
            return 'full_win', 1.0
        elif result > 0 and result <= 0.5:
            return 'half_win', 0.5
        elif abs(result) < 0.001:
            return 'push', 0.0
        elif result >= -0.5 and result < 0:
            return 'half_lose', -0.5
        else:
            return 'full_lose', -1.0
    else:
        if result > 0.001:
            return 'full_win', 1.0
        elif abs(result) < 0.001:
            return 'push', 0.0
        else:
            return 'full_lose', -1.0


def calculate_value_betting_roi(model_probs, y_true, info_list, params):
    """基于价值投注策略计算ROI"""
    total_bet = 0
    total_return = 0
    bet_records = []

    min_edge = params['min_edge']
    max_edge = params['max_edge']
    min_odds = params['min_odds']
    max_odds = params['max_odds']
    vt = params['vt']
    only_positive = params.get('only_positive', False)

    for i, (prob, actual, info) in enumerate(zip(model_probs, y_true, info_list)):
        # 如果只投注正盘口（受让方），跳过负盘口
        if only_positive and info['handicap_line'] < -0.001:
            continue
        odds_win = info['handicap_odds'] * ODDS_MARKUP
        odds_lose = info['handicap_odds_opponent'] * ODDS_MARKUP

        market_prob_win = 1 / (1 + info['handicap_odds'])
        market_prob_lose = 1 / (1 + info['handicap_odds_opponent'])

        bet_made = False
        bet_direction = None
        bet_odds = None
        bet_result = None
        profit = 0
        edge = 0

        goal_diff = info.get('goal_diff', 0)

        if prob > market_prob_win + vt:
            edge = prob - market_prob_win

            if edge < min_edge or edge > max_edge:
                continue
            if odds_win < min_odds or odds_win > max_odds:
                continue

            total_bet += 1
            bet_made = True
            bet_direction = 'win'
            bet_odds = odds_win

            bet_result, profit_mult = calculate_handicap_outcome(
                goal_diff, info['handicap_line'], 'win'
            )

            if profit_mult > 0:
                profit = profit_mult * odds_win
                total_return += 1 + profit
            elif profit_mult == 0:
                profit = 0
                total_return += 1
            else:
                profit = profit_mult
                total_return += 1 + profit

        elif (1 - prob) > market_prob_lose + vt:
            edge = (1 - prob) - market_prob_lose

            if edge < min_edge or edge > max_edge:
                continue
            if odds_lose < min_odds or odds_lose > max_odds:
                continue

            total_bet += 1
            bet_made = True
            bet_direction = 'lose'
            bet_odds = odds_lose

            bet_result, profit_mult = calculate_handicap_outcome(
                goal_diff, info['handicap_line'], 'lose'
            )

            if profit_mult > 0:
                profit = profit_mult * odds_lose
                total_return += 1 + profit
            elif profit_mult == 0:
                profit = 0
                total_return += 1
            else:
                profit = profit_mult
                total_return += 1 + profit

        if bet_made:
            bet_records.append({
                **info,
                'model_prob': prob,
                'market_prob': market_prob_win if bet_direction == 'win' else market_prob_lose,
                'edge': edge,
                'bet_direction': bet_direction,
                'bet_odds': bet_odds,
                'bet_result': bet_result,
                'profit': profit,
            })

    roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
    return roi, total_bet, total_return, bet_records


def train_ensemble_model(X_train, y_train):
    """训练集成模型"""
    models = {}

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=30, max_depth=2, learning_rate=0.03,
        min_child_samples=80, reg_alpha=3.0, reg_lambda=3.0,
        subsample=0.5, colsample_bytree=0.5,
        random_state=42, verbose=-1, n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_cal = CalibratedClassifierCV(lgb_model, method='isotonic', cv=5)
    lgb_cal.fit(X_train, y_train)
    models['LightGBM'] = lgb_cal

    # XGBoost
    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=30, max_depth=2, learning_rate=0.03,
            min_child_weight=50, subsample=0.5, colsample_bytree=0.5,
            reg_alpha=2.0, reg_lambda=2.0,
            random_state=42, n_jobs=-1, verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_cal = CalibratedClassifierCV(xgb_model, method='isotonic', cv=5)
        xgb_cal.fit(X_train, y_train)
        models['XGBoost'] = xgb_cal

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=60, max_depth=3, min_samples_leaf=50,
        max_features=0.5, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_cal = CalibratedClassifierCV(rf_model, method='isotonic', cv=5)
    rf_cal.fit(X_train, y_train)
    models['RandomForest'] = rf_cal

    # Logistic Regression
    lr_model = LogisticRegression(C=0.05, max_iter=1000, random_state=42, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    lr_cal = CalibratedClassifierCV(lr_model, method='isotonic', cv=5)
    lr_cal.fit(X_train, y_train)
    models['LogisticRegression'] = lr_cal

    return models


def ensemble_predict(models, X):
    """集成预测"""
    all_probs = []
    for name, model in models.items():
        probs = model.predict_proba(X)[:, 1]
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)


# 按盘口范围分组数据
print("\n按盘口范围分组数据...")

range_data = {r: {'train': [], 'test': []} for r in HANDICAP_RANGES.keys()}

for i, info in enumerate(train_info):
    r = info['handicap_range']
    if r:
        range_data[r]['train'].append(i)

for i, info in enumerate(test_info):
    r = info['handicap_range']
    if r:
        range_data[r]['test'].append(i)

print("\n各盘口范围样本量:")
for r in HANDICAP_RANGES.keys():
    print(f"  {r:>5}: 训练 {len(range_data[r]['train']):>5}, 测试 {len(range_data[r]['test']):>4}")

# 训练各盘口专项模型
print("\n" + "-" * 50)
print("训练6个专项模型...")
print("-" * 50)

range_models = {}
all_train_records = []
all_test_records = []

for range_name in HANDICAP_RANGES.keys():
    train_idx = range_data[range_name]['train']
    test_idx = range_data[range_name]['test']

    if len(train_idx) < 50:
        print(f"\n[{range_name}] 样本不足，跳过")
        continue

    print(f"\n[{range_name}盘口] 训练样本: {len(train_idx)}, 测试样本: {len(test_idx)}")

    # 准备数据
    X_train_range = pd.DataFrame([train_features[i] for i in train_idx])
    y_train_range = pd.Series([train_labels[i] for i in train_idx])
    info_train_range = [train_info[i] for i in train_idx]

    X_test_range = pd.DataFrame([test_features[i] for i in test_idx]) if test_idx else pd.DataFrame()
    y_test_range = pd.Series([test_labels[i] for i in test_idx]) if test_idx else pd.Series()
    info_test_range = [test_info[i] for i in test_idx] if test_idx else []

    # 对齐特征
    all_cols = list(set(X_train_range.columns) | (set(X_test_range.columns) if len(X_test_range) > 0 else set()))
    for col in all_cols:
        if col not in X_train_range.columns:
            X_train_range[col] = 0
        if len(X_test_range) > 0 and col not in X_test_range.columns:
            X_test_range[col] = 0

    X_train_range = X_train_range[sorted(all_cols)].fillna(0)
    if len(X_test_range) > 0:
        X_test_range = X_test_range[sorted(all_cols)].fillna(0)

    y_train_binary = (y_train_range == 1).astype(int)

    # 训练模型
    models = train_ensemble_model(X_train_range, y_train_binary)
    range_models[range_name] = models

    # 获取预测概率
    train_probs = ensemble_predict(models, X_train_range)

    # 计算训练集ROI
    params = RANGE_PARAMS[range_name]
    train_roi, train_bets, _, train_records = calculate_value_betting_roi(
        train_probs, y_train_range.values, info_train_range, params
    )

    for r in train_records:
        r['dataset'] = 'train'
        r['model_range'] = range_name
    all_train_records.extend(train_records)

    print(f"  训练集: {train_bets} 注, ROI: {train_roi*100:+.2f}%")

    # 计算测试集ROI
    if len(X_test_range) > 0:
        test_probs = ensemble_predict(models, X_test_range)
        test_roi, test_bets, _, test_records = calculate_value_betting_roi(
            test_probs, y_test_range.values, info_test_range, params
        )

        for r in test_records:
            r['dataset'] = 'test'
            r['model_range'] = range_name
        all_test_records.extend(test_records)

        print(f"  测试集: {test_bets} 注, ROI: {test_roi*100:+.2f}%")
        print(f"  过拟合差距: {(train_roi - test_roi)*100:.2f}%")

# 4. 汇总结果
print("\n" + "=" * 70)
print("4. 汇总结果")
print("=" * 70)

all_records = all_train_records + all_test_records
pred_df = pd.DataFrame(all_records)
pred_df.to_csv('pred_record_multimodel.csv', index=False)
print(f"\n预测记录已保存到 pred_record_multimodel.csv ({len(pred_df)} 条)")

# 统计函数
def count_results(df):
    n_full_win = sum(df['bet_result'] == 'full_win')
    n_half_win = sum(df['bet_result'] == 'half_win')
    n_push = sum(df['bet_result'] == 'push')
    n_half_lose = sum(df['bet_result'] == 'half_lose')
    n_full_lose = sum(df['bet_result'] == 'full_lose')
    n_wins = n_full_win + n_half_win
    return n_full_win, n_half_win, n_push, n_half_lose, n_full_lose, n_wins


def calc_stats(df, dim_name, dim_value):
    n_bets = len(df)
    if n_bets == 0:
        return None
    n_full_win, n_half_win, n_push, n_half_lose, n_full_lose, n_wins = count_results(df)
    total_profit = df['profit'].sum()
    roi = total_profit / n_bets
    return {
        'dimension': dim_name,
        'value': dim_value,
        'n_bets': n_bets,
        'n_full_win': n_full_win,
        'n_half_win': n_half_win,
        'n_push': n_push,
        'n_half_lose': n_half_lose,
        'n_full_lose': n_full_lose,
        'win_rate': n_wins / n_bets,
        'avg_odds': df['bet_odds'].mean(),
        'avg_edge': df['edge'].mean(),
        'total_profit': total_profit,
        'roi': roi,
    }


stats_list = []

# 按数据集统计
for dataset in ['train', 'test']:
    subset = pred_df[pred_df['dataset'] == dataset]
    stat = calc_stats(subset, 'dataset', dataset)
    if stat:
        stats_list.append(stat)

# 按盘口范围统计
print("\n--- 按盘口范围统计 (测试集) ---")
print(f"{'盘口':<8} {'投注':>6} {'胜率':>8} {'ROI':>10}")
print("-" * 35)

test_df_records = pred_df[pred_df['dataset'] == 'test']
for range_name in ['0', '0.25', '0.5', '0.75', '1', '1.25']:
    subset = test_df_records[test_df_records['model_range'] == range_name]
    stat = calc_stats(subset, 'handicap_range', range_name)
    if stat:
        stats_list.append(stat)
        print(f"±{range_name:<7} {stat['n_bets']:>6} {stat['win_rate']:>7.1%} {stat['roi']*100:>+9.2f}%")

# 按联赛统计
print("\n--- 按联赛统计 (测试集) ---")
print(f"{'联赛':<25} {'投注':>6} {'胜率':>8} {'ROI':>10}")
print("-" * 55)

for comp in test_df_records['competition'].dropna().unique():
    subset = test_df_records[test_df_records['competition'] == comp]
    stat = calc_stats(subset, 'competition', comp)
    if stat and stat['n_bets'] >= 3:
        stats_list.append(stat)
        print(f"{comp:<25} {stat['n_bets']:>6} {stat['win_rate']:>7.1%} {stat['roi']*100:>+9.2f}%")

# 按月份统计
print("\n--- 按月份统计 (测试集) ---")
print(f"{'月份':<10} {'投注':>6} {'胜率':>8} {'ROI':>10}")
print("-" * 40)

test_df_records['month'] = pd.to_datetime(test_df_records['date']).dt.to_period('M').astype(str)
for month in sorted(test_df_records['month'].unique()):
    subset = test_df_records[test_df_records['month'] == month]
    stat = calc_stats(subset, 'month', month)
    if stat:
        stats_list.append(stat)
        print(f"{month:<10} {stat['n_bets']:>6} {stat['win_rate']:>7.1%} {stat['roi']*100:>+9.2f}%")

# 按投注方向统计
print("\n--- 按投注方向统计 (测试集) ---")
for direction in test_df_records['bet_direction'].unique():
    subset = test_df_records[test_df_records['bet_direction'] == direction]
    stat = calc_stats(subset, 'bet_direction', direction)
    if stat:
        stats_list.append(stat)
        print(f"  {direction}: {stat['n_bets']} 注, 胜率 {stat['win_rate']:.1%}, ROI {stat['roi']*100:+.2f}%")

# 按具体盘口值统计
print("\n--- 按具体盘口值统计 (测试集) ---")
print(f"{'盘口':>8} {'投注':>6} {'胜率':>8} {'ROI':>10}")
print("-" * 35)

for line in sorted(test_df_records['handicap_line'].unique()):
    subset = test_df_records[test_df_records['handicap_line'] == line]
    stat = calc_stats(subset, 'handicap_line', f"{line:+.2f}")
    if stat and stat['n_bets'] >= 3:
        stats_list.append(stat)
        print(f"{line:>+8.2f} {stat['n_bets']:>6} {stat['win_rate']:>7.1%} {stat['roi']*100:>+9.2f}%")

# 保存统计
stats_df = pd.DataFrame(stats_list)
stats_df.to_csv('pred_roi_stats_multimodel.csv', index=False)
print(f"\nROI统计已保存到 pred_roi_stats_multimodel.csv")

# 总体结果
print("\n" + "=" * 70)
print("5. 总体结果")
print("=" * 70)

train_total = pred_df[pred_df['dataset'] == 'train']
test_total = pred_df[pred_df['dataset'] == 'test']

train_roi = train_total['profit'].sum() / len(train_total) if len(train_total) > 0 else 0
test_roi = test_total['profit'].sum() / len(test_total) if len(test_total) > 0 else 0

print(f"\n训练集汇总:")
print(f"  总投注: {len(train_total)} 注")
print(f"  总ROI: {train_roi*100:+.2f}%")

print(f"\n测试集汇总:")
print(f"  总投注: {len(test_total)} 注")
print(f"  总ROI: {test_roi*100:+.2f}%")

print(f"\n过拟合差距: {(train_roi - test_roi)*100:.2f}%")

# 与通用模型对比
print("\n" + "=" * 70)
print("6. 与通用模型对比")
print("=" * 70)

try:
    general_df = pd.read_csv('pred_record.csv')
    general_test = general_df[general_df['dataset'] == 'test']

    # 只对比相同盘口范围
    general_filtered = general_test[general_test['handicap_line'].apply(
        lambda x: get_handicap_range(x) is not None
    )]

    if len(general_filtered) > 0:
        general_roi = general_filtered['profit'].sum() / len(general_filtered)
        print(f"\n通用模型 (相同盘口范围):")
        print(f"  投注: {len(general_filtered)} 注")
        print(f"  ROI: {general_roi*100:+.2f}%")

        print(f"\n专项模型:")
        print(f"  投注: {len(test_total)} 注")
        print(f"  ROI: {test_roi*100:+.2f}%")

        improvement = (test_roi - general_roi) * 100
        print(f"\nROI提升: {improvement:+.2f}%")
except:
    print("无法读取通用模型结果进行对比")

print("\n" + "=" * 70)
print("完成!")
print("=" * 70)
