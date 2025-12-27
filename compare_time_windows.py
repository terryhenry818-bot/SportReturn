"""
时间窗口对比测试
比较4个不同训练集/测试集时间窗口的ROI表现
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

print("=" * 80)
print("时间窗口对比测试 - 4个不同训练集/测试集组合")
print("=" * 80)

# 定义4个时间窗口
TIME_WINDOWS = [
    {
        'name': '窗口1',
        'train_start': datetime(2023, 6, 1),
        'train_end': datetime(2025, 3, 1),
        'test_start': datetime(2025, 3, 2),
        'test_end': datetime(2025, 12, 27),
    },
    {
        'name': '窗口2',
        'train_start': datetime(2023, 4, 1),
        'train_end': datetime(2025, 1, 1),
        'test_start': datetime(2025, 1, 2),
        'test_end': datetime(2025, 10, 27),
    },
    {
        'name': '窗口3',
        'train_start': datetime(2023, 2, 1),
        'train_end': datetime(2024, 11, 1),
        'test_start': datetime(2024, 11, 2),
        'test_end': datetime(2025, 8, 27),
    },
    {
        'name': '窗口4',
        'train_start': datetime(2023, 12, 1),
        'train_end': datetime(2024, 9, 1),
        'test_start': datetime(2024, 9, 2),
        'test_end': datetime(2025, 6, 27),
    },
]

# 加载数据
print("\n加载数据...")
df = pd.read_csv('wide_table.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"总数据量: {len(df)} 条")

# 数据清洗
win007_required_cols = [
    'win007_handicap_kickoff_line',
    'win007_handicap_kickoff_odds',
    'win007_handicap_kickoff_odds_opponent',
]
df_clean = df.dropna(subset=win007_required_cols)
df_clean = df_clean.dropna(subset=['handicap_result'])

# 盘口过滤
HANDICAP_RANGES = {
    '0': [0.0],
    '0.25': [0.25, -0.25],
    '0.5': [0.5, -0.5],
    '0.75': [0.75, -0.75],
    '1': [1.0, -1.0],
    '1.25': [1.25, -1.25],
}

valid_lines = []
for lines in HANDICAP_RANGES.values():
    valid_lines.extend(lines)

def is_valid_line(line):
    for valid in valid_lines:
        if abs(line - valid) < 0.001:
            return True
    return False

df_filtered = df_clean[df_clean['win007_handicap_kickoff_line'].apply(is_valid_line)].copy()

# 排除表现差的联赛
BAD_LEAGUES = ['Championship', '2. Bundesliga', 'Ligue 1', 'Ligue 2', 'Club Friendly Games']
df_filtered = df_filtered[~df_filtered['competition'].isin(BAD_LEAGUES)]
df_filtered = df_filtered.sort_values(['date', 'sofascore_match_id'])

print(f"过滤后数据量: {len(df_filtered)} 条")
print(f"数据日期范围: {df_filtered['date'].min().date()} ~ {df_filtered['date'].max().date()}")

# 特征工程函数
key_stats = [
    'sofascore_xG', 'sofascore_total_shots', 'sofascore_shots_on_target',
    'sofascore_big_chances', 'sofascore_ball_possession',
    'sofascore_pass_accuracy', 'sofascore_corner_kicks',
    'sofascore_goalkeeper_saves', 'sofascore_tackles',
    'sofascore_duels_won_pct', 'sofascore_team_avg_rating',
    'goals_scored', 'goals_conceded', 'goal_diff',
]

def build_team_features(df_all, target_date, team_id, is_home, n_matches=7):
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


def build_dataset(df_source, df_all_for_history):
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

    return features_list, labels_list, info_list


ODDS_MARKUP = 1.015

RANGE_PARAMS = {
    '0': {'min_edge': 0.10, 'max_edge': 0.18, 'min_odds': 0.85, 'max_odds': 1.12, 'vt': 0.10, 'only_positive': False},
    '0.25': {'min_edge': 0.12, 'max_edge': 0.20, 'min_odds': 0.82, 'max_odds': 1.15, 'vt': 0.12, 'only_positive': True},
    '0.5': {'min_edge': 0.08, 'max_edge': 0.22, 'min_odds': 0.80, 'max_odds': 1.18, 'vt': 0.08, 'only_positive': True},
    '0.75': {'min_edge': 0.10, 'max_edge': 0.18, 'min_odds': 0.85, 'max_odds': 1.12, 'vt': 0.10, 'only_positive': True},
    '1': {'min_edge': 0.10, 'max_edge': 0.18, 'min_odds': 0.85, 'max_odds': 1.12, 'vt': 0.10, 'only_positive': True},
    '1.25': {'min_edge': 0.10, 'max_edge': 0.18, 'min_odds': 0.85, 'max_odds': 1.12, 'vt': 0.10, 'only_positive': True},
}


def calculate_handicap_outcome(goal_diff, handicap_line, bet_direction):
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
    total_bet = 0
    total_return = 0

    min_edge = params['min_edge']
    max_edge = params['max_edge']
    min_odds = params['min_odds']
    max_odds = params['max_odds']
    vt = params['vt']
    only_positive = params.get('only_positive', False)

    for i, (prob, actual, info) in enumerate(zip(model_probs, y_true, info_list)):
        if only_positive and info['handicap_line'] < -0.001:
            continue

        odds_win = info['handicap_odds'] * ODDS_MARKUP
        odds_lose = info['handicap_odds_opponent'] * ODDS_MARKUP
        market_prob_win = 1 / (1 + info['handicap_odds'])
        market_prob_lose = 1 / (1 + info['handicap_odds_opponent'])
        goal_diff = info.get('goal_diff', 0)

        if prob > market_prob_win + vt:
            edge = prob - market_prob_win
            if edge < min_edge or edge > max_edge:
                continue
            if odds_win < min_odds or odds_win > max_odds:
                continue

            total_bet += 1
            bet_result, profit_mult = calculate_handicap_outcome(goal_diff, info['handicap_line'], 'win')

            if profit_mult > 0:
                total_return += 1 + profit_mult * odds_win
            elif profit_mult == 0:
                total_return += 1
            else:
                total_return += 1 + profit_mult

        elif (1 - prob) > market_prob_lose + vt:
            edge = (1 - prob) - market_prob_lose
            if edge < min_edge or edge > max_edge:
                continue
            if odds_lose < min_odds or odds_lose > max_odds:
                continue

            total_bet += 1
            bet_result, profit_mult = calculate_handicap_outcome(goal_diff, info['handicap_line'], 'lose')

            if profit_mult > 0:
                total_return += 1 + profit_mult * odds_lose
            elif profit_mult == 0:
                total_return += 1
            else:
                total_return += 1 + profit_mult

    roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
    return roi, total_bet, total_return


def train_ensemble_model(X_train, y_train):
    models = {}

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

    rf_model = RandomForestClassifier(
        n_estimators=60, max_depth=3, min_samples_leaf=50,
        max_features=0.5, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_cal = CalibratedClassifierCV(rf_model, method='isotonic', cv=5)
    rf_cal.fit(X_train, y_train)
    models['RandomForest'] = rf_cal

    return models


def ensemble_predict(models, X):
    all_probs = []
    for name, model in models.items():
        probs = model.predict_proba(X)[:, 1]
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)


# 测试每个时间窗口
results = []

for window in TIME_WINDOWS:
    print(f"\n{'='*80}")
    print(f"测试 {window['name']}")
    print(f"训练集: {window['train_start'].date()} ~ {window['train_end'].date()}")
    print(f"测试集: {window['test_start'].date()} ~ {window['test_end'].date()}")
    print("=" * 80)

    # 分割数据
    train_df = df_filtered[
        (df_filtered['date'] >= window['train_start']) &
        (df_filtered['date'] <= window['train_end'])
    ].copy()

    test_df = df_filtered[
        (df_filtered['date'] >= window['test_start']) &
        (df_filtered['date'] <= window['test_end'])
    ].copy()

    print(f"训练集: {len(train_df)} 条")
    print(f"测试集: {len(test_df)} 条")

    if len(train_df) < 100 or len(test_df) < 50:
        print("数据量不足，跳过")
        results.append({
            'name': window['name'],
            'train_range': f"{window['train_start'].date()} ~ {window['train_end'].date()}",
            'test_range': f"{window['test_start'].date()} ~ {window['test_end'].date()}",
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'total_bets': 0,
            'roi': None,
        })
        continue

    # 构建特征
    print("构建特征...")
    train_features, train_labels, train_info = build_dataset(train_df, df_clean)
    test_features, test_labels, test_info = build_dataset(test_df, df_clean)

    print(f"训练特征: {len(train_features)} 样本")
    print(f"测试特征: {len(test_features)} 样本")

    if len(train_features) < 50 or len(test_features) < 20:
        print("特征构建后数据量不足，跳过")
        results.append({
            'name': window['name'],
            'train_range': f"{window['train_start'].date()} ~ {window['train_end'].date()}",
            'test_range': f"{window['test_start'].date()} ~ {window['test_end'].date()}",
            'train_samples': len(train_features),
            'test_samples': len(test_features),
            'total_bets': 0,
            'roi': None,
        })
        continue

    # 按盘口范围分组
    range_data = {r: {'train': [], 'test': []} for r in HANDICAP_RANGES.keys()}

    for i, info in enumerate(train_info):
        r = info['handicap_range']
        if r:
            range_data[r]['train'].append(i)

    for i, info in enumerate(test_info):
        r = info['handicap_range']
        if r:
            range_data[r]['test'].append(i)

    # 训练和测试
    total_bets = 0
    total_return = 0
    range_results = []

    for range_name, params in RANGE_PARAMS.items():
        train_indices = range_data[range_name]['train']
        test_indices = range_data[range_name]['test']

        if len(train_indices) < 30 or len(test_indices) < 10:
            continue

        X_train = pd.DataFrame([train_features[i] for i in train_indices]).fillna(0)
        y_train = np.array([1 if train_labels[i] == 1 else 0 for i in train_indices])
        train_info_subset = [train_info[i] for i in train_indices]

        X_test = pd.DataFrame([test_features[i] for i in test_indices]).fillna(0)
        y_test = np.array([test_labels[i] for i in test_indices])
        test_info_subset = [test_info[i] for i in test_indices]

        # 对齐特征
        common_cols = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        # 训练模型
        models = train_ensemble_model(X_train, y_train)

        # 预测
        test_probs = ensemble_predict(models, X_test)

        # 计算ROI
        roi, bets, returns = calculate_value_betting_roi(test_probs, y_test, test_info_subset, params)

        total_bets += bets
        total_return += returns

        if bets > 0:
            range_results.append({
                'range': range_name,
                'bets': bets,
                'roi': roi,
            })

    # 汇总结果
    overall_roi = (total_return - total_bets) / total_bets if total_bets > 0 else 0

    print(f"\n各盘口结果:")
    for rr in range_results:
        print(f"  盘口 {rr['range']}: {rr['bets']} 注, ROI {rr['roi']*100:+.2f}%")

    print(f"\n总计: {total_bets} 注, ROI {overall_roi*100:+.2f}%")

    results.append({
        'name': window['name'],
        'train_range': f"{window['train_start'].date()} ~ {window['train_end'].date()}",
        'test_range': f"{window['test_start'].date()} ~ {window['test_end'].date()}",
        'train_samples': len(train_features),
        'test_samples': len(test_features),
        'total_bets': total_bets,
        'roi': overall_roi,
    })

# 汇总对比
print("\n" + "=" * 80)
print("时间窗口对比汇总")
print("=" * 80)

print(f"\n{'窗口':<8} {'训练集':<28} {'测试集':<28} {'投注数':>8} {'ROI':>10}")
print("-" * 90)

valid_rois = []
for r in results:
    roi_str = f"{r['roi']*100:+.2f}%" if r['roi'] is not None else "N/A"
    print(f"{r['name']:<8} {r['train_range']:<28} {r['test_range']:<28} {r['total_bets']:>8} {roi_str:>10}")
    if r['roi'] is not None:
        valid_rois.append(r['roi'])

if valid_rois:
    avg_roi = np.mean(valid_rois)
    std_roi = np.std(valid_rois)
    min_roi = min(valid_rois)
    max_roi = max(valid_rois)

    print("\n" + "-" * 90)
    print(f"\n统计分析:")
    print(f"  平均ROI: {avg_roi*100:+.2f}%")
    print(f"  标准差:  {std_roi*100:.2f}%")
    print(f"  最小ROI: {min_roi*100:+.2f}%")
    print(f"  最大ROI: {max_roi*100:+.2f}%")
    print(f"  波动范围: {(max_roi-min_roi)*100:.2f}%")

    if std_roi < 0.03:
        print(f"\n  结论: ROI波动较小 (标准差<3%), 模型表现稳定")
    elif std_roi < 0.05:
        print(f"\n  结论: ROI波动适中 (标准差3-5%), 模型表现较稳定")
    else:
        print(f"\n  结论: ROI波动较大 (标准差>5%), 模型表现不够稳定")

print("\n完成!")
