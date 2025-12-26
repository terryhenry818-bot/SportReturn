"""
对比不同训练集时间范围对模型效果的影响
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ============ 数据加载 ============
print("加载数据...")
df = pd.read_csv('wide_table.csv')
df['date'] = pd.to_datetime(df['date'])

win007_required_cols = [
    'win007_handicap_kickoff_line',
    'win007_handicap_kickoff_odds',
    'win007_handicap_kickoff_odds_opponent',
]

df_clean = df.dropna(subset=win007_required_cols)
df_clean = df_clean.dropna(subset=['handicap_result'])

# 定义6类盘口范围
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

BAD_LEAGUES = ['Championship', '2. Bundesliga', 'Ligue 1', 'Ligue 2', 'Club Friendly Games']
df_filtered = df_filtered[~df_filtered['competition'].isin(BAD_LEAGUES)]
df_filtered = df_filtered.sort_values(['date', 'sofascore_match_id'])

print(f"有效数据: {len(df_filtered)} 条")

# ============ 特征工程函数 ============
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

# ============ 模型和投注函数 ============
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
    bet_records = []

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

        bet_made = False
        bet_direction = None
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

            bet_result, profit_mult = calculate_handicap_outcome(goal_diff, info['handicap_line'], 'win')
            if profit_mult > 0:
                profit = profit_mult * odds_win
                total_return += 1 + profit
            elif profit_mult == 0:
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

            bet_result, profit_mult = calculate_handicap_outcome(goal_diff, info['handicap_line'], 'lose')
            if profit_mult > 0:
                profit = profit_mult * odds_lose
                total_return += 1 + profit
            elif profit_mult == 0:
                total_return += 1
            else:
                profit = profit_mult
                total_return += 1 + profit

        if bet_made:
            bet_records.append({'profit': profit})

    roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
    return roi, total_bet, bet_records

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

    lr_model = LogisticRegression(C=0.05, max_iter=1000, random_state=42, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    lr_cal = CalibratedClassifierCV(lr_model, method='isotonic', cv=5)
    lr_cal.fit(X_train, y_train)
    models['LogisticRegression'] = lr_cal

    return models

def ensemble_predict(models, X):
    all_probs = []
    for name, model in models.items():
        probs = model.predict_proba(X)[:, 1]
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)

# ============ 主函数：对比不同训练集范围 ============
def run_experiment(train_start, train_end, test_start, experiment_name):
    """运行单次实验"""
    train_df = df_filtered[(df_filtered['date'] >= train_start) & (df_filtered['date'] <= train_end)].copy()
    test_df = df_filtered[df_filtered['date'] > test_start].copy()

    # 构建数据集
    def build_dataset(df_source):
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

                feat = build_match_features(row, opponent_row, df_clean)
                if feat is not None:
                    features_list.append(feat)
                    labels_list.append(row['handicap_result'])
                    info_list.append({
                        'handicap_line': row['win007_handicap_kickoff_line'],
                        'handicap_odds': row['win007_handicap_kickoff_odds'],
                        'handicap_odds_opponent': row['win007_handicap_kickoff_odds_opponent'],
                        'goal_diff': row['goal_diff'],
                        'handicap_range': get_handicap_range(row['win007_handicap_kickoff_line']),
                    })

        return features_list, labels_list, info_list

    train_features, train_labels, train_info = build_dataset(train_df)
    test_features, test_labels, test_info = build_dataset(test_df)

    # 按盘口分组
    range_data = {r: {'train': [], 'test': []} for r in HANDICAP_RANGES.keys()}

    for i, info in enumerate(train_info):
        r = info['handicap_range']
        if r:
            range_data[r]['train'].append(i)

    for i, info in enumerate(test_info):
        r = info['handicap_range']
        if r:
            range_data[r]['test'].append(i)

    # 训练各盘口模型并计算ROI
    total_train_bets = 0
    total_train_profit = 0
    total_test_bets = 0
    total_test_profit = 0

    for range_name in HANDICAP_RANGES.keys():
        train_idx = range_data[range_name]['train']
        test_idx = range_data[range_name]['test']

        if len(train_idx) < 50:
            continue

        X_train_range = pd.DataFrame([train_features[i] for i in train_idx])
        y_train_range = pd.Series([train_labels[i] for i in train_idx])
        info_train_range = [train_info[i] for i in train_idx]

        X_test_range = pd.DataFrame([test_features[i] for i in test_idx]) if test_idx else pd.DataFrame()
        y_test_range = pd.Series([test_labels[i] for i in test_idx]) if test_idx else pd.Series()
        info_test_range = [test_info[i] for i in test_idx] if test_idx else []

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

        models = train_ensemble_model(X_train_range, y_train_binary)
        train_probs = ensemble_predict(models, X_train_range)

        params = RANGE_PARAMS[range_name]
        train_roi, train_bets, train_records = calculate_value_betting_roi(
            train_probs, y_train_range.values, info_train_range, params
        )

        total_train_bets += train_bets
        total_train_profit += sum(r['profit'] for r in train_records)

        if len(X_test_range) > 0:
            test_probs = ensemble_predict(models, X_test_range)
            test_roi, test_bets, test_records = calculate_value_betting_roi(
                test_probs, y_test_range.values, info_test_range, params
            )
            total_test_bets += test_bets
            total_test_profit += sum(r['profit'] for r in test_records)

    train_roi = total_train_profit / total_train_bets if total_train_bets > 0 else 0
    test_roi = total_test_profit / total_test_bets if total_test_bets > 0 else 0
    overfit = train_roi - test_roi

    return {
        'experiment': experiment_name,
        'train_start': train_start.strftime('%Y-%m-%d'),
        'train_samples': len(train_features),
        'train_bets': total_train_bets,
        'train_roi': train_roi,
        'test_bets': total_test_bets,
        'test_roi': test_roi,
        'overfit': overfit,
    }


# ============ 运行5组实验 ============
print("\n" + "=" * 70)
print("对比5种不同训练集时间范围")
print("=" * 70)

test_cutoff = datetime(2025, 3, 1)

experiments = [
    (datetime(2022, 6, 1), "2022.6.1 - 2025.3.1 (33个月)"),
    (datetime(2022, 9, 1), "2022.9.1 - 2025.3.1 (30个月)"),
    (datetime(2022, 12, 1), "2022.12.1 - 2025.3.1 (27个月)"),
    (datetime(2023, 3, 1), "2023.3.1 - 2025.3.1 (24个月)"),
    (datetime(2023, 6, 1), "2023.6.1 - 2025.3.1 (21个月)"),
]

results = []
for train_start, name in experiments:
    print(f"\n正在训练: {name}...")
    result = run_experiment(train_start, test_cutoff, test_cutoff, name)
    results.append(result)
    print(f"  训练集: {result['train_samples']} 样本, {result['train_bets']} 注, ROI: {result['train_roi']*100:+.2f}%")
    print(f"  测试集: {result['test_bets']} 注, ROI: {result['test_roi']*100:+.2f}%")
    print(f"  过拟合: {result['overfit']*100:+.2f}%")

# ============ 汇总对比 ============
print("\n" + "=" * 70)
print("结果对比汇总")
print("=" * 70)

print(f"\n{'训练集范围':<30} {'训练样本':>8} {'训练ROI':>10} {'测试投注':>8} {'测试ROI':>10} {'过拟合':>10}")
print("-" * 80)

for r in results:
    print(f"{r['experiment']:<30} {r['train_samples']:>8} {r['train_roi']*100:>+9.2f}% {r['test_bets']:>8} {r['test_roi']*100:>+9.2f}% {r['overfit']*100:>+9.2f}%")

# 找出最佳
best_roi = max(results, key=lambda x: x['test_roi'])
least_overfit = min(results, key=lambda x: abs(x['overfit']))

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
print(f"\n最高测试ROI: {best_roi['experiment']}")
print(f"  测试ROI: {best_roi['test_roi']*100:+.2f}%, 过拟合: {best_roi['overfit']*100:+.2f}%")

print(f"\n最小过拟合: {least_overfit['experiment']}")
print(f"  测试ROI: {least_overfit['test_roi']*100:+.2f}%, 过拟合: {least_overfit['overfit']*100:+.2f}%")
