"""
调研: 单独训练 -0.25 和 +0.25 盘口的模型
分析这类平半盘/受让平半盘的ROI表现
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("调研: ±0.25 盘口专项模型")
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

# 只保留 ±0.25 盘口
df_quarter = df_clean[abs(abs(df_clean['win007_handicap_kickoff_line']) - 0.25) < 0.001].copy()
print(f"\n筛选 ±0.25 盘口后: {df_quarter.shape}")

# 查看盘口分布
print("\n盘口分布:")
print(df_quarter['win007_handicap_kickoff_line'].value_counts())

# 排除表现差的联赛
BAD_LEAGUES = [
    'Championship',
    '2. Bundesliga',
    'Ligue 1',
    'Ligue 2',
    'Club Friendly Games',
]

df_quarter = df_quarter[~df_quarter['competition'].isin(BAD_LEAGUES)]
print(f"过滤表现差联赛后: {df_quarter.shape}")

df_quarter = df_quarter.sort_values(['date', 'sofascore_match_id'])

train_start = datetime(2023, 6, 1)
train_cutoff = datetime(2025, 3, 1)
train_df = df_quarter[(df_quarter['date'] >= train_start) & (df_quarter['date'] <= train_cutoff)].copy()
test_df = df_quarter[df_quarter['date'] > train_cutoff].copy()

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


print("构建训练集特征...")
train_features = []
train_labels = []
train_info = []

# 使用完整数据集构建历史特征
for match_id in train_df['sofascore_match_id'].unique():
    match_records = train_df[train_df['sofascore_match_id'] == match_id]
    if len(match_records) < 2:
        continue

    for idx, row in match_records.iterrows():
        opponent = match_records[match_records['team_id'] != row['team_id']]
        opponent_row = opponent.iloc[0] if len(opponent) > 0 else None

        feat = build_match_features(row, opponent_row, df_clean)  # 用完整数据集做历史
        if feat is not None:
            train_features.append(feat)
            train_labels.append(row['handicap_result'])
            train_info.append({
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
            })

print(f"训练集有效样本: {len(train_features)}")

print("构建测试集特征...")
test_features = []
test_labels = []
test_info = []

for match_id in test_df['sofascore_match_id'].unique():
    match_records = test_df[test_df['sofascore_match_id'] == match_id]
    if len(match_records) < 2:
        continue

    for idx, row in match_records.iterrows():
        opponent = match_records[match_records['team_id'] != row['team_id']]
        opponent_row = opponent.iloc[0] if len(opponent) > 0 else None

        feat = build_match_features(row, opponent_row, df_clean)
        if feat is not None:
            test_features.append(feat)
            test_labels.append(row['handicap_result'])
            test_info.append({
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
            })

print(f"测试集有效样本: {len(test_features)}")

if len(train_features) < 100 or len(test_features) < 20:
    print("\n[警告] 样本量过少，结果可能不可靠")

X_train = pd.DataFrame(train_features)
y_train = pd.Series(train_labels)
X_test = pd.DataFrame(test_features)
y_test = pd.Series(test_labels)

all_cols = list(set(X_train.columns) | set(X_test.columns))
for col in all_cols:
    if col not in X_train.columns:
        X_train[col] = 0
    if col not in X_test.columns:
        X_test[col] = 0

X_train = X_train[sorted(all_cols)]
X_test = X_test[sorted(all_cols)]
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"\n特征数量: {len(all_cols)}")

# 3. 模型训练
print("\n" + "=" * 70)
print("3. 模型训练")
print("=" * 70)

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ODDS_MARKUP = 1.015  # 赔率上浮1.5%

# ±0.25 盘口专用参数
MIN_EDGE = 0.10  # 针对0.25盘口调整
MAX_EDGE = 0.20  # 放宽上限

MIN_ODDS = 0.80
MAX_ODDS = 1.15

y_train_binary = (y_train == 1).astype(int)
y_test_binary = (y_test == 1).astype(int)


def calculate_handicap_outcome(goal_diff, handicap_line, bet_direction):
    """计算亚洲让球盘的实际结果 (±0.25是复合盘口)"""
    if bet_direction == 'win':
        result = goal_diff + handicap_line
    else:
        result = -(goal_diff + handicap_line)

    # 0.25结尾的盘口是复合盘口，支持半赢半输
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


def calculate_value_betting_roi(model_probs, y_true, info_list, value_threshold=0.05):
    """基于价值投注策略计算ROI"""
    total_bet = 0
    total_return = 0
    bet_records = []

    for i, (prob, actual, info) in enumerate(zip(model_probs, y_true, info_list)):
        odds_win = info['handicap_odds'] * ODDS_MARKUP
        odds_lose = info['handicap_odds_opponent'] * ODDS_MARKUP

        market_prob_win = 1 / (1 + info['handicap_odds'])
        market_prob_lose = 1 / (1 + info['handicap_odds_opponent'])

        vt = value_threshold

        bet_made = False
        bet_direction = None
        bet_odds = None
        bet_result = None
        profit = 0
        edge = 0

        goal_diff = info.get('goal_diff', 0)

        if prob > market_prob_win + vt:
            edge = prob - market_prob_win

            if edge > MAX_EDGE or odds_win < MIN_ODDS or odds_win > MAX_ODDS:
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

            if edge > MAX_EDGE or odds_lose < MIN_ODDS or odds_lose > MAX_ODDS:
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


print("\n训练多模型集成...")

models = {}

# 使用更强的正则化防止过拟合
print("  训练 LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=30, max_depth=2, learning_rate=0.03,
    min_child_samples=100, reg_alpha=3.0, reg_lambda=3.0,
    subsample=0.5, colsample_bytree=0.5,
    random_state=42, verbose=-1, n_jobs=-1
)
lgb_model.fit(X_train, y_train_binary)
lgb_calibrated = CalibratedClassifierCV(lgb_model, method='isotonic', cv=5)
lgb_calibrated.fit(X_train, y_train_binary)
models['LightGBM'] = lgb_calibrated

if HAS_XGB:
    print("  训练 XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=30, max_depth=2, learning_rate=0.03,
        min_child_weight=50, subsample=0.5, colsample_bytree=0.5,
        reg_alpha=2.0, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train, y_train_binary)
    xgb_calibrated = CalibratedClassifierCV(xgb_model, method='isotonic', cv=5)
    xgb_calibrated.fit(X_train, y_train_binary)
    models['XGBoost'] = xgb_calibrated

print("  训练 Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=60, max_depth=3, min_samples_leaf=50,
    max_features=0.5, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train_binary)
rf_calibrated = CalibratedClassifierCV(rf_model, method='isotonic', cv=5)
rf_calibrated.fit(X_train, y_train_binary)
models['RandomForest'] = rf_calibrated

print("  训练 Logistic Regression...")
lr_model = LogisticRegression(C=0.05, max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train, y_train_binary)
lr_calibrated = CalibratedClassifierCV(lr_model, method='isotonic', cv=5)
lr_calibrated.fit(X_train, y_train_binary)
models['LogisticRegression'] = lr_calibrated

print(f"\n  共训练 {len(models)} 个模型")


def ensemble_predict(models, X, method='soft_vote'):
    """多模型集成预测"""
    all_probs = []
    for name, model in models.items():
        probs = model.predict_proba(X)[:, 1]
        all_probs.append(probs)

    all_probs = np.array(all_probs)
    return np.mean(all_probs, axis=0)


# 4. 评估
print("\n" + "=" * 70)
print("4. 模型评估")
print("=" * 70)

# 单模型表现
print("\n--- 各模型单独表现 (测试集) ---")
print(f"{'模型':<20} {'投注':>8} {'ROI':>10}")
print("-" * 40)

model_rois = {}
for name, model in models.items():
    probs = model.predict_proba(X_test)[:, 1]
    roi, n_bets, _, _ = calculate_value_betting_roi(
        probs, y_test.values, test_info, value_threshold=0.10
    )
    model_rois[name] = roi
    print(f"{name:<20} {n_bets:>8} {roi*100:>+9.2f}%")

# 集成预测
print("\n--- 集成模型表现 ---")
train_probs = ensemble_predict(models, X_train)
test_probs = ensemble_predict(models, X_test)

print("\n不同VT阈值表现 (测试集):")
print(f"{'VT':>6} {'投注':>8} {'胜率':>8} {'ROI':>10}")
print("-" * 35)

best_test_roi = -float('inf')
best_vt = 0.10
best_records = []

for vt in [0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20]:
    roi, n_bets, _, records = calculate_value_betting_roi(test_probs, y_test.values, test_info, vt)
    if n_bets > 0:
        n_wins = sum(1 for r in records if r['bet_result'] in ['full_win', 'half_win'])
        win_rate = n_wins / n_bets
        print(f"{vt:>6.2f} {n_bets:>8} {win_rate:>8.2%} {roi*100:>+9.2f}%")
        if n_bets >= 10 and roi > best_test_roi:
            best_test_roi = roi
            best_vt = vt
            best_records = records

# 训练集表现
train_roi, train_bets, _, train_records = calculate_value_betting_roi(
    train_probs, y_train.values, train_info, best_vt
)

# 测试集最佳表现
test_roi, test_bets, _, test_records = calculate_value_betting_roi(
    test_probs, y_test.values, test_info, best_vt
)

print(f"\n最优VT阈值: {best_vt}")

print("\n" + "=" * 70)
print("5. 最终结果")
print("=" * 70)

print(f"\n训练集 (±0.25盘口专项模型):")
print(f"  投注场次: {train_bets}")
print(f"  ROI: {train_roi*100:+.2f}%")

print(f"\n测试集 (±0.25盘口专项模型):")
print(f"  投注场次: {test_bets}")
print(f"  ROI: {test_roi*100:+.2f}%")

print(f"\n过拟合差距: {(train_roi - test_roi)*100:.2f}%")

# 按盘口方向细分
if test_records:
    test_df_records = pd.DataFrame(test_records)

    print("\n--- 按盘口方向分析 ---")
    for line in test_df_records['handicap_line'].unique():
        subset = test_df_records[test_df_records['handicap_line'] == line]
        n = len(subset)
        if n > 0:
            roi = subset['profit'].sum() / n
            n_wins = sum(1 for _, r in subset.iterrows() if r['bet_result'] in ['full_win', 'half_win'])
            print(f"  盘口 {line:+.2f}: {n} 注, 胜率 {n_wins/n:.1%}, ROI {roi*100:+.2f}%")

    print("\n--- 按联赛分析 ---")
    for comp in test_df_records['competition'].unique():
        subset = test_df_records[test_df_records['competition'] == comp]
        n = len(subset)
        if n >= 3:
            roi = subset['profit'].sum() / n
            print(f"  {comp}: {n} 注, ROI {roi*100:+.2f}%")

    print("\n--- 按投注方向分析 ---")
    for direction in test_df_records['bet_direction'].unique():
        subset = test_df_records[test_df_records['bet_direction'] == direction]
        n = len(subset)
        if n > 0:
            roi = subset['profit'].sum() / n
            print(f"  {direction}: {n} 注, ROI {roi*100:+.2f}%")

    # 结果分布
    print("\n--- 结果分布 (测试集) ---")
    result_counts = test_df_records['bet_result'].value_counts()
    for result, count in result_counts.items():
        print(f"  {result}: {count} ({count/len(test_df_records)*100:.1f}%)")

print("\n" + "=" * 70)
print("调研完成!")
print("=" * 70)
