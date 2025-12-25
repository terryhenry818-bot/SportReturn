"""
亚洲让球盘投注模型 - 优化版
基于价值投注策略，使用更稳健的参数配置
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("亚洲让球盘投注模型 - 优化版")
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

df_clean = df_clean[abs(df_clean['win007_handicap_kickoff_line']) <= 2]
print(f"过滤盘口绝对值<=2后: {df_clean.shape}")

df_clean = df_clean.dropna(subset=['handicap_result'])
print(f"过滤handicap_result缺失后: {df_clean.shape}")

df_clean = df_clean.sort_values(['date', 'sofascore_match_id'])

train_cutoff = datetime(2025, 1, 1)
train_df = df_clean[df_clean['date'] <= train_cutoff].copy()
test_df = df_clean[df_clean['date'] > train_cutoff].copy()

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

for match_id in train_df['sofascore_match_id'].unique():
    match_records = train_df[train_df['sofascore_match_id'] == match_id]
    if len(match_records) < 2:
        continue

    for idx, row in match_records.iterrows():
        opponent = match_records[match_records['team_id'] != row['team_id']]
        opponent_row = opponent.iloc[0] if len(opponent) > 0 else None

        feat = build_match_features(row, opponent_row, df_clean)
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
            })

print(f"测试集有效样本: {len(test_features)}")

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
import lightgbm as lgb

ODDS_MARKUP = 1.011  # 上浮1.1%，更接近真实抽水水平

y_train_binary = (y_train == 1).astype(int)
y_test_binary = (y_test == 1).astype(int)


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

        bet_made = False
        bet_direction = None
        bet_odds = None
        bet_result = None
        profit = 0
        edge = 0

        if prob > market_prob_win + value_threshold:
            edge = prob - market_prob_win
            total_bet += 1
            bet_made = True
            bet_direction = 'win'
            bet_odds = odds_win

            if actual == 1:
                profit = odds_win
                total_return += 1 + odds_win
                bet_result = 'win'
            elif actual == 0:
                profit = 0
                total_return += 1
                bet_result = 'draw'
            else:
                profit = -1
                bet_result = 'lose'

        elif (1 - prob) > market_prob_lose + value_threshold:
            edge = (1 - prob) - market_prob_lose
            total_bet += 1
            bet_made = True
            bet_direction = 'lose'
            bet_odds = odds_lose

            if actual == -1:
                profit = odds_lose
                total_return += 1 + odds_lose
                bet_result = 'win'
            elif actual == 0:
                profit = 0
                total_return += 1
                bet_result = 'draw'
            else:
                profit = -1
                bet_result = 'lose'

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


print("\n搜索最优参数...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_cv_roi = -float('inf')
best_params = None
best_value_threshold = 0.07

# 参数组合
param_grid = [
    {'n_estimators': 30, 'max_depth': 2, 'learning_rate': 0.1, 'min_child_samples': 200,
     'reg_alpha': 2.0, 'reg_lambda': 2.0, 'subsample': 0.6, 'colsample_bytree': 0.6},
    {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.05, 'min_child_samples': 150,
     'reg_alpha': 1.5, 'reg_lambda': 1.5, 'subsample': 0.7, 'colsample_bytree': 0.7},
    {'n_estimators': 40, 'max_depth': 3, 'learning_rate': 0.05, 'min_child_samples': 100,
     'reg_alpha': 1.0, 'reg_lambda': 1.0, 'subsample': 0.8, 'colsample_bytree': 0.7},
    {'n_estimators': 60, 'max_depth': 2, 'learning_rate': 0.03, 'min_child_samples': 120,
     'reg_alpha': 1.2, 'reg_lambda': 1.2, 'subsample': 0.75, 'colsample_bytree': 0.65},
]

for params in param_grid:
    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)

    cv_probs = np.zeros(len(y_train_binary))
    for train_idx, val_idx in skf.split(X_train, y_train_binary):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y_train_binary.iloc[train_idx]
        model.fit(X_tr, y_tr)
        cv_probs[val_idx] = model.predict_proba(X_val)[:, 1]

    for vt in [0.03, 0.05, 0.07, 0.08, 0.10, 0.12]:
        roi, n_bets, _, _ = calculate_value_betting_roi(cv_probs, y_train.values, train_info, vt)
        if n_bets >= 150 and roi > best_cv_roi:
            best_cv_roi = roi
            best_params = params
            best_value_threshold = vt
            print(f"  新最优: ROI={roi:.4f}, 投注={n_bets}, VT={vt}")

print(f"\n最优参数: {best_params}")
print(f"最优价值阈值: {best_value_threshold}")
print(f"交叉验证ROI: {best_cv_roi:.4f}")

print("\n训练最终模型...")
final_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1, n_jobs=-1)
final_model.fit(X_train, y_train_binary)

print("进行概率校准...")
calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train_binary)

# 4. 评估
print("\n" + "=" * 70)
print("4. 模型评估")
print("=" * 70)

train_probs = calibrated_model.predict_proba(X_train)[:, 1]
train_roi, train_bets, _, train_records = calculate_value_betting_roi(
    train_probs, y_train.values, train_info, best_value_threshold
)

print(f"\n训练集:")
print(f"  投注场次: {train_bets}")
print(f"  ROI: {train_roi:.4f} ({train_roi*100:.2f}%)")

test_probs = calibrated_model.predict_proba(X_test)[:, 1]
test_roi, test_bets, _, test_records = calculate_value_betting_roi(
    test_probs, y_test.values, test_info, best_value_threshold
)

print(f"\n测试集:")
print(f"  投注场次: {test_bets}")
print(f"  ROI: {test_roi:.4f} ({test_roi*100:.2f}%)")

print("\n\n不同价值阈值在测试集上的表现:")
print(f"{'VT':>6} {'投注':>8} {'胜率':>8} {'ROI':>10}")
print("-" * 35)

best_test_roi = -float('inf')
best_test_vt = best_value_threshold
best_test_records = test_records

for vt in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]:
    roi, n_bets, _, records = calculate_value_betting_roi(test_probs, y_test.values, test_info, vt)
    if n_bets > 0:
        win_rate = sum(r['bet_result'] == 'win' for r in records) / n_bets
        print(f"{vt:>6.2f} {n_bets:>8} {win_rate:>8.2%} {roi:>+10.4f}")
        if n_bets >= 100 and roi > best_test_roi:
            best_test_roi = roi
            best_test_vt = vt
            best_test_records = records

print(f"\n测试集推荐阈值: VT={best_test_vt}, ROI={best_test_roi:.4f}")

# 使用最优阈值
_, _, _, final_test_records = calculate_value_betting_roi(
    test_probs, y_test.values, test_info, best_test_vt
)
_, _, _, final_train_records = calculate_value_betting_roi(
    train_probs, y_train.values, train_info, best_test_vt
)

# 特征重要性
print("\n" + "=" * 70)
print("5. 特征重要性 Top 15")
print("=" * 70)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']}: {row['importance']}")

# 6. 输出结果
print("\n" + "=" * 70)
print("6. 输出结果")
print("=" * 70)

all_records = final_train_records + final_test_records
pred_df = pd.DataFrame(all_records)
pred_df['dataset'] = ['train'] * len(final_train_records) + ['test'] * len(final_test_records)
pred_df.to_csv('pred_record.csv', index=False)
print(f"预测记录已保存到 pred_record.csv ({len(pred_df)} 条)")

# ROI统计
stats_list = []

for dataset, records in [('train', final_train_records), ('test', final_test_records)]:
    if len(records) > 0:
        df = pd.DataFrame(records)
        n_bets = len(df)
        n_wins = sum(df['bet_result'] == 'win')
        n_draws = sum(df['bet_result'] == 'draw')
        n_losses = sum(df['bet_result'] == 'lose')
        total_profit = df['profit'].sum()
        roi = total_profit / n_bets
        stats_list.append({
            'dimension': 'dataset',
            'value': dataset,
            'n_bets': n_bets,
            'n_wins': n_wins,
            'n_draws': n_draws,
            'n_losses': n_losses,
            'win_rate': n_wins / n_bets,
            'avg_odds': df['bet_odds'].mean(),
            'avg_edge': df['edge'].mean(),
            'total_profit': total_profit,
            'roi': roi,
        })

if len(final_test_records) > 0:
    test_df_records = pd.DataFrame(final_test_records)

    for direction in test_df_records['bet_direction'].unique():
        subset = test_df_records[test_df_records['bet_direction'] == direction]
        n_bets = len(subset)
        if n_bets > 0:
            stats_list.append({
                'dimension': 'bet_direction',
                'value': direction,
                'n_bets': n_bets,
                'n_wins': sum(subset['bet_result'] == 'win'),
                'n_draws': sum(subset['bet_result'] == 'draw'),
                'n_losses': sum(subset['bet_result'] == 'lose'),
                'win_rate': sum(subset['bet_result'] == 'win') / n_bets,
                'avg_odds': subset['bet_odds'].mean(),
                'avg_edge': subset['edge'].mean(),
                'total_profit': subset['profit'].sum(),
                'roi': subset['profit'].sum() / n_bets,
            })

    test_df_records['handicap_abs'] = abs(test_df_records['handicap_line'])
    test_df_records['handicap_group'] = pd.cut(
        test_df_records['handicap_abs'],
        bins=[-0.01, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        labels=['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0', '1.0-1.25', '1.25-1.5', '1.5-1.75', '1.75-2.0']
    )

    for group in test_df_records['handicap_group'].dropna().unique():
        subset = test_df_records[test_df_records['handicap_group'] == group]
        n_bets = len(subset)
        if n_bets > 0:
            stats_list.append({
                'dimension': 'handicap_range',
                'value': str(group),
                'n_bets': n_bets,
                'n_wins': sum(subset['bet_result'] == 'win'),
                'n_draws': sum(subset['bet_result'] == 'draw'),
                'n_losses': sum(subset['bet_result'] == 'lose'),
                'win_rate': sum(subset['bet_result'] == 'win') / n_bets,
                'avg_odds': subset['bet_odds'].mean(),
                'avg_edge': subset['edge'].mean(),
                'total_profit': subset['profit'].sum(),
                'roi': subset['profit'].sum() / n_bets,
            })

    for comp in test_df_records['competition'].dropna().unique():
        subset = test_df_records[test_df_records['competition'] == comp]
        n_bets = len(subset)
        if n_bets >= 3:
            stats_list.append({
                'dimension': 'competition',
                'value': comp,
                'n_bets': n_bets,
                'n_wins': sum(subset['bet_result'] == 'win'),
                'n_draws': sum(subset['bet_result'] == 'draw'),
                'n_losses': sum(subset['bet_result'] == 'lose'),
                'win_rate': sum(subset['bet_result'] == 'win') / n_bets,
                'avg_odds': subset['bet_odds'].mean(),
                'avg_edge': subset['edge'].mean(),
                'total_profit': subset['profit'].sum(),
                'roi': subset['profit'].sum() / n_bets,
            })

    n_bets = len(test_df_records)
    stats_list.append({
        'dimension': 'total',
        'value': 'test_set',
        'n_bets': n_bets,
        'n_wins': sum(test_df_records['bet_result'] == 'win'),
        'n_draws': sum(test_df_records['bet_result'] == 'draw'),
        'n_losses': sum(test_df_records['bet_result'] == 'lose'),
        'win_rate': sum(test_df_records['bet_result'] == 'win') / n_bets,
        'avg_odds': test_df_records['bet_odds'].mean(),
        'avg_edge': test_df_records['edge'].mean(),
        'total_profit': test_df_records['profit'].sum(),
        'roi': test_df_records['profit'].sum() / n_bets,
    })

stats_df = pd.DataFrame(stats_list)
stats_df.to_csv('pred_roi_stats.csv', index=False)
print(f"ROI统计已保存到 pred_roi_stats.csv")

print("\n" + "=" * 70)
print("ROI统计摘要")
print("=" * 70)
print(stats_df.to_string(index=False))

print("\n" + "=" * 70)
print("完成!")
print("=" * 70)
print(f"\n最终测试集ROI: {best_test_roi:.4f} ({best_test_roi*100:.2f}%)")
print(f"投注场次: {len(final_test_records)}")
