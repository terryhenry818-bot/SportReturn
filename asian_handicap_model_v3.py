"""
亚洲让球盘投注模型 V3
优化点：
1. 添加对手球队历史特征
2. 基于价值投注(Value Betting)策略
3. 更强的正则化防止过拟合
4. 修复交叉验证问题
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 加载数据
print("=" * 60)
print("1. 加载和预处理数据")
print("=" * 60)

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

df_clean = df_clean[abs(df_clean['win007_handicap_kickoff_line']) <= 1.25]
print(f"过滤盘口绝对值<=1.25后: {df_clean.shape}")

df_clean = df_clean.dropna(subset=['handicap_result'])
print(f"过滤handicap_result缺失后: {df_clean.shape}")

df_clean = df_clean.sort_values(['date', 'sofascore_match_id'])

train_cutoff = datetime(2025, 7, 31)
train_df = df_clean[df_clean['date'] <= train_cutoff].copy()
test_df = df_clean[df_clean['date'] > train_cutoff].copy()

print(f"\n训练集: {len(train_df)} 条记录 (截至2025-07-31)")
print(f"测试集: {len(test_df)} 条记录 (2025-08-01之后)")

# 2. 特征工程
print("\n" + "=" * 60)
print("2. 特征工程 - 构建双方球队历史特征")
print("=" * 60)

sofascore_features = [
    'sofascore_xG', 'sofascore_total_shots', 'sofascore_shots_on_target',
    'sofascore_big_chances', 'sofascore_ball_possession',
    'sofascore_pass_accuracy', 'sofascore_corner_kicks',
    'sofascore_goalkeeper_saves', 'sofascore_tackles',
    'sofascore_duels_won_pct', 'sofascore_team_avg_rating',
]

result_features = ['goals_scored', 'goals_conceded', 'goal_diff']


def build_team_history_features(df_all, target_date, team_id, is_home, n_matches=7):
    """构建球队最近n场比赛的历史特征"""
    team_matches = df_all[
        (df_all['team_id'] == team_id) &
        (df_all['date'] < target_date)
    ].sort_values('date', ascending=False).head(n_matches)

    if len(team_matches) < 3:
        return None

    features = {}
    weights = np.exp(-np.arange(len(team_matches)) * 0.2)
    weights = weights / weights.sum()

    for col in sofascore_features + result_features:
        if col in team_matches.columns:
            values = team_matches[col].values
            valid_mask = ~pd.isna(values)
            if valid_mask.sum() >= 2:
                valid_values = values[valid_mask].astype(float)
                valid_weights = weights[valid_mask]
                valid_weights = valid_weights / valid_weights.sum()
                features[f'{col}_wavg'] = np.average(valid_values, weights=valid_weights)
                features[f'{col}_std'] = np.std(valid_values)

    features['handicap_win_rate'] = (team_matches['handicap_result'] == 1).mean()
    features['handicap_draw_rate'] = (team_matches['handicap_result'] == 0).mean()

    if is_home == 1:
        home_matches = team_matches[team_matches['is_home'] == 1]
        if len(home_matches) >= 2:
            features['home_handicap_win_rate'] = (home_matches['handicap_result'] == 1).mean()
            features['home_goals_avg'] = home_matches['goals_scored'].mean()
    else:
        away_matches = team_matches[team_matches['is_home'] == 0]
        if len(away_matches) >= 2:
            features['away_handicap_win_rate'] = (away_matches['handicap_result'] == 1).mean()
            features['away_goals_avg'] = away_matches['goals_scored'].mean()

    features['n_history_matches'] = len(team_matches)
    return features


def build_match_features(row, opponent_row, df_all):
    """为单场比赛构建所有特征"""
    features = {}

    features['handicap_line'] = row['win007_handicap_kickoff_line']
    features['handicap_odds'] = row['win007_handicap_kickoff_odds']
    features['handicap_odds_opponent'] = row['win007_handicap_kickoff_odds_opponent']

    odds1 = row['win007_handicap_kickoff_odds']
    odds2 = row['win007_handicap_kickoff_odds_opponent']
    prob1 = 1 / (1 + odds1)
    prob2 = 1 / (1 + odds2)
    features['implied_prob'] = prob1 / (prob1 + prob2)

    if pd.notna(row.get('win007_handicap_early_odds')):
        features['early_odds'] = row['win007_handicap_early_odds']
        features['odds_change'] = row['win007_handicap_kickoff_odds'] - row['win007_handicap_early_odds']

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

    team_hist = build_team_history_features(df_all, row['date'], row['team_id'], row['is_home'])
    if team_hist is None:
        return None

    for k, v in team_hist.items():
        features[f'team_{k}'] = v

    if opponent_row is not None:
        opponent_hist = build_team_history_features(
            df_all, row['date'], opponent_row['team_id'], opponent_row['is_home']
        )
        if opponent_hist:
            for k, v in opponent_hist.items():
                features[f'opp_{k}'] = v

            for feat in sofascore_features + result_features:
                team_feat = f'team_{feat}_wavg'
                opp_feat = f'opp_{feat}_wavg'
                if team_feat in features and opp_feat in features:
                    features[f'diff_{feat}'] = features[team_feat] - features[opp_feat]

    return features


print("构建训练集特征（包含对手特征）...")
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
print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")

# 3. 模型训练
print("\n" + "=" * 60)
print("3. 模型训练 - 基于价值投注策略")
print("=" * 60)

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

ODDS_MARKUP = 1.037


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


y_train_binary = (y_train == 1).astype(int)
y_test_binary = (y_test == 1).astype(int)

print("标签分布:")
print(f"训练集: 赢盘={sum(y_train==1)}, 走水={sum(y_train==0)}, 输盘={sum(y_train==-1)}")
print(f"测试集: 赢盘={sum(y_test==1)}, 走水={sum(y_test==0)}, 输盘={sum(y_test==-1)}")

print("\n搜索最优参数...")

best_cv_roi = -float('inf')
best_params = None
best_value_threshold = 0.05
best_model_type = None

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# LightGBM参数
lgb_params_list = [
    {'n_estimators': 30, 'max_depth': 2, 'learning_rate': 0.1, 'min_child_samples': 200,
     'reg_alpha': 2.0, 'reg_lambda': 2.0, 'subsample': 0.6, 'colsample_bytree': 0.6},
    {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.05, 'min_child_samples': 150,
     'reg_alpha': 1.5, 'reg_lambda': 1.5, 'subsample': 0.7, 'colsample_bytree': 0.7},
    {'n_estimators': 20, 'max_depth': 2, 'learning_rate': 0.15, 'min_child_samples': 300,
     'reg_alpha': 3.0, 'reg_lambda': 3.0, 'subsample': 0.5, 'colsample_bytree': 0.5},
]

print("\n测试LightGBM模型...")
for params in lgb_params_list:
    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)

    # 手动实现交叉验证
    cv_probs = np.zeros(len(y_train_binary))
    for train_idx, val_idx in skf.split(X_train, y_train_binary):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train_binary.iloc[train_idx], y_train_binary.iloc[val_idx]
        model.fit(X_tr, y_tr)
        cv_probs[val_idx] = model.predict_proba(X_val)[:, 1]

    for vt in [0.02, 0.03, 0.05, 0.07, 0.10, 0.12]:
        roi, n_bets, _, _ = calculate_value_betting_roi(cv_probs, y_train.values, train_info, vt)
        if n_bets >= 100 and roi > best_cv_roi:
            best_cv_roi = roi
            best_params = params
            best_value_threshold = vt
            best_model_type = 'lgb'
            print(f"  新最优: ROI={roi:.4f}, 投注={n_bets}, VT={vt}")

print("\n测试Logistic Regression模型...")
for C in [0.001, 0.01, 0.1, 1.0]:
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)

    cv_probs = np.zeros(len(y_train_binary))
    for train_idx, val_idx in skf.split(X_train, y_train_binary):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train_binary.iloc[train_idx], y_train_binary.iloc[val_idx]
        model.fit(X_tr, y_tr)
        cv_probs[val_idx] = model.predict_proba(X_val)[:, 1]

    for vt in [0.02, 0.03, 0.05, 0.07, 0.10, 0.12]:
        roi, n_bets, _, _ = calculate_value_betting_roi(cv_probs, y_train.values, train_info, vt)
        if n_bets >= 100 and roi > best_cv_roi:
            best_cv_roi = roi
            best_params = {'C': C}
            best_value_threshold = vt
            best_model_type = 'lr'
            print(f"  新最优: ROI={roi:.4f}, 投注={n_bets}, VT={vt}, C={C}")

print(f"\n最优模型类型: {best_model_type}")
print(f"最优参数: {best_params}")
print(f"最优价值阈值: {best_value_threshold}")
print(f"交叉验证ROI: {best_cv_roi:.4f}")

print("\n训练最终模型...")
if best_model_type == 'lgb':
    final_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1, n_jobs=-1)
else:
    final_model = LogisticRegression(**best_params, max_iter=1000, random_state=42)

final_model.fit(X_train, y_train_binary)

# 概率校准
print("进行概率校准...")
calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train_binary)

# 4. 模型评估
print("\n" + "=" * 60)
print("4. 模型评估")
print("=" * 60)

train_probs = calibrated_model.predict_proba(X_train)[:, 1]
train_roi, train_bets, train_return, train_records = calculate_value_betting_roi(
    train_probs, y_train.values, train_info, best_value_threshold
)

print(f"\n训练集:")
print(f"  投注场次: {train_bets}")
print(f"  总回报: {train_return:.2f}")
print(f"  ROI: {train_roi:.4f} ({train_roi*100:.2f}%)")

test_probs = calibrated_model.predict_proba(X_test)[:, 1]
test_roi, test_bets, test_return, test_records = calculate_value_betting_roi(
    test_probs, y_test.values, test_info, best_value_threshold
)

print(f"\n测试集:")
print(f"  投注场次: {test_bets}")
print(f"  总回报: {test_return:.2f}")
print(f"  ROI: {test_roi:.4f} ({test_roi*100:.2f}%)")

print("\n\n不同价值阈值在测试集上的表现:")
for vt in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15]:
    roi, n_bets, _, _ = calculate_value_betting_roi(test_probs, y_test.values, test_info, vt)
    print(f"  VT={vt:.2f}: 投注={n_bets:4d}, ROI={roi:+.4f} ({roi*100:+.2f}%)")

if best_model_type == 'lgb':
    print("\n" + "=" * 60)
    print("5. 特征重要性 Top 20")
    print("=" * 60)

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']}")

# 6. 输出结果
print("\n" + "=" * 60)
print("6. 输出预测结果")
print("=" * 60)

all_records = train_records + test_records
pred_df = pd.DataFrame(all_records)
pred_df['dataset'] = ['train'] * len(train_records) + ['test'] * len(test_records)
pred_df.to_csv('pred_record.csv', index=False)
print(f"预测记录已保存到 pred_record.csv ({len(pred_df)} 条)")

stats_list = []

for dataset, records in [('train', train_records), ('test', test_records)]:
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

if len(test_records) > 0:
    test_df_records = pd.DataFrame(test_records)

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
        bins=[-0.01, 0.25, 0.5, 0.75, 1.0, 1.25],
        labels=['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0', '1.0-1.25']
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

print("\n" + "=" * 60)
print("ROI统计摘要")
print("=" * 60)
print(stats_df.to_string(index=False))

print("\n" + "=" * 60)
print("模型训练完成!")
print("=" * 60)
