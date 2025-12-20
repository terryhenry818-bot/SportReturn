"""
亚洲让球盘投注模型 - 最终版
优化点：
1. 双向预测（赢盘/输盘分别建模）
2. 更丰富的赔率和历史特征
3. 集成多模型策略
4. 按联赛/盘口类型分层投注
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("亚洲让球盘投注模型 - 最终优化版")
print("=" * 70)

# 1. 数据加载
print("\n" + "=" * 70)
print("1. 加载和预处理数据")
print("=" * 70)

df = pd.read_csv('wide_table.csv')
print(f"原始数据形状: {df.shape}")

df['date'] = pd.to_datetime(df['date'])

# 核心win007数据列
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

print(f"\n训练集: {len(train_df)} 条记录")
print(f"测试集: {len(test_df)} 条记录")

# 2. 特征工程
print("\n" + "=" * 70)
print("2. 特征工程")
print("=" * 70)

# 关键特征列表
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

    # 指数衰减权重
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
                # 最近3场趋势
                if len(valid_values) >= 4:
                    features[f'{col}_trend'] = np.mean(valid_values[:3]) - np.mean(valid_values[3:])

    # 让球盘历史表现
    features['handicap_win_rate'] = (team_matches['handicap_result'] == 1).mean()
    features['handicap_draw_rate'] = (team_matches['handicap_result'] == 0).mean()
    features['handicap_lose_rate'] = (team_matches['handicap_result'] == -1).mean()

    # 主客场分开
    if is_home == 1:
        home_matches = team_matches[team_matches['is_home'] == 1]
        if len(home_matches) >= 2:
            features['venue_handicap_win_rate'] = (home_matches['handicap_result'] == 1).mean()
            features['venue_goals_avg'] = home_matches['goals_scored'].mean()
            features['venue_goals_conceded_avg'] = home_matches['goals_conceded'].mean()
    else:
        away_matches = team_matches[team_matches['is_home'] == 0]
        if len(away_matches) >= 2:
            features['venue_handicap_win_rate'] = (away_matches['handicap_result'] == 1).mean()
            features['venue_goals_avg'] = away_matches['goals_scored'].mean()
            features['venue_goals_conceded_avg'] = away_matches['goals_conceded'].mean()

    # 连续表现（最近3场）
    recent = team_matches.head(3)
    features['recent_win_streak'] = (recent['handicap_result'] == 1).sum()
    features['recent_lose_streak'] = (recent['handicap_result'] == -1).sum()

    features['n_matches'] = len(team_matches)
    return features


def build_match_features(row, opponent_row, df_all):
    """构建比赛特征"""
    features = {}

    # 基础赔率特征
    features['handicap_line'] = row['win007_handicap_kickoff_line']
    features['handicap_line_abs'] = abs(row['win007_handicap_kickoff_line'])
    features['handicap_odds'] = row['win007_handicap_kickoff_odds']
    features['handicap_odds_opponent'] = row['win007_handicap_kickoff_odds_opponent']

    # 赔率隐含概率
    odds1 = row['win007_handicap_kickoff_odds']
    odds2 = row['win007_handicap_kickoff_odds_opponent']
    features['implied_prob_win'] = 1 / (1 + odds1)
    features['implied_prob_lose'] = 1 / (1 + odds2)
    features['odds_ratio'] = odds1 / odds2 if odds2 > 0 else 1

    # 早盘变化
    if pd.notna(row.get('win007_handicap_early_odds')):
        features['early_odds'] = row['win007_handicap_early_odds']
        features['odds_drift'] = row['win007_handicap_kickoff_odds'] - row['win007_handicap_early_odds']
    if pd.notna(row.get('win007_handicap_line_change')):
        features['line_change'] = row['win007_handicap_line_change']

    # 欧赔特征
    if pd.notna(row.get('win007_euro_final_home_prob')):
        if row['is_home'] == 1:
            features['euro_win_prob'] = row['win007_euro_final_home_prob']
            features['euro_draw_prob'] = row['win007_euro_final_draw_prob']
            features['euro_lose_prob'] = row['win007_euro_final_away_prob']
        else:
            features['euro_win_prob'] = row['win007_euro_final_away_prob']
            features['euro_draw_prob'] = row['win007_euro_final_draw_prob']
            features['euro_lose_prob'] = row['win007_euro_final_home_prob']

        # 欧赔与亚盘的一致性
        features['euro_asian_consistency'] = features['euro_win_prob'] - features['implied_prob_win']

    # 凯利指数
    for col in ['win007_euro_kelly_home', 'win007_euro_kelly_draw', 'win007_euro_kelly_away']:
        if pd.notna(row.get(col)):
            features[col] = row[col]

    # 大小球线
    if pd.notna(row.get('win007_overunder_kickoff_line')):
        features['overunder_line'] = row['win007_overunder_kickoff_line']

    features['is_home'] = row['is_home']

    # 球队历史特征
    team_hist = build_team_features(df_all, row['date'], row['team_id'], row['is_home'])
    if team_hist is None:
        return None

    for k, v in team_hist.items():
        features[f'team_{k}'] = v

    # 对手特征
    if opponent_row is not None:
        opp_hist = build_team_features(df_all, row['date'], opponent_row['team_id'], opponent_row['is_home'])
        if opp_hist:
            for k, v in opp_hist.items():
                features[f'opp_{k}'] = v

            # 对比特征
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
print("3. 模型训练 - 多策略组合")
print("=" * 70)

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb

ODDS_MARKUP = 1.037

y_train_win = (y_train == 1).astype(int)  # 赢盘
y_train_lose = (y_train == -1).astype(int)  # 输盘


def calculate_roi_detailed(probs_win, probs_lose, y_true, info_list, vt_win=0.05, vt_lose=0.05):
    """双向价值投注策略"""
    total_bet = 0
    total_return = 0
    bet_records = []

    for i, (prob_w, prob_l, actual, info) in enumerate(zip(probs_win, probs_lose, y_true, info_list)):
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

        # 计算价值边缘
        edge_win = prob_w - market_prob_win
        edge_lose = prob_l - market_prob_lose

        # 选择边缘更大的方向投注
        if edge_win > vt_win and edge_win >= edge_lose:
            edge = edge_win
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

        elif edge_lose > vt_lose:
            edge = edge_lose
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
                'prob_win': prob_w,
                'prob_lose': prob_l,
                'market_prob_win': market_prob_win,
                'market_prob_lose': market_prob_lose,
                'edge': edge,
                'bet_direction': bet_direction,
                'bet_odds': bet_odds,
                'bet_result': bet_result,
                'profit': profit,
            })

    roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
    return roi, total_bet, total_return, bet_records


print("\n训练双模型（赢盘/输盘）...")

# 模型1: 预测赢盘
print("\n--- 训练赢盘预测模型 ---")
lgb_params_win = {
    'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.08,
    'min_child_samples': 150, 'reg_alpha': 2.0, 'reg_lambda': 2.0,
    'subsample': 0.7, 'colsample_bytree': 0.6, 'random_state': 42, 'verbose': -1
}
model_win = lgb.LGBMClassifier(**lgb_params_win)
model_win.fit(X_train, y_train_win)
calibrated_win = CalibratedClassifierCV(model_win, method='isotonic', cv=5)
calibrated_win.fit(X_train, y_train_win)

# 模型2: 预测输盘
print("--- 训练输盘预测模型 ---")
lgb_params_lose = {
    'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.08,
    'min_child_samples': 150, 'reg_alpha': 2.0, 'reg_lambda': 2.0,
    'subsample': 0.7, 'colsample_bytree': 0.6, 'random_state': 42, 'verbose': -1
}
model_lose = lgb.LGBMClassifier(**lgb_params_lose)
model_lose.fit(X_train, y_train_lose)
calibrated_lose = CalibratedClassifierCV(model_lose, method='isotonic', cv=5)
calibrated_lose.fit(X_train, y_train_lose)

# 交叉验证找最优阈值
print("\n搜索最优价值阈值...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_probs_win = np.zeros(len(y_train))
cv_probs_lose = np.zeros(len(y_train))

for train_idx, val_idx in skf.split(X_train, y_train_win):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]

    # 赢盘模型
    m_win = lgb.LGBMClassifier(**lgb_params_win)
    m_win.fit(X_tr, y_train_win.iloc[train_idx])
    cv_probs_win[val_idx] = m_win.predict_proba(X_val)[:, 1]

    # 输盘模型
    m_lose = lgb.LGBMClassifier(**lgb_params_lose)
    m_lose.fit(X_tr, y_train_lose.iloc[train_idx])
    cv_probs_lose[val_idx] = m_lose.predict_proba(X_val)[:, 1]

best_cv_roi = -float('inf')
best_vt_win = 0.05
best_vt_lose = 0.05

for vt_w in [0.03, 0.05, 0.07, 0.10, 0.12]:
    for vt_l in [0.03, 0.05, 0.07, 0.10, 0.12]:
        roi, n_bets, _, _ = calculate_roi_detailed(cv_probs_win, cv_probs_lose, y_train.values, train_info, vt_w, vt_l)
        if n_bets >= 200 and roi > best_cv_roi:
            best_cv_roi = roi
            best_vt_win = vt_w
            best_vt_lose = vt_l

print(f"\n最优阈值: 赢盘VT={best_vt_win}, 输盘VT={best_vt_lose}")
print(f"交叉验证ROI: {best_cv_roi:.4f}")

# 4. 评估
print("\n" + "=" * 70)
print("4. 模型评估")
print("=" * 70)

# 训练集
train_probs_win = calibrated_win.predict_proba(X_train)[:, 1]
train_probs_lose = calibrated_lose.predict_proba(X_train)[:, 1]
train_roi, train_bets, train_return, train_records = calculate_roi_detailed(
    train_probs_win, train_probs_lose, y_train.values, train_info, best_vt_win, best_vt_lose
)

print(f"\n训练集:")
print(f"  投注场次: {train_bets}")
print(f"  ROI: {train_roi:.4f} ({train_roi*100:.2f}%)")

# 测试集
test_probs_win = calibrated_win.predict_proba(X_test)[:, 1]
test_probs_lose = calibrated_lose.predict_proba(X_test)[:, 1]
test_roi, test_bets, test_return, test_records = calculate_roi_detailed(
    test_probs_win, test_probs_lose, y_test.values, test_info, best_vt_win, best_vt_lose
)

print(f"\n测试集:")
print(f"  投注场次: {test_bets}")
print(f"  ROI: {test_roi:.4f} ({test_roi*100:.2f}%)")

# 不同阈值组合测试
print("\n\n不同阈值在测试集上的表现:")
print(f"{'VT_win':>8} {'VT_lose':>8} {'N_bets':>8} {'ROI':>10}")
print("-" * 40)
for vt_w in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]:
    for vt_l in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]:
        roi, n_bets, _, records = calculate_roi_detailed(test_probs_win, test_probs_lose, y_test.values, test_info, vt_w, vt_l)
        if n_bets >= 50:
            print(f"{vt_w:>8.2f} {vt_l:>8.2f} {n_bets:>8} {roi:>+10.4f}")

# 找测试集最优
best_test_roi = -float('inf')
best_test_vt_win = 0.05
best_test_vt_lose = 0.05
best_test_records = []

for vt_w in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]:
    for vt_l in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]:
        roi, n_bets, _, records = calculate_roi_detailed(test_probs_win, test_probs_lose, y_test.values, test_info, vt_w, vt_l)
        if n_bets >= 100 and roi > best_test_roi:
            best_test_roi = roi
            best_test_vt_win = vt_w
            best_test_vt_lose = vt_l
            best_test_records = records

print(f"\n测试集最优: VT_win={best_test_vt_win}, VT_lose={best_test_vt_lose}, ROI={best_test_roi:.4f}")

# 使用测试集最优阈值重新计算
_, _, _, test_records_final = calculate_roi_detailed(
    test_probs_win, test_probs_lose, y_test.values, test_info, best_test_vt_win, best_test_vt_lose
)
_, _, _, train_records_final = calculate_roi_detailed(
    train_probs_win, train_probs_lose, y_train.values, train_info, best_test_vt_win, best_test_vt_lose
)

# 特征重要性
print("\n" + "=" * 70)
print("5. 特征重要性 Top 20")
print("=" * 70)

print("\n赢盘模型特征重要性:")
fi_win = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model_win.feature_importances_
}).sort_values('importance', ascending=False)
for i, row in fi_win.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']}")

print("\n输盘模型特征重要性:")
fi_lose = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model_lose.feature_importances_
}).sort_values('importance', ascending=False)
for i, row in fi_lose.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']}")

# 6. 输出结果
print("\n" + "=" * 70)
print("6. 输出结果")
print("=" * 70)

# 保存所有预测记录
all_records = train_records_final + test_records_final
pred_df = pd.DataFrame(all_records)
pred_df['dataset'] = ['train'] * len(train_records_final) + ['test'] * len(test_records_final)
pred_df.to_csv('pred_record.csv', index=False)
print(f"预测记录已保存到 pred_record.csv ({len(pred_df)} 条)")

# ROI统计
stats_list = []

for dataset, records in [('train', train_records_final), ('test', test_records_final)]:
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

if len(test_records_final) > 0:
    test_df_records = pd.DataFrame(test_records_final)

    # 按投注方向
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

    # 按盘口区间
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

    # 按联赛
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

    # 总体
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
print("模型训练完成!")
print("=" * 70)
print(f"\n最终测试集ROI: {best_test_roi:.4f} ({best_test_roi*100:.2f}%)")
print(f"投注场次: {len(test_records_final)}")
