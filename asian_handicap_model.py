"""
亚洲让球盘投注模型
目标：最大化ROI
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

# 转换日期
df['date'] = pd.to_datetime(df['date'])

# 定义win007让球盘相关的必需列
win007_handicap_cols = [
    'win007_handicap_kickoff_line',
    'win007_handicap_kickoff_odds',
    'win007_handicap_kickoff_odds_opponent',
    'win007_handicap_early_line',
    'win007_handicap_early_odds',
    'win007_handicap_early_odds_opponent',
]

# 过滤：win007让球盘数据完整
df_clean = df.dropna(subset=win007_handicap_cols)
print(f"过滤win007让球盘缺失后: {df_clean.shape}")

# 过滤：盘口绝对值 <= 1.25
df_clean = df_clean[abs(df_clean['win007_handicap_kickoff_line']) <= 1.25]
print(f"过滤盘口绝对值<=1.25后: {df_clean.shape}")

# 过滤：handicap_result不为空
df_clean = df_clean.dropna(subset=['handicap_result'])
print(f"过滤handicap_result缺失后: {df_clean.shape}")

# 划分训练集和测试集
train_cutoff = datetime(2025, 7, 31)
train_df = df_clean[df_clean['date'] <= train_cutoff].copy()
test_df = df_clean[df_clean['date'] > train_cutoff].copy()

print(f"\n训练集: {len(train_df)} 条记录 (截至2025-07-31)")
print(f"测试集: {len(test_df)} 条记录 (2025-08-01之后)")

print("\n训练集日期范围:", train_df['date'].min(), "至", train_df['date'].max())
print("测试集日期范围:", test_df['date'].min(), "至", test_df['date'].max())

# 2. 特征工程
print("\n" + "=" * 60)
print("2. 特征工程 - 构建球队历史7场比赛特征")
print("=" * 60)

# 定义sofascore统计特征（比赛后才有的数据）
sofascore_match_features = [
    'sofascore_xG', 'sofascore_total_shots', 'sofascore_shots_on_target',
    'sofascore_shots_inside_box', 'sofascore_shots_outside_box', 'sofascore_blocked_shots',
    'sofascore_big_chances', 'sofascore_big_chances_scored', 'sofascore_big_chances_missed',
    'sofascore_ball_possession', 'sofascore_total_passes', 'sofascore_accurate_passes',
    'sofascore_pass_accuracy', 'sofascore_touches_in_box', 'sofascore_corner_kicks',
    'sofascore_goalkeeper_saves', 'sofascore_tackles', 'sofascore_interceptions',
    'sofascore_clearances', 'sofascore_recoveries', 'sofascore_duels_won_pct',
    'sofascore_dribbles_successful', 'sofascore_fouls', 'sofascore_team_avg_rating',
]

# 用于历史特征的列
historical_stats = sofascore_match_features + [
    'goals_scored', 'goals_conceded', 'goal_diff',
]

# 可以使用的赛前赔率特征
odds_features = [
    # 让球盘赔率
    'win007_handicap_kickoff_line',
    'win007_handicap_kickoff_odds',
    'win007_handicap_kickoff_odds_opponent',
    'win007_handicap_early_line',
    'win007_handicap_early_odds',
    'win007_handicap_early_odds_opponent',
    # 大小球赔率
    'win007_overunder_kickoff_line',
    'win007_overunder_kickoff_over_odds',
    'win007_overunder_kickoff_under_odds',
    # 欧赔
    'win007_euro_final_home_odds',
    'win007_euro_final_draw_odds',
    'win007_euro_final_away_odds',
    'win007_euro_final_return_rate',
    'win007_euro_final_home_prob',
    'win007_euro_final_draw_prob',
    'win007_euro_final_away_prob',
    # 凯利指数
    'win007_euro_kelly_home',
    'win007_euro_kelly_draw',
    'win007_euro_kelly_away',
    # 赔率变化
    'win007_handicap_line_change',
    'win007_euro_home_odds_change',
    'win007_euro_draw_odds_change',
    'win007_euro_away_odds_change',
]


def build_team_history_features(df_all, target_date, team_id, is_home, n_matches=7):
    """
    构建球队最近n场比赛的历史特征
    只使用target_date之前的比赛数据
    """
    # 获取该球队在目标日期之前的所有比赛
    team_matches = df_all[
        (df_all['team_id'] == team_id) &
        (df_all['date'] < target_date)
    ].sort_values('date', ascending=False).head(n_matches)

    if len(team_matches) < 3:  # 至少需要3场历史比赛
        return None

    features = {}

    # 按权重计算历史特征（越近的比赛权重越高）
    weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4][:len(team_matches)])
    weights = weights / weights.sum()

    for col in historical_stats:
        if col in team_matches.columns:
            values = team_matches[col].values
            valid_mask = ~pd.isna(values)
            if valid_mask.sum() > 0:
                valid_values = values[valid_mask].astype(float)
                valid_weights = weights[valid_mask]
                valid_weights = valid_weights / valid_weights.sum()

                # 加权平均
                features[f'hist_{col}_wavg'] = np.average(valid_values, weights=valid_weights)
                # 标准差
                features[f'hist_{col}_std'] = np.std(valid_values) if len(valid_values) > 1 else 0
                # 趋势（最近3场vs之前）
                if len(valid_values) >= 4:
                    recent = np.mean(valid_values[:3])
                    older = np.mean(valid_values[3:])
                    features[f'hist_{col}_trend'] = recent - older

    # 主客场胜率
    home_matches = team_matches[team_matches['is_home'] == 1]
    away_matches = team_matches[team_matches['is_home'] == 0]

    # 让球盘胜率
    features['hist_handicap_win_rate'] = (team_matches['handicap_result'] == 1).mean()
    if len(home_matches) > 0:
        features['hist_home_handicap_win_rate'] = (home_matches['handicap_result'] == 1).mean()
    else:
        features['hist_home_handicap_win_rate'] = 0.5
    if len(away_matches) > 0:
        features['hist_away_handicap_win_rate'] = (away_matches['handicap_result'] == 1).mean()
    else:
        features['hist_away_handicap_win_rate'] = 0.5

    # 平均进球和失球
    features['hist_avg_goals_scored'] = team_matches['goals_scored'].mean()
    features['hist_avg_goals_conceded'] = team_matches['goals_conceded'].mean()

    # 比赛场次
    features['hist_n_matches'] = len(team_matches)

    return features


def build_match_features(row, df_all):
    """
    为单场比赛构建所有特征
    """
    features = {}

    # 1. 赔率特征（赛前可用）
    for col in odds_features:
        if col in row.index and pd.notna(row[col]):
            features[col] = row[col]

    # 2. 构建球队历史特征
    team_hist = build_team_history_features(
        df_all, row['date'], row['team_id'], row['is_home']
    )

    if team_hist is None:
        return None

    for k, v in team_hist.items():
        features[f'team_{k}'] = v

    # 3. 基于is_home的标记
    features['is_home'] = row['is_home']

    # 4. 赔率派生特征
    if 'win007_handicap_kickoff_odds' in features and 'win007_handicap_kickoff_odds_opponent' in features:
        # 赔率隐含概率（让球盘赔率不含本金）
        odds1 = features['win007_handicap_kickoff_odds']
        odds2 = features['win007_handicap_kickoff_odds_opponent']
        prob1 = 1 / (1 + odds1)  # 不含本金的赔率转概率
        prob2 = 1 / (1 + odds2)
        features['implied_prob'] = prob1 / (prob1 + prob2)
        features['odds_ratio'] = odds1 / odds2 if odds2 > 0 else 1

    # 5. 欧赔派生特征
    if 'win007_euro_final_home_prob' in features:
        home_prob = features.get('win007_euro_final_home_prob', 0.33)
        draw_prob = features.get('win007_euro_final_draw_prob', 0.33)
        away_prob = features.get('win007_euro_final_away_prob', 0.33)

        if row['is_home'] == 1:
            features['euro_win_prob'] = home_prob
            features['euro_not_lose_prob'] = home_prob + draw_prob
        else:
            features['euro_win_prob'] = away_prob
            features['euro_not_lose_prob'] = away_prob + draw_prob

    return features


print("构建训练集特征...")
train_features = []
train_labels = []
train_info = []

for idx, row in train_df.iterrows():
    feat = build_match_features(row, df_clean)
    if feat is not None:
        train_features.append(feat)
        train_labels.append(row['handicap_result'])
        train_info.append({
            'date': row['date'],
            'team_id': row['team_id'],
            'team_name': row['team_name'],
            'is_home': row['is_home'],
            'handicap_line': row['win007_handicap_kickoff_line'],
            'handicap_odds': row['win007_handicap_kickoff_odds'],
            'handicap_odds_opponent': row['win007_handicap_kickoff_odds_opponent'],
            'handicap_result': row['handicap_result'],
            'sofascore_match_id': row['sofascore_match_id'],
            'competition': row['competition'],
        })

print(f"训练集有效样本: {len(train_features)}")

print("构建测试集特征...")
test_features = []
test_labels = []
test_info = []

for idx, row in test_df.iterrows():
    feat = build_match_features(row, df_clean)  # 使用所有历史数据构建特征
    if feat is not None:
        test_features.append(feat)
        test_labels.append(row['handicap_result'])
        test_info.append({
            'date': row['date'],
            'team_id': row['team_id'],
            'team_name': row['team_name'],
            'is_home': row['is_home'],
            'handicap_line': row['win007_handicap_kickoff_line'],
            'handicap_odds': row['win007_handicap_kickoff_odds'],
            'handicap_odds_opponent': row['win007_handicap_kickoff_odds_opponent'],
            'handicap_result': row['handicap_result'],
            'sofascore_match_id': row['sofascore_match_id'],
            'competition': row['competition'],
        })

print(f"测试集有效样本: {len(test_features)}")

# 转换为DataFrame
X_train = pd.DataFrame(train_features)
y_train = pd.Series(train_labels)
X_test = pd.DataFrame(test_features)
y_test = pd.Series(test_labels)

# 对齐列
all_cols = list(set(X_train.columns) | set(X_test.columns))
for col in all_cols:
    if col not in X_train.columns:
        X_train[col] = 0
    if col not in X_test.columns:
        X_test[col] = 0

X_train = X_train[sorted(all_cols)]
X_test = X_test[sorted(all_cols)]

# 填充缺失值
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"\n特征数量: {len(all_cols)}")
print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")

# 3. 模型训练
print("\n" + "=" * 60)
print("3. 模型训练 - 使用LightGBM优化ROI")
print("=" * 60)

from sklearn.model_selection import cross_val_predict, StratifiedKFold
import lightgbm as lgb

# 结算赔率（上浮3.7%）
ODDS_MARKUP = 1.037


def calculate_roi(predictions, y_true, info_list, threshold=0.5, bet_type='both'):
    """
    计算投注ROI
    bet_type: 'win' - 只投主队赢盘, 'lose' - 只投主队输盘, 'both' - 两边都投
    """
    total_bet = 0
    total_return = 0
    bet_records = []

    for i, (pred, actual, info) in enumerate(zip(predictions, y_true, info_list)):
        # 结算赔率
        odds_win = info['handicap_odds'] * ODDS_MARKUP
        odds_lose = info['handicap_odds_opponent'] * ODDS_MARKUP

        bet_made = False
        bet_direction = None
        bet_odds = None
        bet_result = None
        profit = 0

        # 判断是否投注
        if bet_type in ['win', 'both'] and pred > threshold:
            # 预测主队赢盘
            total_bet += 1
            bet_made = True
            bet_direction = 'win'
            bet_odds = odds_win

            if actual == 1:  # 实际赢盘
                profit = odds_win
                total_return += 1 + odds_win
                bet_result = 'win'
            elif actual == 0:  # 走水
                profit = 0
                total_return += 1
                bet_result = 'draw'
            else:  # 输盘
                profit = -1
                bet_result = 'lose'

        elif bet_type in ['lose', 'both'] and pred < (1 - threshold):
            # 预测主队输盘（投对手）
            total_bet += 1
            bet_made = True
            bet_direction = 'lose'
            bet_odds = odds_lose

            if actual == -1:  # 实际输盘（对手赢）
                profit = odds_lose
                total_return += 1 + odds_lose
                bet_result = 'win'
            elif actual == 0:  # 走水
                profit = 0
                total_return += 1
                bet_result = 'draw'
            else:  # 主队赢（对手输）
                profit = -1
                bet_result = 'lose'

        if bet_made:
            bet_records.append({
                **info,
                'prediction': pred,
                'bet_direction': bet_direction,
                'bet_odds': bet_odds,
                'bet_result': bet_result,
                'profit': profit,
            })

    roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
    return roi, total_bet, total_return, bet_records


# 将标签转换为二分类（赢盘=1，其他=0）
y_train_binary = (y_train == 1).astype(int)
y_test_binary = (y_test == 1).astype(int)

print("标签分布:")
print(f"训练集: 赢盘={sum(y_train==1)}, 走水={sum(y_train==0)}, 输盘={sum(y_train==-1)}")
print(f"测试集: 赢盘={sum(y_test==1)}, 走水={sum(y_test==0)}, 输盘={sum(y_test==-1)}")

# 使用LightGBM进行训练
print("\n训练LightGBM模型...")

# 寻找最优阈值的函数
def find_optimal_threshold(model, X_val, y_val, info_val):
    """通过交叉验证找到最优阈值"""
    probs = model.predict_proba(X_val)[:, 1]

    best_roi = -float('inf')
    best_threshold = 0.5
    best_bet_type = 'both'

    for threshold in np.arange(0.50, 0.75, 0.02):
        for bet_type in ['win', 'lose', 'both']:
            roi, n_bets, _, _ = calculate_roi(probs, y_val.values, info_val, threshold, bet_type)
            if n_bets >= 20 and roi > best_roi:
                best_roi = roi
                best_threshold = threshold
                best_bet_type = bet_type

    return best_threshold, best_bet_type, best_roi


# 使用不同的模型参数组合
best_model = None
best_cv_roi = -float('inf')
best_params = None
best_threshold = 0.5
best_bet_type = 'both'

param_grid = [
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
    {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.03, 'min_child_samples': 30, 'reg_alpha': 0.2, 'reg_lambda': 0.2},
    {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.02, 'min_child_samples': 20, 'reg_alpha': 0.3, 'reg_lambda': 0.3},
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'min_child_samples': 100, 'reg_alpha': 0.5, 'reg_lambda': 0.5},
    {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.1, 'min_child_samples': 100, 'reg_alpha': 1.0, 'reg_lambda': 1.0},
]

print("\n搜索最优参数...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for params in param_grid:
    model = lgb.LGBMClassifier(
        **params,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    # 使用交叉验证计算ROI
    cv_probs = cross_val_predict(model, X_train, y_train_binary, cv=skf, method='predict_proba')[:, 1]

    # 尝试不同阈值
    for threshold in np.arange(0.50, 0.70, 0.02):
        for bet_type in ['win', 'lose', 'both']:
            roi, n_bets, _, _ = calculate_roi(cv_probs, y_train.values, train_info, threshold, bet_type)

            if n_bets >= 50 and roi > best_cv_roi:
                best_cv_roi = roi
                best_params = params
                best_threshold = threshold
                best_bet_type = bet_type

print(f"\n最优参数: {best_params}")
print(f"最优阈值: {best_threshold}")
print(f"最优投注类型: {best_bet_type}")
print(f"交叉验证ROI: {best_cv_roi:.4f}")

# 使用最优参数训练最终模型
print("\n训练最终模型...")
final_model = lgb.LGBMClassifier(
    **best_params,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)
final_model.fit(X_train, y_train_binary)

# 4. 模型评估
print("\n" + "=" * 60)
print("4. 模型评估")
print("=" * 60)

# 训练集预测
train_probs = final_model.predict_proba(X_train)[:, 1]
train_roi, train_bets, train_return, train_records = calculate_roi(
    train_probs, y_train.values, train_info, best_threshold, best_bet_type
)

print(f"\n训练集:")
print(f"  投注场次: {train_bets}")
print(f"  总投入: {train_bets}")
print(f"  总回报: {train_return:.2f}")
print(f"  ROI: {train_roi:.4f} ({train_roi*100:.2f}%)")

# 测试集预测
test_probs = final_model.predict_proba(X_test)[:, 1]
test_roi, test_bets, test_return, test_records = calculate_roi(
    test_probs, y_test.values, test_info, best_threshold, best_bet_type
)

print(f"\n测试集:")
print(f"  投注场次: {test_bets}")
print(f"  总投入: {test_bets}")
print(f"  总回报: {test_return:.2f}")
print(f"  ROI: {test_roi:.4f} ({test_roi*100:.2f}%)")

# 特征重要性
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

# 所有预测记录
all_records = train_records + test_records
pred_df = pd.DataFrame(all_records)
pred_df['dataset'] = ['train'] * len(train_records) + ['test'] * len(test_records)
pred_df.to_csv('pred_record.csv', index=False)
print(f"预测记录已保存到 pred_record.csv ({len(pred_df)} 条)")

# ROI统计
print("\n计算各维度ROI统计...")

def compute_roi_stats(records, group_col):
    """按维度计算ROI"""
    df = pd.DataFrame(records)
    if len(df) == 0:
        return pd.DataFrame()

    stats = []
    for val in df[group_col].unique():
        subset = df[df[group_col] == val]
        n_bets = len(subset)
        n_wins = sum(subset['bet_result'] == 'win')
        n_draws = sum(subset['bet_result'] == 'draw')
        n_losses = sum(subset['bet_result'] == 'lose')
        total_profit = subset['profit'].sum()
        roi = total_profit / n_bets if n_bets > 0 else 0

        stats.append({
            'dimension': group_col,
            'value': val,
            'n_bets': n_bets,
            'n_wins': n_wins,
            'n_draws': n_draws,
            'n_losses': n_losses,
            'win_rate': n_wins / n_bets if n_bets > 0 else 0,
            'total_profit': total_profit,
            'roi': roi,
        })

    return pd.DataFrame(stats)

# 各维度统计
stats_list = []

# 按数据集
for dataset in ['train', 'test']:
    records = train_records if dataset == 'train' else test_records
    if len(records) > 0:
        df = pd.DataFrame(records)
        n_bets = len(df)
        n_wins = sum(df['bet_result'] == 'win')
        n_draws = sum(df['bet_result'] == 'draw')
        n_losses = sum(df['bet_result'] == 'lose')
        total_profit = df['profit'].sum()
        roi = total_profit / n_bets if n_bets > 0 else 0
        stats_list.append({
            'dimension': 'dataset',
            'value': dataset,
            'n_bets': n_bets,
            'n_wins': n_wins,
            'n_draws': n_draws,
            'n_losses': n_losses,
            'win_rate': n_wins / n_bets,
            'total_profit': total_profit,
            'roi': roi,
        })

# 按投注方向
for direction in ['win', 'lose']:
    records = [r for r in test_records if r['bet_direction'] == direction]
    if len(records) > 0:
        df = pd.DataFrame(records)
        n_bets = len(df)
        n_wins = sum(df['bet_result'] == 'win')
        n_draws = sum(df['bet_result'] == 'draw')
        n_losses = sum(df['bet_result'] == 'lose')
        total_profit = df['profit'].sum()
        roi = total_profit / n_bets if n_bets > 0 else 0
        stats_list.append({
            'dimension': 'bet_direction',
            'value': direction,
            'n_bets': n_bets,
            'n_wins': n_wins,
            'n_draws': n_draws,
            'n_losses': n_losses,
            'win_rate': n_wins / n_bets,
            'total_profit': total_profit,
            'roi': roi,
        })

# 按盘口区间
test_df_records = pd.DataFrame(test_records)
if len(test_df_records) > 0:
    test_df_records['handicap_abs'] = abs(test_df_records['handicap_line'])
    test_df_records['handicap_group'] = pd.cut(
        test_df_records['handicap_abs'],
        bins=[0, 0.25, 0.5, 0.75, 1.0, 1.25],
        labels=['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0', '1.0-1.25']
    )

    for group in test_df_records['handicap_group'].dropna().unique():
        subset = test_df_records[test_df_records['handicap_group'] == group]
        n_bets = len(subset)
        if n_bets > 0:
            n_wins = sum(subset['bet_result'] == 'win')
            n_draws = sum(subset['bet_result'] == 'draw')
            n_losses = sum(subset['bet_result'] == 'lose')
            total_profit = subset['profit'].sum()
            roi = total_profit / n_bets
            stats_list.append({
                'dimension': 'handicap_range',
                'value': str(group),
                'n_bets': n_bets,
                'n_wins': n_wins,
                'n_draws': n_draws,
                'n_losses': n_losses,
                'win_rate': n_wins / n_bets,
                'total_profit': total_profit,
                'roi': roi,
            })

# 按联赛
if 'competition' in test_df_records.columns:
    for comp in test_df_records['competition'].dropna().unique():
        subset = test_df_records[test_df_records['competition'] == comp]
        n_bets = len(subset)
        if n_bets >= 5:  # 至少5场
            n_wins = sum(subset['bet_result'] == 'win')
            n_draws = sum(subset['bet_result'] == 'draw')
            n_losses = sum(subset['bet_result'] == 'lose')
            total_profit = subset['profit'].sum()
            roi = total_profit / n_bets
            stats_list.append({
                'dimension': 'competition',
                'value': comp,
                'n_bets': n_bets,
                'n_wins': n_wins,
                'n_draws': n_draws,
                'n_losses': n_losses,
                'win_rate': n_wins / n_bets,
                'total_profit': total_profit,
                'roi': roi,
            })

# 总体统计
all_test_records = pd.DataFrame(test_records)
if len(all_test_records) > 0:
    n_bets = len(all_test_records)
    n_wins = sum(all_test_records['bet_result'] == 'win')
    n_draws = sum(all_test_records['bet_result'] == 'draw')
    n_losses = sum(all_test_records['bet_result'] == 'lose')
    total_profit = all_test_records['profit'].sum()
    roi = total_profit / n_bets
    stats_list.append({
        'dimension': 'total',
        'value': 'test_set',
        'n_bets': n_bets,
        'n_wins': n_wins,
        'n_draws': n_draws,
        'n_losses': n_losses,
        'win_rate': n_wins / n_bets,
        'total_profit': total_profit,
        'roi': roi,
    })

stats_df = pd.DataFrame(stats_list)
stats_df.to_csv('pred_roi_stats.csv', index=False)
print(f"ROI统计已保存到 pred_roi_stats.csv")

# 打印主要统计
print("\n" + "=" * 60)
print("ROI统计摘要")
print("=" * 60)
print(stats_df.to_string(index=False))

print("\n" + "=" * 60)
print("模型训练完成!")
print("=" * 60)
