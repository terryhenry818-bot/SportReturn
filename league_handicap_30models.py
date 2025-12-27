"""
联赛×盘口 专项模型
针对5大联赛+其他联赛 × 5类盘口 = 30种组合建立专项模型
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

print("=" * 80)
print("联赛×盘口 30种专项模型")
print("=" * 80)

# 定义联赛分组
LEAGUES = {
    'LaLiga': ['LaLiga'],
    'LaLiga 2': ['LaLiga 2'],
    'Serie A': ['Serie A'],
    'Premier League': ['Premier League'],
    'Bundesliga': ['Bundesliga'],
    'Others': None,  # 其他所有联赛
}

MAIN_LEAGUES = ['LaLiga', 'LaLiga 2', 'Serie A', 'Premier League', 'Bundesliga']

# 定义盘口范围
HANDICAP_RANGES = {
    '0': [0.0],
    '0.25': [0.25, -0.25],
    '0.5': [0.5, -0.5],
    '0.75': [0.75, -0.75],
    '1': [1.0, -1.0],
}

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

# 训练集和测试集划分
train_start = datetime(2023, 6, 1)
train_cutoff = datetime(2025, 3, 1)

train_df = df_filtered[(df_filtered['date'] >= train_start) & (df_filtered['date'] <= train_cutoff)].copy()
test_df = df_filtered[df_filtered['date'] > train_cutoff].copy()

print(f"训练集: {len(train_df)} 条")
print(f"测试集: {len(test_df)} 条")

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

# 投注参数 (统一使用)
BET_PARAMS = {
    'min_edge': 0.10,
    'max_edge': 0.20,
    'min_odds': 0.80,
    'max_odds': 1.15,
    'vt': 0.10,
    'only_positive': True,  # 只投注正盘口
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
    only_positive = params.get('only_positive', True)

    for i, (prob, actual, info) in enumerate(zip(model_probs, y_true, info_list)):
        if only_positive and info['handicap_line'] < -0.001:
            continue

        odds_win = info['handicap_odds'] * ODDS_MARKUP
        odds_lose = info['handicap_odds_opponent'] * ODDS_MARKUP
        market_prob_win = 1 / (1 + info['handicap_odds'])
        market_prob_lose = 1 / (1 + info['handicap_odds_opponent'])
        goal_diff = info.get('goal_diff', 0)

        bet_made = False
        profit = 0

        if prob > market_prob_win + vt:
            edge = prob - market_prob_win
            if edge < min_edge or edge > max_edge:
                continue
            if odds_win < min_odds or odds_win > max_odds:
                continue

            total_bet += 1
            bet_made = True
            bet_result, profit_mult = calculate_handicap_outcome(goal_diff, info['handicap_line'], 'win')

            if profit_mult > 0:
                profit = profit_mult * odds_win
                total_return += 1 + profit
            elif profit_mult == 0:
                total_return += 1
            else:
                profit = profit_mult
                total_return += 1 + profit

        if bet_made:
            bet_records.append({**info, 'profit': profit})

    roi = (total_return - total_bet) / total_bet if total_bet > 0 else 0
    return roi, total_bet, total_return, bet_records


def train_ensemble_model(X_train, y_train):
    models = {}

    lgb_model = lgb.LGBMClassifier(
        n_estimators=30, max_depth=2, learning_rate=0.03,
        min_child_samples=50, reg_alpha=3.0, reg_lambda=3.0,
        subsample=0.5, colsample_bytree=0.5,
        random_state=42, verbose=-1, n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_cal = CalibratedClassifierCV(lgb_model, method='isotonic', cv=3)
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
        xgb_cal = CalibratedClassifierCV(xgb_model, method='isotonic', cv=3)
        xgb_cal.fit(X_train, y_train)
        models['XGBoost'] = xgb_cal

    rf_model = RandomForestClassifier(
        n_estimators=50, max_depth=3, min_samples_leaf=30,
        max_features=0.5, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_cal = CalibratedClassifierCV(rf_model, method='isotonic', cv=3)
    rf_cal.fit(X_train, y_train)
    models['RandomForest'] = rf_cal

    lr_model = LogisticRegression(C=0.05, max_iter=1000, random_state=42, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    lr_cal = CalibratedClassifierCV(lr_model, method='isotonic', cv=3)
    lr_cal.fit(X_train, y_train)
    models['LogisticRegression'] = lr_cal

    return models


def ensemble_predict(models, X):
    all_probs = []
    for name, model in models.items():
        probs = model.predict_proba(X)[:, 1]
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)


# 构建特征
print("\n构建特征...")
train_features, train_labels, train_info = build_dataset(train_df, df_clean)
test_features, test_labels, test_info = build_dataset(test_df, df_clean)

print(f"训练特征: {len(train_features)} 样本")
print(f"测试特征: {len(test_features)} 样本")

# 转换为DataFrame
train_feat_df = pd.DataFrame(train_features)
test_feat_df = pd.DataFrame(test_features)
train_info_df = pd.DataFrame(train_info)
test_info_df = pd.DataFrame(test_info)

# 存储结果
results = []
all_bet_records = []

print("\n" + "=" * 80)
print("训练30种 联赛×盘口 专项模型")
print("=" * 80)

# 遍历每个联赛×盘口组合
for league_name, league_list in LEAGUES.items():
    for handicap_name, handicap_values in HANDICAP_RANGES.items():

        # 筛选训练数据
        if league_list is None:  # Others
            train_league_mask = ~train_info_df['competition'].isin(MAIN_LEAGUES)
            test_league_mask = ~test_info_df['competition'].isin(MAIN_LEAGUES)
        else:
            train_league_mask = train_info_df['competition'].isin(league_list)
            test_league_mask = test_info_df['competition'].isin(league_list)

        # 筛选盘口 (只正盘口)
        train_handicap_mask = train_info_df['handicap_line'].apply(
            lambda x: x > 0 and any(abs(x - h) < 0.001 for h in handicap_values)
        )
        test_handicap_mask = test_info_df['handicap_line'].apply(
            lambda x: x > 0 and any(abs(x - h) < 0.001 for h in handicap_values)
        )

        train_mask = train_league_mask & train_handicap_mask
        test_mask = test_league_mask & test_handicap_mask

        train_count = train_mask.sum()
        test_count = test_mask.sum()

        # 样本量检查
        if train_count < 30 or test_count < 10:
            results.append({
                'league': league_name,
                'handicap': handicap_name,
                'train_samples': train_count,
                'test_samples': test_count,
                'test_bets': 0,
                'test_roi': None,
                'status': 'SKIP (样本不足)'
            })
            continue

        # 准备数据
        X_train = train_feat_df[train_mask].fillna(0)
        y_train = np.array([1 if train_labels[i] == 1 else 0 for i in train_mask[train_mask].index])
        train_info_subset = [train_info[i] for i in train_mask[train_mask].index]

        X_test = test_feat_df[test_mask].fillna(0)
        y_test = np.array([test_labels[i] for i in test_mask[test_mask].index])
        test_info_subset = [test_info[i] for i in test_mask[test_mask].index]

        # 对齐特征
        common_cols = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]

        try:
            # 训练模型
            models = train_ensemble_model(X_train, y_train)

            # 预测
            test_probs = ensemble_predict(models, X_test)

            # 计算ROI
            roi, bets, returns, bet_records = calculate_value_betting_roi(
                test_probs, y_test, test_info_subset, BET_PARAMS
            )

            # 记录结果
            results.append({
                'league': league_name,
                'handicap': handicap_name,
                'train_samples': train_count,
                'test_samples': test_count,
                'test_bets': bets,
                'test_roi': roi,
                'status': 'OK'
            })

            # 记录投注
            for rec in bet_records:
                rec['model_league'] = league_name
                rec['model_handicap'] = handicap_name
                all_bet_records.append(rec)

            roi_str = f"{roi*100:+.2f}%" if bets > 0 else "N/A"
            print(f"  {league_name:15} × ±{handicap_name:5}: 训练{train_count:4}, 测试{test_count:4}, 投注{bets:3}, ROI {roi_str}")

        except Exception as e:
            results.append({
                'league': league_name,
                'handicap': handicap_name,
                'train_samples': train_count,
                'test_samples': test_count,
                'test_bets': 0,
                'test_roi': None,
                'status': f'ERROR: {str(e)[:30]}'
            })

# 输出汇总结果
print("\n" + "=" * 80)
print("30种模型ROI汇总")
print("=" * 80)

results_df = pd.DataFrame(results)

# 按联赛和盘口排序
league_order = ['LaLiga', 'LaLiga 2', 'Serie A', 'Premier League', 'Bundesliga', 'Others']
handicap_order = ['0', '0.25', '0.5', '0.75', '1']

results_df['league_order'] = results_df['league'].apply(lambda x: league_order.index(x) if x in league_order else 99)
results_df['handicap_order'] = results_df['handicap'].apply(lambda x: handicap_order.index(x) if x in handicap_order else 99)
results_df = results_df.sort_values(['league_order', 'handicap_order'])

print(f"\n{'联赛':<18} {'盘口':<8} {'训练':>8} {'测试':>8} {'投注':>8} {'ROI':>10} {'状态':<15}")
print("-" * 90)

for _, row in results_df.iterrows():
    roi_str = f"{row['test_roi']*100:+.2f}%" if row['test_roi'] is not None else "N/A"
    print(f"{row['league']:<18} ±{row['handicap']:<6} {row['train_samples']:>8} {row['test_samples']:>8} {row['test_bets']:>8} {roi_str:>10} {row['status']:<15}")

# 按联赛汇总
print("\n" + "=" * 80)
print("按联赛汇总")
print("=" * 80)

valid_results = results_df[results_df['test_bets'] > 0]
league_summary = valid_results.groupby('league').agg({
    'test_bets': 'sum',
    'test_roi': lambda x: np.average(x, weights=valid_results.loc[x.index, 'test_bets']) if valid_results.loc[x.index, 'test_bets'].sum() > 0 else 0
}).reset_index()
league_summary.columns = ['联赛', '总投注', '加权ROI']
league_summary = league_summary.sort_values('加权ROI', ascending=False)

print(f"\n{'联赛':<18} {'总投注':>10} {'加权ROI':>12}")
print("-" * 45)
for _, row in league_summary.iterrows():
    print(f"{row['联赛']:<18} {int(row['总投注']):>10} {row['加权ROI']*100:>+11.2f}%")

# 按盘口汇总
print("\n" + "=" * 80)
print("按盘口汇总")
print("=" * 80)

handicap_summary = valid_results.groupby('handicap').agg({
    'test_bets': 'sum',
    'test_roi': lambda x: np.average(x, weights=valid_results.loc[x.index, 'test_bets']) if valid_results.loc[x.index, 'test_bets'].sum() > 0 else 0
}).reset_index()
handicap_summary.columns = ['盘口', '总投注', '加权ROI']
handicap_summary['handicap_order'] = handicap_summary['盘口'].apply(lambda x: handicap_order.index(x) if x in handicap_order else 99)
handicap_summary = handicap_summary.sort_values('handicap_order')

print(f"\n{'盘口':<10} {'总投注':>10} {'加权ROI':>12}")
print("-" * 35)
for _, row in handicap_summary.iterrows():
    print(f"±{row['盘口']:<8} {int(row['总投注']):>10} {row['加权ROI']*100:>+11.2f}%")

# 总体汇总
total_bets = valid_results['test_bets'].sum()
if total_bets > 0:
    total_roi = np.average(valid_results['test_roi'], weights=valid_results['test_bets'])
    print(f"\n{'总计':<10} {int(total_bets):>10} {total_roi*100:>+11.2f}%")

# 保存结果
results_df.to_csv('league_handicap_30models_results.csv', index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 league_handicap_30models_results.csv")

if all_bet_records:
    bet_df = pd.DataFrame(all_bet_records)
    bet_df.to_csv('league_handicap_30models_bets.csv', index=False, encoding='utf-8-sig')
    print(f"投注记录已保存到 league_handicap_30models_bets.csv")

print("\n完成!")
