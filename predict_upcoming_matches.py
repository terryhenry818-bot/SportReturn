"""
未来比赛高价值投注预测脚本
基于 asian_handicap_multimodel.py 的多模型组合算法
读取历史数据构建特征，预测 upcoming_wide_table.csv 中的高价值投注
支持邮件通知功能
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ============ 邮件配置 ============
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': os.environ.get('SMTP_EMAIL', ''),
    'sender_password': os.environ.get('SMTP_PASSWORD', ''),
    'recipient_email': 'terryhenry818@gmail.com',
}


def send_email_notification(subject, body_text, body_html=None, attachments=None):
    """
    发送邮件通知

    Args:
        subject: 邮件主题
        body_text: 纯文本正文
        body_html: HTML正文 (可选)
        attachments: 附件文件路径列表 (可选)

    Returns:
        bool: 发送成功返回True，否则返回False
    """
    sender = EMAIL_CONFIG['sender_email']
    password = EMAIL_CONFIG['sender_password']
    recipient = EMAIL_CONFIG['recipient_email']

    if not sender or not password:
        print("    警告: 邮件配置不完整 (需要设置 SMTP_EMAIL 和 SMTP_PASSWORD 环境变量)")
        return False

    try:
        # 创建邮件
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = recipient

        # 添加正文
        part1 = MIMEText(body_text, 'plain', 'utf-8')
        msg.attach(part1)

        if body_html:
            part2 = MIMEText(body_html, 'html', 'utf-8')
            msg.attach(part2)

        # 添加附件
        if attachments:
            for filepath in attachments:
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    filename = os.path.basename(filepath)
                    part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                    msg.attach(part)

        # 发送邮件
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())

        print(f"    邮件已发送至: {recipient}")
        return True

    except Exception as e:
        print(f"    邮件发送失败: {e}")
        return False


def generate_email_content(pred_df, value_bets):
    """
    生成邮件内容

    Args:
        pred_df: 所有预测的DataFrame
        value_bets: 高价值投注的DataFrame

    Returns:
        tuple: (text_content, html_content)
    """
    today = datetime.now().strftime('%Y-%m-%d')

    # 纯文本内容
    text_lines = [
        f"足球亚盘高价值投注预测报告",
        f"生成时间: {today}",
        f"",
        f"=" * 60,
        f"统计摘要",
        f"=" * 60,
        f"总预测数: {len(pred_df)}",
        f"高价值投注: {len(value_bets)} 注",
        f"正边际投注: {len(pred_df[pred_df['edge'] > 0])} 注",
        f"",
    ]

    if len(value_bets) > 0:
        text_lines.append("=" * 60)
        text_lines.append("高价值投注推荐")
        text_lines.append("=" * 60)
        text_lines.append("")
        text_lines.append(f"{'日期':<12} {'联赛':<15} {'球队':<15} {'主客':<4} {'盘口':>6} {'赔率':>5} {'预测':>6} {'边际':>7}")
        text_lines.append("-" * 80)

        for _, row in value_bets.iterrows():
            text_lines.append(
                f"{str(row['date'])[:10]:<12} {str(row['competition'])[:13]:<15} {str(row['team_name'])[:13]:<15} "
                f"{row['is_home']:<4} {row['handicap_line']:>+6.2f} {row['handicap_odds']:>5.2f} "
                f"{row['pred_prob']*100:>5.1f}% {row['edge']*100:>+6.1f}%"
            )
    else:
        text_lines.append("暂无高价值投注推荐")

    text_content = "\n".join(text_lines)

    # HTML内容
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .positive {{ color: green; font-weight: bold; }}
            .negative {{ color: red; }}
            .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>⚽ 足球亚盘高价值投注预测报告</h1>
        <p>生成时间: {today}</p>

        <div class="summary">
            <h3>统计摘要</h3>
            <ul>
                <li>总预测数: <strong>{len(pred_df)}</strong></li>
                <li>高价值投注: <strong class="positive">{len(value_bets)} 注</strong></li>
                <li>正边际投注: <strong>{len(pred_df[pred_df['edge'] > 0])} 注</strong></li>
            </ul>
        </div>
    """

    if len(value_bets) > 0:
        html_content += """
        <h2>🎯 高价值投注推荐</h2>
        <table>
            <tr>
                <th>日期</th>
                <th>联赛</th>
                <th>球队</th>
                <th>主客</th>
                <th>盘口</th>
                <th>赔率</th>
                <th>预测概率</th>
                <th>边际</th>
            </tr>
        """

        for _, row in value_bets.iterrows():
            edge_class = 'positive' if row['edge'] > 0 else 'negative'
            html_content += f"""
            <tr>
                <td>{str(row['date'])[:10]}</td>
                <td>{row['competition']}</td>
                <td>{row['team_name']}</td>
                <td>{row['is_home']}</td>
                <td>{row['handicap_line']:+.2f}</td>
                <td>{row['handicap_odds']:.2f}</td>
                <td>{row['pred_prob']*100:.1f}%</td>
                <td class="{edge_class}">{row['edge']*100:+.1f}%</td>
            </tr>
            """

        html_content += "</table>"
    else:
        html_content += "<p><em>暂无高价值投注推荐</em></p>"

    html_content += """
    </body>
    </html>
    """

    return text_content, html_content


print("=" * 70)
print("未来比赛高价值投注预测")
print("=" * 70)

# ============ 1. 加载数据 ============
print("\n[1] 加载数据...")

# 历史数据 - 用于训练和构建特征
df_history = pd.read_csv('wide_table.csv')
df_history['date'] = pd.to_datetime(df_history['date'])
print(f"    历史数据: {len(df_history)} 条")

# 未来比赛数据
df_upcoming = pd.read_csv('upcoming_wide_table.csv')
df_upcoming['date'] = pd.to_datetime(df_upcoming['date'])
print(f"    未来比赛: {len(df_upcoming)} 条 ({len(df_upcoming)//2} 场)")

# 构建球队名称到ID的映射 (从历史数据中获取)
print("\n[1.1] 构建球队名称映射...")
team_name_to_id = {}
for _, row in df_history.drop_duplicates('team_id').iterrows():
    team_name_to_id[row['team_name']] = row['team_id']
print(f"    已知球队: {len(team_name_to_id)} 支")

# 修复 upcoming 中的 team_id
fixed_count = 0
for idx, row in df_upcoming.iterrows():
    team_name = row['team_name']
    if team_name in team_name_to_id:
        df_upcoming.loc[idx, 'team_id'] = team_name_to_id[team_name]
        fixed_count += 1
    else:
        print(f"    警告: 未找到球队 '{team_name}' 的历史记录")

print(f"    已修复team_id: {fixed_count}/{len(df_upcoming)} 条")

# ============ 2. 配置 ============
# 盘口范围定义
HANDICAP_RANGES = {
    '0': [0.0],
    '0.25': [0.25, -0.25],
    '0.5': [0.5, -0.5],
    '0.75': [0.75, -0.75],
    '1': [1.0, -1.0],
    '1.25': [1.25, -1.25],
}

# 每个盘口范围的参数
RANGE_PARAMS = {
    '0': {'only_positive': False, 'threshold': 0.52},
    '0.25': {'only_positive': True, 'threshold': 0.53},
    '0.5': {'only_positive': True, 'threshold': 0.52},
    '0.75': {'only_positive': True, 'threshold': 0.52},
    '1': {'only_positive': True, 'threshold': 0.52},
    '1.25': {'only_positive': True, 'threshold': 0.52},
}

# 排除表现差的联赛
BAD_LEAGUES = [
    'Championship', '2. Bundesliga', 'Ligue 1', 'Ligue 2', 'Club Friendly Games',
]

# 有效盘口
valid_lines = []
for lines in HANDICAP_RANGES.values():
    valid_lines.extend(lines)

def is_valid_line(line):
    if pd.isna(line):
        return False
    for valid in valid_lines:
        if abs(line - valid) < 0.001:
            return True
    return False

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

# ============ 3. 特征工程 ============
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

    # 使用 kickoff 盘口，如果没有则使用 early 盘口
    handicap_line = row.get('win007_handicap_kickoff_line')
    handicap_odds = row.get('win007_handicap_kickoff_odds')
    handicap_odds_opp = row.get('win007_handicap_kickoff_odds_opponent')

    if pd.isna(handicap_line):
        handicap_line = row.get('win007_handicap_early_line')
        handicap_odds = row.get('win007_handicap_early_odds')
        handicap_odds_opp = row.get('win007_handicap_early_odds_opponent')

    if pd.isna(handicap_line) or pd.isna(handicap_odds):
        return None

    features['handicap_line'] = handicap_line
    features['handicap_odds'] = handicap_odds
    features['handicap_odds_opponent'] = handicap_odds_opp

    features['implied_prob'] = 1 / (1 + handicap_odds) if handicap_odds > 0 else 0.5
    features['odds_ratio'] = handicap_odds / handicap_odds_opp if handicap_odds_opp > 0 else 1

    if pd.notna(row.get('win007_handicap_early_odds')):
        features['early_odds'] = row['win007_handicap_early_odds']
        features['odds_drift'] = handicap_odds - row['win007_handicap_early_odds']

    # 欧赔概率
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

    # 大小球盘口
    overunder_line = row.get('win007_overunder_kickoff_line')
    if pd.isna(overunder_line):
        overunder_line = row.get('win007_overunder_early_line')
    if pd.notna(overunder_line):
        features['overunder_line'] = overunder_line

    features['is_home'] = row['is_home']

    # 球队历史特征
    team_hist = build_team_features(df_all, row['date'], row['team_id'], row['is_home'])
    if team_hist is None:
        return None

    for k, v in team_hist.items():
        features[f'team_{k}'] = v

    # 对手历史特征
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


def build_dataset(df_source, df_all_for_history, include_label=True):
    """构建数据集"""
    features_list = []
    info_list = []
    labels_list = []

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

                # 使用 kickoff 盘口，如果没有则使用 early 盘口
                handicap_line = row.get('win007_handicap_kickoff_line')
                handicap_odds = row.get('win007_handicap_kickoff_odds')
                handicap_odds_opp = row.get('win007_handicap_kickoff_odds_opponent')

                if pd.isna(handicap_line):
                    handicap_line = row.get('win007_handicap_early_line')
                    handicap_odds = row.get('win007_handicap_early_odds')
                    handicap_odds_opp = row.get('win007_handicap_early_odds_opponent')

                info_list.append({
                    'date': row['date'],
                    'match_id': match_id,
                    'team_id': row['team_id'],
                    'team_name': row['team_name'],
                    'is_home': row['is_home'],
                    'competition': row['competition'],
                    'handicap_line': handicap_line,
                    'handicap_odds': handicap_odds,
                    'handicap_odds_opponent': handicap_odds_opp,
                })

                if include_label and 'handicap_result' in row:
                    labels_list.append(row['handicap_result'])

    if not features_list:
        return None, None, None

    feature_df = pd.DataFrame(features_list)
    info_df = pd.DataFrame(info_list)
    labels = np.array(labels_list) if labels_list else None

    return feature_df, info_df, labels


# ============ 4. 准备训练数据 ============
print("\n[2] 准备训练数据...")

# 过滤历史数据
df_history_filtered = df_history.dropna(subset=['win007_handicap_kickoff_line', 'handicap_result'])
df_history_filtered = df_history_filtered[df_history_filtered['win007_handicap_kickoff_line'].apply(is_valid_line)]
df_history_filtered = df_history_filtered[~df_history_filtered['competition'].isin(BAD_LEAGUES)]

# 使用最近的数据作为训练集
train_start = datetime(2023, 6, 1)
train_cutoff = datetime(2025, 3, 1)
train_df = df_history_filtered[
    (df_history_filtered['date'] >= train_start) &
    (df_history_filtered['date'] <= train_cutoff)
].copy()

print(f"    训练集范围: {train_start.date()} ~ {train_cutoff.date()}")
print(f"    训练集记录: {len(train_df)} 条")

# 构建训练特征
train_features, train_info, train_labels = build_dataset(train_df, df_history, include_label=True)
print(f"    构建训练特征: {len(train_features)} 条")


# ============ 5. 准备预测数据 ============
print("\n[3] 准备预测数据...")

# 过滤未来比赛
df_upcoming_filtered = df_upcoming.copy()

# 检查盘口有效性
def get_effective_line(row):
    line = row.get('win007_handicap_kickoff_line')
    if pd.isna(line):
        line = row.get('win007_handicap_early_line')
    return line

df_upcoming_filtered['effective_line'] = df_upcoming_filtered.apply(get_effective_line, axis=1)
df_upcoming_filtered = df_upcoming_filtered[df_upcoming_filtered['effective_line'].apply(is_valid_line)]
df_upcoming_filtered = df_upcoming_filtered[~df_upcoming_filtered['competition'].isin(BAD_LEAGUES)]

print(f"    有效未来比赛: {len(df_upcoming_filtered)} 条 ({len(df_upcoming_filtered)//2} 场)")

# 构建预测特征
pred_features, pred_info, _ = build_dataset(df_upcoming_filtered, df_history, include_label=False)

if pred_features is None or len(pred_features) == 0:
    print("    警告: 没有可预测的比赛!")
    exit()

print(f"    构建预测特征: {len(pred_features)} 条")


# ============ 6. 训练模型并预测 ============
print("\n[4] 训练模型并预测...")

# 存储预测结果
predictions = []

for range_name, params in RANGE_PARAMS.items():
    range_lines = HANDICAP_RANGES[range_name]
    threshold = params['threshold']
    only_positive = params['only_positive']

    # 筛选训练数据
    if only_positive:
        train_mask = train_info['handicap_line'].apply(
            lambda x: x > 0 and any(abs(x - l) < 0.001 for l in range_lines))
    else:
        train_mask = train_info['handicap_line'].apply(
            lambda x: any(abs(x - l) < 0.001 for l in range_lines))

    X_train = train_features[train_mask]
    y_train = (train_labels[train_mask] == 1).astype(int)

    if len(X_train) < 50:
        continue

    # 筛选预测数据
    if only_positive:
        pred_mask = pred_info['handicap_line'].apply(
            lambda x: x > 0 and any(abs(x - l) < 0.001 for l in range_lines))
    else:
        pred_mask = pred_info['handicap_line'].apply(
            lambda x: any(abs(x - l) < 0.001 for l in range_lines))

    if pred_mask.sum() == 0:
        continue

    X_pred = pred_features[pred_mask]
    pred_info_subset = pred_info[pred_mask]

    # 确保特征列一致
    common_cols = list(set(X_train.columns) & set(X_pred.columns))
    X_train_aligned = X_train[common_cols].fillna(0)
    X_pred_aligned = X_pred[common_cols].fillna(0)

    # 训练模型集成
    models = []

    lgb_model = lgb.LGBMClassifier(
        n_estimators=30, max_depth=2, learning_rate=0.03,
        min_child_samples=50, subsample=0.5, colsample_bytree=0.5,
        reg_alpha=3.0, reg_lambda=3.0, random_state=42, n_jobs=-1, verbosity=-1
    )
    lgb_model.fit(X_train_aligned, y_train)
    models.append(lgb_model)

    if HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=30, max_depth=2, learning_rate=0.03,
            min_child_weight=50, subsample=0.5, colsample_bytree=0.5,
            reg_alpha=2.0, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=0
        )
        xgb_model.fit(X_train_aligned, y_train)
        models.append(xgb_model)

    rf_model = RandomForestClassifier(
        n_estimators=30, max_depth=3, min_samples_leaf=50,
        max_features=0.3, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_aligned, y_train)
    models.append(rf_model)

    # 集成预测
    pred_probs = np.zeros(len(X_pred_aligned))
    for model in models:
        pred_probs += model.predict_proba(X_pred_aligned)[:, 1]
    pred_probs /= len(models)

    # 收集所有预测（不只是高价值）
    for i, (idx, row) in enumerate(pred_info_subset.iterrows()):
        prob = pred_probs[i]
        implied_prob = 1 / (1 + row['handicap_odds']) if row['handicap_odds'] > 0 else 0.5
        predictions.append({
            'date': row['date'],
            'competition': row['competition'],
            'team_name': row['team_name'],
            'is_home': '主' if row['is_home'] == 1 else '客',
            'handicap_line': row['handicap_line'],
            'handicap_odds': row['handicap_odds'],
            'pred_prob': prob,
            'implied_prob': implied_prob,
            'threshold': threshold,
            'edge': prob - implied_prob,
            'is_value_bet': prob >= threshold,
            'confidence': 'HIGH' if prob >= threshold + 0.05 else ('MEDIUM' if prob >= threshold else 'LOW'),
            'range': range_name,
        })

    print(f"    盘口 {range_name}: 训练{len(X_train)}条, 预测{len(X_pred)}条, 推荐{sum(1 for p in predictions if p['range'] == range_name)}注")


# ============ 7. 输出结果 ============
print("\n[5] 输出结果...")

if not predictions:
    print("    没有预测结果!")
else:
    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values(['is_value_bet', 'edge'], ascending=[False, False])

    # 保存所有预测到CSV
    output_file = 'upcoming_predictions.csv'
    pred_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"    所有预测已保存到: {output_file}")

    # 筛选高价值投注
    value_bets = pred_df[pred_df['is_value_bet'] == True]

    if len(value_bets) > 0:
        value_file = 'upcoming_high_value_bets.csv'
        value_bets.to_csv(value_file, index=False, encoding='utf-8-sig')
        print(f"    高价值投注已保存到: {value_file}")

    # 打印所有预测
    print("\n" + "=" * 90)
    print("所有预测结果 (按边际值排序)")
    print("=" * 90)

    print(f"\n{'日期':<12} {'联赛':<18} {'球队':<18} {'主客':<4} {'盘口':>6} {'赔率':>5} {'预测':>6} {'隐含':>6} {'边际':>7} {'推荐':<4}")
    print("-" * 100)

    for _, row in pred_df.iterrows():
        recommend = "★" if row['is_value_bet'] else ""
        print(f"{str(row['date'])[:10]:<12} {row['competition'][:16]:<18} {row['team_name'][:16]:<18} "
              f"{row['is_home']:<4} {row['handicap_line']:>+6.2f} {row['handicap_odds']:>5.2f} "
              f"{row['pred_prob']*100:>5.1f}% {row['implied_prob']*100:>5.1f}% {row['edge']*100:>+6.1f}% {recommend:<4}")

    # 统计
    print("\n" + "=" * 90)
    print("统计汇总")
    print("=" * 90)
    print(f"\n总预测数: {len(pred_df)}")
    print(f"高价值投注: {len(value_bets)} 注 (边际>0 且 概率>阈值)")
    print(f"正边际投注: {len(pred_df[pred_df['edge'] > 0])} 注")

    # 按联赛统计
    print("\n按联赛分布:")
    league_stats = pred_df.groupby('competition').agg({
        'edge': 'mean',
        'is_value_bet': 'sum'
    }).sort_values('edge', ascending=False)
    for league, stats in league_stats.iterrows():
        print(f"  {league}: 平均边际 {stats['edge']*100:+.1f}%, 推荐 {int(stats['is_value_bet'])} 注")

    # ============ 8. 发送邮件通知 ============
    print("\n[6] 发送邮件通知...")

    # 生成邮件内容
    text_content, html_content = generate_email_content(pred_df, value_bets)

    # 邮件主题
    today = datetime.now().strftime('%Y-%m-%d')
    subject = f"[足球预测] {today} 亚盘高价值投注推荐 ({len(value_bets)}注)"

    # 附件
    attachments = []
    if os.path.exists(output_file):
        attachments.append(output_file)
    if len(value_bets) > 0 and os.path.exists(value_file):
        attachments.append(value_file)

    # 发送邮件
    send_email_notification(subject, text_content, html_content, attachments)

print("\n完成!")
