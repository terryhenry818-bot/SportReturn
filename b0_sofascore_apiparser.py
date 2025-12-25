#!/usr/bin/env python3
"""
SofaScore API Data Extractor v2.1 (Fixed)
==========================================
修复版本 - 解决以下问题:
1. lineups.csv 球员统计字段为空 (Accurate_passes, Duels_won等)
2. formation.csv 位置坐标为空 (player_pos_x, player_pos_y)
3. 新增 shotmap.csv (xG, xGOT, Outcome, Situation等)
4. 新增 player_match_stats.csv (Rating breakdown, 详细统计)

API端点:
- /event/{id} - 基本信息
- /event/{id}/lineups - 阵容和球员统计
- /event/{id}/shotmap - 射门图
- /event/{id}/statistics - 统计数据
- /event/{id}/graph - 动量图
- /event/{id}/odds/1/all - 赔率
"""

import os
import json
import time
import random
import logging
import requests
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SofaScoreAPI:
    """SofaScore API 数据提取器 (修复版)"""
    
    BASE_URL = "https://api.sofascore.com/api/v1"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Origin': 'https://www.sofascore.com',
        'Referer': 'https://www.sofascore.com/',
        'Cache-Control': 'no-cache',
    }
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        os.makedirs(output_dir, exist_ok=True)
    
    def _get(self, endpoint: str, event_id: int = None) -> Optional[Dict]:
        """发送GET请求"""
        if event_id:
            if endpoint:
                url = f"{self.BASE_URL}/event/{event_id}/{endpoint}"
            else:
                url = f"{self.BASE_URL}/event/{event_id}"
        else:
            url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"API返回 {resp.status_code}: {url}")
        except Exception as e:
            logger.error(f"API请求失败: {url} - {e}")
        return None
    
    # ==================== 1. Basic Info ====================
    
    def get_basic_info(self, event_id: int) -> Dict:
        """获取比赛基本信息"""
        data = self._get('', event_id)
        if not data:
            return {}
        
        event = data.get('event', data)
        
        home_team = event.get('homeTeam', {})
        away_team = event.get('awayTeam', {})
        tournament = event.get('tournament', {})
        venue = event.get('venue', {})
        
        # 转换时间戳
        start_ts = event.get('startTimestamp', 0)
        start_time = datetime.fromtimestamp(start_ts).strftime('%d/%m/%Y %H:%M') if start_ts else ''
        
        # 获取球场名称
        stadium = ''
        if venue:
            if isinstance(venue.get('stadium'), dict):
                stadium = venue['stadium'].get('name', '')
            elif isinstance(venue.get('stadium'), str):
                stadium = venue['stadium']
            elif venue.get('city'):
                stadium = venue['city'].get('name', '') if isinstance(venue['city'], dict) else venue['city']
        
        return {
            'match_id': event_id,
            'home_team_id': home_team.get('id', ''),
            'away_team_id': away_team.get('id', ''),
            'home_team_name': home_team.get('name', ''),
            'away_team_name': away_team.get('name', ''),
            'match_start_time': start_time,
            'league_name': tournament.get('name', ''),
            'stadium': stadium
        }
    
    # ==================== 2. Odds ====================
    
    def get_odds(self, event_id: int) -> Dict:
        """获取赔率数据"""
        odds_data = {}
        
        endpoints = ['odds/1/all', 'odds/1/featured', 'provider/1/odds']
        
        raw_data = None
        for endpoint in endpoints:
            data = self._get(endpoint, event_id)
            if data:
                raw_data = data
                break
        
        if not raw_data:
            return odds_data
        
        markets = raw_data.get('markets', [])
        
        for market in markets:
            market_name = market.get('marketName', '').lower()
            choices = market.get('choices', [])
            
            if 'asian' in market_name and 'handicap' in market_name:
                asian_data = {}
                for choice in choices:
                    name = choice.get('name', '')
                    handicap = choice.get('handicap', '')
                    odds_val = choice.get('fractionalValue', '')
                    key = f"({handicap}){name}" if handicap else name
                    if odds_val:
                        try:
                            asian_data[key] = float(odds_val)
                        except:
                            asian_data[key] = odds_val
                if asian_data:
                    odds_data['Asian handicap'] = asian_data
            
            elif ('1x2' in market_name or 'full time' in market_name or 'full-time' in market_name) and 'half' not in market_name:
                ft_data = {}
                for choice in choices:
                    name = choice.get('name', '')
                    odds_val = choice.get('fractionalValue', '')
                    if name.lower() in ['1', 'home']:
                        key = '1'
                    elif name.lower() in ['x', 'draw']:
                        key = 'X'
                    elif name.lower() in ['2', 'away']:
                        key = '2'
                    else:
                        key = name
                    if odds_val:
                        try:
                            ft_data[key] = float(odds_val)
                        except:
                            ft_data[key] = odds_val
                if ft_data:
                    odds_data['Full-time'] = ft_data
            
            elif 'double chance' in market_name:
                dc_data = {}
                for choice in choices:
                    name = choice.get('name', '')
                    odds_val = choice.get('fractionalValue', '')
                    if odds_val:
                        try:
                            dc_data[name] = float(odds_val)
                        except:
                            dc_data[name] = odds_val
                if dc_data:
                    odds_data['Double chance'] = dc_data
            
            elif '1st half' in market_name or 'first half' in market_name:
                half_data = {}
                for choice in choices:
                    name = choice.get('name', '')
                    odds_val = choice.get('fractionalValue', '')
                    if name.lower() in ['1', 'home']:
                        key = '1'
                    elif name.lower() in ['x', 'draw']:
                        key = 'X'
                    elif name.lower() in ['2', 'away']:
                        key = '2'
                    else:
                        key = name
                    if odds_val:
                        try:
                            half_data[key] = float(odds_val)
                        except:
                            half_data[key] = odds_val
                if half_data:
                    odds_data['1st half'] = half_data
            
            elif 'draw no bet' in market_name or 'draw not bet' in market_name:
                dnb_data = {}
                for choice in choices:
                    name = choice.get('name', '')
                    odds_val = choice.get('fractionalValue', '')
                    if name.lower() in ['1', 'home']:
                        key = '1'
                    elif name.lower() in ['2', 'away']:
                        key = '2'
                    else:
                        key = name
                    if odds_val:
                        try:
                            dnb_data[key] = float(odds_val)
                        except:
                            dnb_data[key] = odds_val
                if dnb_data:
                    odds_data['Draw not bet'] = dnb_data
            
            elif 'both teams' in market_name or 'btts' in market_name:
                btts_data = {}
                for choice in choices:
                    name = choice.get('name', '')
                    odds_val = choice.get('fractionalValue', '')
                    if odds_val:
                        try:
                            btts_data[name] = float(odds_val)
                        except:
                            btts_data[name] = odds_val
                if btts_data:
                    odds_data['Both teams to score'] = btts_data
            
            elif 'over' in market_name or 'under' in market_name or 'total' in market_name:
                if 'Match goals' not in odds_data:
                    odds_data['Match goals'] = []
                
                current_items = {}
                for choice in choices:
                    name = choice.get('name', '')
                    handicap = choice.get('handicap', '')
                    odds_val = choice.get('fractionalValue', '')
                    
                    if odds_val and handicap:
                        try:
                            val = float(odds_val)
                        except:
                            val = odds_val
                        
                        if handicap not in current_items:
                            current_items[handicap] = {}
                        
                        if 'over' in name.lower():
                            current_items[handicap][f'{handicap} Over'] = val
                        elif 'under' in name.lower():
                            current_items[handicap][f'{handicap} Under'] = val
                
                for hc, item in sorted(current_items.items()):
                    if item:
                        odds_data['Match goals'].append(item)
        
        return odds_data
    
    # ==================== 3. Attack Momentum ====================
    
    def get_attack_momentum(self, event_id: int) -> List[Dict]:
        """获取进攻动量数据"""
        data = self._get('graph', event_id)
        if not data:
            return []
        
        records = []
        graph_points = data.get('graphPoints', [])
        
        for point in graph_points:
            minute = point.get('minute', 0)
            value = point.get('value', 0)
            
            records.append({
                'match_id': event_id,
                'minute': minute,
                'momentum_value': value,
                'dominant_team': 'home' if value >= 0 else 'away',
                'abs_value': abs(value)
            })
        
        return records
    
    # ==================== 4. Lineups (修复版) ====================
    
    def get_lineups(self, event_id: int) -> Tuple[List[Dict], List[Dict]]:
        """
        获取阵容数据 (修复版)
        
        修复: 正确解析 statistics 中的所有字段
        """
        data = self._get('lineups', event_id)
        if not data:
            return [], []
        
        player_stats = []
        formation_data = []
        
        home_formation = data.get('home', {}).get('formation', '')
        away_formation = data.get('away', {}).get('formation', '')
        
        for team_key in ['home', 'away']:
            team_data = data.get(team_key, {})
            team_label = 'home_team' if team_key == 'home' else 'away_team'
            players = team_data.get('players', [])
            
            for player_info in players:
                player = player_info.get('player', {})
                stats = player_info.get('statistics', {})
                position = player_info.get('position', '')
                
                player_id = player.get('id', '')
                player_name = player.get('name', '')
                player_slug = player.get('slug', '')
                
                # ===== 修复: 正确解析球员统计 =====
                
                # Goals & Assists
                goals = stats.get('goals', 0)
                assists = stats.get('goalAssist', stats.get('assists', 0))
                
                # Tackles - 使用正确的API字段名
                total_tackle = stats.get('totalTackle', 0)
                won_tackle = stats.get('wonTackle', 0)
                tackles_str = f"{total_tackle} ({won_tackle})" if total_tackle else ''
                
                # Accurate passes - 修复字段名
                accurate_pass = stats.get('accuratePass', 0)
                total_pass = stats.get('totalPass', 0)
                pass_pct = round(accurate_pass / total_pass * 100) if total_pass > 0 else 0
                passes_str = f"{accurate_pass}/{total_pass} ({pass_pct}%)" if total_pass else ''
                
                # Duels - 修复字段名
                duel_won = stats.get('duelWon', 0)
                duel_lost = stats.get('duelLost', 0)
                total_duels = duel_won + duel_lost
                duels_str = f"{total_duels} ({duel_won})" if total_duels else ''
                
                # Ground duels
                challenge_won = stats.get('challengeWon', 0)
                challenge_lost = stats.get('challengeLost', 0)
                ground_total = challenge_won + challenge_lost
                ground_duels_str = f"{ground_total} ({challenge_won})" if ground_total else ''
                
                # Aerial duels
                aerial_won = stats.get('aerialWon', 0)
                aerial_lost = stats.get('aerialLost', 0)
                aerial_total = aerial_won + aerial_lost
                aerial_str = f"{aerial_total} ({aerial_won})" if aerial_total else ''
                
                # Minutes played
                minutes_played = stats.get('minutesPlayed', '')
                minutes_str = f"{minutes_played}'" if minutes_played else ''
                
                # Rating
                rating = stats.get('rating', '')
                if rating:
                    try:
                        rating = round(float(rating), 1)
                    except:
                        pass
                
                # 球员统计记录
                player_stats.append({
                    'match_id': event_id,
                    'team': team_label,
                    'player': player_name,
                    'player_id': player_id,
                    'player_link': f"https://www.sofascore.com/player/{player_slug}/{player_id}",
                    'Goals': goals,
                    'Assists': assists,
                    'Tackles_won': tackles_str,
                    'Accurate_passes': passes_str,
                    'Duels_won': duels_str,
                    'Ground_duels_won': ground_duels_str,
                    'Aerial_duels_won': aerial_str,
                    'Minutes_played': minutes_str,
                    'Position': position or player.get('position', ''),
                    'Sofascore_Rating': rating
                })
                
                # ===== 修复: 阵型位置数据 =====
                avg_x = player_info.get('averageX', 0)
                avg_y = player_info.get('averageY', 0)
                
                # 如果没有平均位置，从position估算
                if avg_x == 0 and avg_y == 0:
                    pos_str = position or player.get('position', '')
                    if pos_str:
                        avg_x, avg_y = self._estimate_position_coords(pos_str, team_key)
                
                formation_data.append({
                    'match_id': event_id,
                    'team': team_label,
                    'player': player_name,
                    'player_id': player_id,
                    'player_link': f"https://www.sofascore.com/player/{player_slug}/{player_id}",
                    'player_pos_x': avg_x,
                    'player_pos_y': avg_y,
                    'home_formation': home_formation,
                    'away_formation': away_formation
                })
        
        return player_stats, formation_data
    
    def _estimate_position_coords(self, position: str, team: str) -> Tuple[float, float]:
        """根据位置估算坐标 (0-100)"""
        position_map = {
            'G': (5, 50), 'GK': (5, 50),
            'D': (20, 50), 'DC': (20, 50), 'CB': (20, 50),
            'DL': (20, 20), 'LB': (20, 20), 'LWB': (25, 15),
            'DR': (20, 80), 'RB': (20, 80), 'RWB': (25, 85),
            'DM': (35, 50), 'CDM': (35, 50),
            'M': (50, 50), 'MC': (50, 50), 'CM': (50, 50),
            'ML': (50, 20), 'LM': (50, 20),
            'MR': (50, 80), 'RM': (50, 80),
            'AM': (65, 50), 'AMC': (65, 50), 'CAM': (65, 50),
            'AML': (65, 20), 'LW': (70, 15),
            'AMR': (65, 80), 'RW': (70, 85),
            'F': (85, 50), 'FW': (85, 50), 'CF': (85, 50),
            'ST': (90, 50), 'SS': (80, 50),
        }
        
        pos_upper = position.upper()
        coords = position_map.get(pos_upper, (50, 50))
        
        if team == 'away':
            return (100 - coords[0], 100 - coords[1])
        
        return coords
    
    # ==================== 5. Shotmap (新增) ====================
    
    def get_shotmap(self, event_id: int) -> List[Dict]:
        """
        获取射门图数据 (新增)
        """
        data = self._get('shotmap', event_id)
        if not data:
            return []
        
        records = []
        shotmap = data.get('shotmap', [])
        
        for shot in shotmap:
            player = shot.get('player', {})
            player_id = player.get('id', '')
            player_name = player.get('name', '')
            
            is_home = shot.get('isHome', True)
            home_or_away = 'home' if is_home else 'away'
            
            shot_type = shot.get('shotType', '')
            
            outcome_map = {
                'goal': 'Goal',
                'save': 'Saved',
                'saved': 'Saved',
                'miss': 'Off Target',
                'missed': 'Off Target',
                'block': 'Blocked',
                'blocked': 'Blocked',
                'post': 'Hit Post',
            }
            outcome = outcome_map.get(shot_type.lower(), shot_type) if shot_type else ''
            
            situation = shot.get('situation', '')
            situation_map = {
                'regular': 'Open Play',
                'assisted': 'Assisted',
                'set-piece': 'Set Piece',
                'fast-break': 'Fast Break',
                'corner': 'Corner',
                'free-kick': 'Free Kick',
                'penalty': 'Penalty',
            }
            situation_str = situation_map.get(situation.lower(), situation) if situation else ''
            
            body_part = shot.get('bodyPart', '')
            body_map = {
                'left-foot': 'Left Foot',
                'right-foot': 'Right Foot',
                'head': 'Header',
            }
            shot_body = body_map.get(body_part.lower(), body_part) if body_part else ''
            
            goal_mouth_location = shot.get('goalMouthLocation', '')
            goal_zone_map = {
                'high-centre': 'High Centre', 'high-left': 'High Left', 'high-right': 'High Right',
                'low-centre': 'Low Centre', 'low-left': 'Low Left', 'low-right': 'Low Right',
                'close-high': 'Close High', 'close-left': 'Close Left', 'close-right': 'Close Right',
            }
            goal_zone = goal_zone_map.get(goal_mouth_location.lower(), goal_mouth_location) if goal_mouth_location else ''
            
            xg = shot.get('xg', '')
            xgot = shot.get('xgot', '')
            
            if xg:
                try:
                    xg = round(float(xg), 3)
                except:
                    pass
            
            if xgot:
                try:
                    xgot = round(float(xgot), 3)
                except:
                    pass
            
            player_coords = shot.get('playerCoordinates', {})
            
            records.append({
                'match_id': event_id,
                'home_or_away': home_or_away,
                'player': player_name,
                'player_id': player_id,
                'xG': xg,
                'xGOT': xgot,
                'Outcome': outcome,
                'Situation': situation_str,
                'Shot_type': shot_body,
                'Goal_zone': goal_zone,
                'minute': shot.get('time', ''),
                'added_time': shot.get('addedTime', ''),
                'x_coord': player_coords.get('x', ''),
                'y_coord': player_coords.get('y', ''),
            })
        
        return records
    
    # ==================== 6. Player Match Stats (新增) ====================
    
    def get_all_player_match_stats(self, event_id: int, players: List[Dict]) -> List[Dict]:
        """
        获取所有球员的比赛详细统计
        
        从 lineups 数据中提取每个球员的完整统计
        """
        data = self._get('lineups', event_id)
        if not data:
            return []
        
        all_stats = []
        
        for team_key in ['home', 'away']:
            team_data = data.get(team_key, {})
            players_data = team_data.get('players', [])
            
            for player_info in players_data:
                player = player_info.get('player', {})
                stats = player_info.get('statistics', {})
                
                player_id = player.get('id', '')
                player_name = player.get('name', '')
                player_slug = player.get('slug', '')
                
                # 构建详细统计
                player_stats = {
                    'match_id': event_id,
                    'player_id': player_id,
                    'player_name': player_name,
                    'player_link': f"https://www.sofascore.com/player/{player_slug}/{player_id}",
                    'team': 'home' if team_key == 'home' else 'away',
                    
                    # Rating 及分项
                    'Rating': stats.get('rating', ''),
                    
                    # Goals & Scoring
                    'Goals': stats.get('goals', 0),
                    'Expected_goals_xG': stats.get('expectedGoals', ''),
                    
                    # Shots
                    'Total_shots': stats.get('totalShotsOnTarget', 0) + stats.get('shotOffTarget', 0) + stats.get('blockedScoringAttempt', 0),
                    'Shots_on_target': stats.get('onTargetScoringAttempt', 0),
                    'Shots_off_target': stats.get('shotOffTarget', 0),
                    'Blocked_shots': stats.get('blockedScoringAttempt', 0),
                    
                    # Key passes & Crosses
                    'Key_passes': stats.get('keyPass', 0),
                    'Crosses_accurate': stats.get('accurateCross', 0),
                    'Crosses_total': stats.get('totalCross', 0),
                    
                    # Accurate passes
                    'Accurate_passes': stats.get('accuratePass', 0),
                    'Total_passes': stats.get('totalPass', 0),
                    'Pass_accuracy_pct': round(stats.get('accuratePass', 0) / max(stats.get('totalPass', 1), 1) * 100, 1),
                    'Passes_into_final_third': stats.get('accurateFinalThirdPasses', 0),
                    'Passes_in_opposition_half': stats.get('accurateOppositionHalfPasses', 0),
                    'Long_balls_accurate': stats.get('accurateLongBalls', 0),
                    'Long_balls_total': stats.get('totalLongBalls', 0),
                    
                    # Dribbles
                    'Dribbles_successful': stats.get('successfulDribbles', 0),
                    'Dribbles_attempted': stats.get('totalDribbles', 0),
                    
                    # Duels
                    'Duels_won': stats.get('duelWon', 0),
                    'Duels_lost': stats.get('duelLost', 0),
                    'Ground_duels_won': stats.get('challengeWon', 0),
                    'Ground_duels_lost': stats.get('challengeLost', 0),
                    'Aerial_duels_won': stats.get('aerialWon', 0),
                    'Aerial_duels_lost': stats.get('aerialLost', 0),
                    
                    # Defending
                    'Tackles_total': stats.get('totalTackle', 0),
                    'Tackles_won': stats.get('wonTackle', 0),
                    'Interceptions': stats.get('interceptionWon', 0),
                    'Clearances': stats.get('totalClearance', 0),
                    'Blocks': stats.get('outfielderBlock', 0),
                    
                    # Fouls & Cards
                    'Fouls_committed': stats.get('fouls', 0),
                    'Was_fouled': stats.get('wasFouled', 0),
                    
                    # Other
                    'Assists': stats.get('goalAssist', 0),
                    'Big_chances_created': stats.get('bigChanceCreated', 0),
                    'Big_chances_missed': stats.get('bigChanceMissed', 0),
                    'Touches': stats.get('touches', 0),
                    'Possession_lost': stats.get('possessionLost', 0),
                    'Minutes_played': stats.get('minutesPlayed', ''),
                    
                    # Goalkeeper
                    'Saves': stats.get('saves', 0),
                    'Punches': stats.get('punches', 0),
                    'Runs_out': stats.get('runsOut', 0),
                    'Goals_conceded': stats.get('goalsConceded', 0),
                }
                
                # 只添加有评分的球员
                if stats.get('rating'):
                    all_stats.append(player_stats)
        
        return all_stats
    
    # ==================== 7. Statistics ====================
    
    def get_statistics(self, event_id: int, home_team_id: str = '', 
                       away_team_id: str = '') -> Tuple[Dict, Dict]:
        """获取统计数据"""
        data = self._get('statistics', event_id)
        if not data:
            return {}, {}
        
        home_stats = {'match_id': event_id, 'team_type': 'home', 'team_id': home_team_id}
        away_stats = {'match_id': event_id, 'team_type': 'away', 'team_id': away_team_id}
        
        statistics = data.get('statistics', [])
        
        for stat_group in statistics:
            period = stat_group.get('period', 'ALL')
            groups = stat_group.get('groups', [])
            
            for group in groups:
                group_name = group.get('groupName', '')
                items = group.get('statisticsItems', [])
                
                for item in items:
                    stat_name = item.get('name', '').replace(' ', '_').replace('(', '').replace(')', '')
                    home_val = item.get('home', '')
                    away_val = item.get('away', '')
                    
                    if item.get('homeTotal'):
                        home_val = f"{item.get('home', 0)}/{item.get('homeTotal')}"
                    if item.get('awayTotal'):
                        away_val = f"{item.get('away', 0)}/{item.get('awayTotal')}"
                    
                    prefix = group_name.replace(' ', '_') + '_' if group_name else ''
                    full_name = f"{prefix}{stat_name}"
                    
                    home_stats[full_name] = home_val
                    away_stats[full_name] = away_val
        
        return home_stats, away_stats
    
    # ==================== 主流程 ====================
    
    def scrape_match(self, match_id: int, match_url: str = '') -> Dict:
        """爬取单场比赛 (完整版)"""
        logger.info(f"{'='*50}")
        logger.info(f"爬取比赛: {match_id}")
        
        result = {'match_id': match_id, 'success': True, 'errors': [], 'files': []}
        
        try:
            # 1. 基本信息
            basic_info = self.get_basic_info(match_id)
            if basic_info:
                df = pd.DataFrame([basic_info])
                path = os.path.join(self.output_dir, f'{match_id}_basic_info.csv')
                df.to_csv(path, index=False, encoding='utf-8-sig')
                result['files'].append(path)
                logger.info(f"  ✓ 基本信息: {path}")
            
            home_team_id = basic_info.get('home_team_id', '')
            away_team_id = basic_info.get('away_team_id', '')
            
            # 2. 赔率
            odds = self.get_odds(match_id)
            if odds:
                path = os.path.join(self.output_dir, f'{match_id}_odds.json')
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(odds, f, ensure_ascii=False, indent=2)
                result['files'].append(path)
                logger.info(f"  ✓ 赔率数据: {path}")
            
            # 3. 动量
            momentum = self.get_attack_momentum(match_id)
            if momentum:
                df = pd.DataFrame(momentum)
                path = os.path.join(self.output_dir, f'{match_id}_attack_momentum.csv')
                df.to_csv(path, index=False, encoding='utf-8-sig')
                result['files'].append(path)
                logger.info(f"  ✓ 动量数据: {path}")
            
            # 4. 阵容 (修复版)
            player_stats, formation = self.get_lineups(match_id)
            
            if player_stats:
                df = pd.DataFrame(player_stats)
                path = os.path.join(self.output_dir, f'{match_id}_lineups.csv')
                df.to_csv(path, index=False, encoding='utf-8-sig')
                result['files'].append(path)
                logger.info(f"  ✓ 球员统计: {path} ({len(player_stats)}人)")
            
            if formation:
                df = pd.DataFrame(formation)
                path = os.path.join(self.output_dir, f'{match_id}_formation.csv')
                df.to_csv(path, index=False, encoding='utf-8-sig')
                result['files'].append(path)
                logger.info(f"  ✓ 阵型数据: {path}")
            
            # 5. 射门图 (新增)
            shotmap = self.get_shotmap(match_id)
            if shotmap:
                df = pd.DataFrame(shotmap)
                path = os.path.join(self.output_dir, f'{match_id}_shotmap.csv')
                df.to_csv(path, index=False, encoding='utf-8-sig')
                result['files'].append(path)
                logger.info(f"  ✓ 射门图: {path} ({len(shotmap)}次射门)")
            
            # 6. 球员详细统计 (新增) - 所有球员合并到一个文件
            all_player_stats = self.get_all_player_match_stats(match_id, player_stats)
            if all_player_stats:
                df = pd.DataFrame(all_player_stats)
                path = os.path.join(self.output_dir, f'{match_id}_all_players_stats.csv')
                df.to_csv(path, index=False, encoding='utf-8-sig')
                result['files'].append(path)
                logger.info(f"  ✓ 球员详细统计: {path} ({len(all_player_stats)}人)")
            
            # 7. 统计
            home_stats, away_stats = self.get_statistics(match_id, home_team_id, away_team_id)
            
            if home_stats and len(home_stats) > 3:
                df = pd.DataFrame([home_stats])
                path = os.path.join(self.output_dir, f'{match_id}_home_{home_team_id}_stats.csv')
                df.to_csv(path, index=False, encoding='utf-8-sig')
                result['files'].append(path)
                logger.info(f"  ✓ 主队统计: {path}")
            
            if away_stats and len(away_stats) > 3:
                df = pd.DataFrame([away_stats])
                path = os.path.join(self.output_dir, f'{match_id}_away_{away_team_id}_stats.csv')
                df.to_csv(path, index=False, encoding='utf-8-sig')
                result['files'].append(path)
                logger.info(f"  ✓ 客队统计: {path}")
            
        except Exception as e:
            result['success'] = False
            result['errors'].append(str(e))
            logger.error(f"爬取失败: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def scrape_all_matches(self, csv_path: str, delay: float = 2.0, 
                          limit: int = None) -> List[Dict]:
        """批量爬取"""
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        if 'status' in df.columns:
            df = df[df['status'] != 'canceled']
        
        if limit:
            df = df.head(limit)
        
        total = len(df)
        logger.info(f"准备爬取 {total} 场比赛")
        
        results = []
        
        for idx, row in df.iterrows():
            match_id = row['match_id']
            match_url = row.get('match_url', '')
            
            logger.info(f"\n[{idx+1}/{total}] 比赛 {match_id}")
            
            result = self.scrape_match(match_id, match_url)
            results.append(result)
            
            if idx < total - 1:
                time.sleep(delay + random.uniform(0, 0.5))
        
        # 保存汇总
        summary = pd.DataFrame([{
            'match_id': r['match_id'],
            'success': r['success'],
            'files_count': len(r.get('files', [])),
            'errors': '; '.join(r.get('errors', []))
        } for r in results])
        
        summary_path = os.path.join(self.output_dir, 'scrape_summary.csv')
        summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        success = sum(1 for r in results if r['success'])
        logger.info(f"\n{'='*50}")
        logger.info(f"完成! 成功: {success}/{total}")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SofaScore API Data Extractor v2.1 (Fixed)')
    parser.add_argument('-i', '--input', default='dec1_4_matches_unique.csv', help='输入CSV')
    parser.add_argument('-o', '--output', default='dec14', help='输出目录')
    parser.add_argument('-d', '--delay', type=float, default=2.0, help='延迟秒数')
    parser.add_argument('-l', '--limit', type=int, default=None, help='限制数量')
    parser.add_argument('-m', '--match-id', type=int, default=None, help='单场比赛')
    
    args = parser.parse_args()
    
    api = SofaScoreAPI(output_dir=args.output)
    
    if args.match_id:
        result = api.scrape_match(args.match_id)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        api.scrape_all_matches(args.input, args.delay, args.limit)


if __name__ == '__main__':
    main()
