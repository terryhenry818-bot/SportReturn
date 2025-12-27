#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足球比赛数据爬虫工具 - V2 版本 (Selenium)
支持历史比赛和未来赛程两种模式
- 历史比赛: https://bf.titan007.com/football/Over_{date_str}.htm
- 未来赛程: https://bf.titan007.com/football/Next_{date_str}.htm
"""

import re
import csv
import time
import random
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_top5_league_teams(csv_path: str = 'a0_sofascore_and_win007_teams.csv') -> Set[str]:
    teams = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                team_name = row.get('win007_team_name', '').strip()
                if team_name:
                    teams.add(team_name)
        logger.info(f'已加载 {len(teams)} 支五大联赛球队')
    except FileNotFoundError:
        logger.warning(f'球队CSV文件不存在: {csv_path}')
    except Exception as e:
        logger.error(f'加载球队CSV失败: {e}')
    return teams


def clean_team_name(team_name: str) -> str:
    if not team_name:
        return team_name
    cleaned = re.sub(r'\[.*?\]', '', team_name)
    return cleaned.strip()


class FootballScraperV2:
    URL_PATTERN_HISTORY = 'https://bf.titan007.com/football/Over_{date_str}.htm'
    URL_PATTERN_FUTURE = 'https://bf.titan007.com/football/Next_{date_str}.htm'

    def __init__(self, headless=True, delay_range=(0, 1), timeout=15, teams_csv='a0_sofascore_and_win007_teams.csv'):
        self.delay_range = delay_range
        self.timeout = timeout
        self.headless = headless
        self.failed_dates = []
        self.top5_teams = load_top5_league_teams(teams_csv)
        self.driver = None
        
    def _init_browser(self):
        if self.driver is None:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            ]
            chrome_options.add_argument(f'user-agent={random.choice(user_agents)}')
            
            prefs = {'profile.managed_default_content_settings.images': 2}
            chrome_options.add_experimental_option('prefs', prefs)
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            logger.info('Chrome浏览器初始化成功')
    
    def _random_delay(self):
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)
    
    def _is_future_date(self, date_str: str) -> bool:
        target_date = datetime.strptime(date_str, '%Y%m%d').date()
        today = datetime.now().date()
        return target_date >= today
    
    def _get_url_for_date(self, date_str: str) -> str:
        if self._is_future_date(date_str):
            return self.URL_PATTERN_FUTURE.format(date_str=date_str)
        else:
            return self.URL_PATTERN_HISTORY.format(date_str=date_str)
    
    def _format_score(self, score_text: str) -> str:
        if not score_text or score_text.strip() == '':
            return ''
        score_text = score_text.strip()
        match = re.search(r'(\d+)[-−–](\d+)', score_text)
        if match:
            return f'[{match.group(1)}:{match.group(2)}]'
        return score_text
    
    def _extract_match_id_from_row(self, row) -> Optional[str]:
        sid = row.get('sid')
        if sid:
            return sid
        row_html = str(row)
        patterns = [
            r"sId='(\d+)'",
            r'sId="(\d+)"',
            r'showgoallist\((\d+)\)',
            r"AsianOdds\((\d+)\)",
            r"EuropeOdds\((\d+)\)",
            r"analysis\((\d+)\)",
        ]
        for pattern in patterns:
            match = re.search(pattern, row_html)
            if match:
                return match.group(1)
        return None
    
    def _parse_event_time(self, date_str: str, event_time_text: str) -> str:
        if not event_time_text or event_time_text.strip() == '':
            return ''
        try:
            base_date = datetime.strptime(date_str, '%Y%m%d')
            event_time_text = event_time_text.strip()
            
            match = re.search(r'(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})', event_time_text)
            if match:
                month = int(match.group(1))
                day = int(match.group(2))
                hour = int(match.group(3))
                minute = int(match.group(4))
                year = base_date.year
                if month < base_date.month:
                    year += 1
                result_time = datetime(year, month, day, hour, minute)
                return result_time.strftime('%Y-%m-%d %H:%M')
            
            match = re.search(r'(\d+)日\s*(\d{1,2}):(\d{2})', event_time_text)
            if match:
                event_day = int(match.group(1))
                hour = int(match.group(2))
                minute = int(match.group(3))
                day_diff = event_day - base_date.day
                result_date = base_date + timedelta(days=day_diff)
                result_time = result_date.replace(hour=hour, minute=minute, second=0)
                return result_time.strftime('%Y-%m-%d %H:%M')
            
            match2 = re.search(r'^(\d{1,2}):(\d{2})$', event_time_text)
            if match2:
                hour = int(match2.group(1))
                minute = int(match2.group(2))
                result_time = base_date.replace(hour=hour, minute=minute, second=0)
                return result_time.strftime('%Y-%m-%d %H:%M')
            
            return event_time_text
        except Exception as e:
            logger.debug(f'解析赛事时间失败: {e}')
            return event_time_text
    
    def _convert_time_format(self, event_time: str) -> str:
        match = re.search(r'(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})', event_time)
        if match:
            day = int(match.group(2))
            hour = match.group(3)
            minute = match.group(4)
            return f"{day}日{hour}:{minute}"
        return event_time
    
    def parse_page_with_beautifulsoup(self, html: str, date_str: str, is_future: bool = False) -> List[Dict]:
        soup = BeautifulSoup(html, 'html.parser')
        matches = []
        
        table = soup.find('table', id='table_live') or soup.find('table', id='table')
        if not table:
            logger.warning('未找到数据表格')
            return matches
        
        rows = table.find_all('tr')
        logger.debug(f'找到 {len(rows)} 行')
        
        for row in rows:
            try:
                if row.find('td', bgcolor='#990000') or not row.get('id'):
                    continue
                
                match_id = self._extract_match_id_from_row(row)
                if not match_id:
                    continue
                
                cells = row.find_all('td')
                if len(cells) < 6:
                    continue
                
                if is_future:
                    match_data = self._extract_future_match_data_v2(cells, match_id, date_str)
                else:
                    match_data = self._extract_match_data(cells, match_id, date_str)
                
                if match_data:
                    matches.append(match_data)
                    
            except Exception as e:
                logger.debug(f'解析行数据失败: {e}')
                continue
        
        return matches
    
    def _extract_match_data(self, cells, match_id: str, date_str: str) -> Optional[Dict]:
        try:
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            
            score_index = -1
            for i, cell in enumerate(cell_texts):
                if re.search(r'\d+[-−–]\d+', cell):
                    score_index = i
                    break
            if score_index == -1:
                return None
            
            league = cell_texts[max(0, score_index - 4)] if score_index >= 4 else ''
            event_time = cell_texts[max(0, score_index - 3)] if score_index >= 3 else ''
            status = cell_texts[max(0, score_index - 2)] if score_index >= 2 else ''
            home_team = cell_texts[max(0, score_index - 1)] if score_index >= 1 else ''
            score = self._format_score(cell_texts[score_index])
            away_team = cell_texts[score_index + 1] if score_index + 1 < len(cell_texts) else ''
            half_score = self._format_score(cell_texts[score_index + 2]) if score_index + 2 < len(cell_texts) else ''
            asian_handicap = cell_texts[score_index + 3] if score_index + 3 < len(cell_texts) else ''
            total_goals = cell_texts[score_index + 4] if score_index + 4 < len(cell_texts) else ''
            data = cell_texts[score_index + 5] if score_index + 5 < len(cell_texts) else ''
            
            event_time_display = self._convert_time_format(event_time)
            full_start_time = self._parse_event_time(date_str, event_time)
            
            return {
                'date_str': date_str, 'full_start_time': full_start_time, 'match_id': match_id,
                '联赛': league, '赛事时间': event_time_display, '状态': status,
                '主场球队': home_team, '比分': score, '客场球队': away_team,
                '半场': half_score, '亚让': asian_handicap, '进球数': total_goals, '数据': data,
            }
        except Exception as e:
            logger.debug(f'提取比赛数据失败: {e}')
            return None
    
    def _extract_future_match_data_v2(self, cells, match_id: str, date_str: str) -> Optional[Dict]:
        try:
            if len(cells) < 6:
                return None
            
            league_cell = cells[0]
            league = league_cell.find('span')
            league = league.get_text(strip=True) if league else league_cell.get_text(strip=True)
            
            event_time = cells[1].get_text(strip=True)
            home_team = cells[3].get_text(strip=True)
            away_team = cells[5].get_text(strip=True)
            
            half_score = ''
            if len(cells) > 6:
                half_text = cells[6].get_text(strip=True)
                if half_text and not half_text.startswith('香港'):
                    half_score = half_text
            
            asian_handicap = cells[7].get_text(strip=True) if len(cells) > 7 else ''
            total_goals = cells[8].get_text(strip=True) if len(cells) > 8 else ''
            
            data = ''
            if len(cells) > 9:
                data_cell = cells[9]
                links = data_cell.find_all('a')
                data = ''.join([a.get_text(strip=True) for a in links])
            
            if not home_team or not away_team:
                return None
            
            event_time_display = self._convert_time_format(event_time)
            full_start_time = self._parse_event_time(date_str, event_time)
            
            return {
                'date_str': date_str, 
                'full_start_time': full_start_time, 
                'match_id': match_id,
                '联赛': league, 
                '赛事时间': event_time_display, 
                '状态': '未',
                '主场球队': home_team, 
                '比分': '', 
                '客场球队': away_team,
                '半场': '', 
                '亚让': asian_handicap, 
                '进球数': total_goals, 
                '数据': data,
            }
        except Exception as e:
            logger.debug(f'提取未来比赛数据失败: {e}')
            return None
    
    def parse_page(self, date_str: str, retry_count: int = 2) -> List[Dict]:
        self._init_browser()
        is_future = self._is_future_date(date_str)
        url = self._get_url_for_date(date_str)
        mode_str = "未来赛程" if is_future else "历史比赛"
        
        for attempt in range(retry_count + 1):
            try:
                if attempt > 0:
                    logger.info(f'  重试 {attempt}/{retry_count}...')
                    time.sleep(random.uniform(3, 6))
                
                logger.info(f'正在访问 [{mode_str}]: {url}')
                self.driver.get(url)
                time.sleep(random.uniform(10, 15))
                
                title = self.driver.title
                if "403" in title or "访问被拒绝" in title:
                    logger.warning(f'  访问被拒绝')
                    if attempt < retry_count:
                        continue
                    return []
                
                page_source = self.driver.page_source
                matches = self.parse_page_with_beautifulsoup(page_source, date_str, is_future)
                logger.info(f'  ✓ 成功解析 {len(matches)} 条数据')
                return matches
                
            except TimeoutException:
                logger.warning(f'  ⚠ 页面加载超时')
                if attempt < retry_count:
                    continue
                self.failed_dates.append(date_str)
                return []
            except WebDriverException as e:
                logger.warning(f'  ⚠ WebDriver错误: {e}')
                if attempt < retry_count:
                    self.close()
                    self.driver = None
                    continue
                self.failed_dates.append(date_str)
                return []
            except Exception as e:
                logger.warning(f'  ⚠ 访问失败: {e}')
                if attempt < retry_count:
                    continue
                self.failed_dates.append(date_str)
                return []
        return []
    
    def parse_local_file(self, file_path: str, date_str: str, is_future: bool = True) -> List[Dict]:
        with open(file_path, 'rb') as f:
            raw = f.read()
        for encoding in ['gb2312', 'gbk', 'gb18030', 'utf-8']:
            try:
                html = raw.decode(encoding)
                break
            except:
                continue
        else:
            html = raw.decode('utf-8', errors='ignore')
        return self.parse_page_with_beautifulsoup(html, date_str, is_future)
    
    def scrape_date_range(self, start_date: str, end_date: str, output_file: str, checkpoint_file: str = None):
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        processed_dates = set()
        if checkpoint_file and Path(checkpoint_file).exists():
            with open(checkpoint_file, 'r') as f:
                processed_dates = set(line.strip() for line in f)
            logger.info(f'从检查点恢复，已处理 {len(processed_dates)} 天')
        
        all_matches = []
        current = start
        total_days = (end - start).days + 1
        processed = 0
        skipped = 0
        
        today = datetime.now().date()
        start_is_future = start.date() >= today
        mode_desc = "未来赛程模式" if start_is_future else "历史比赛模式"
        
        logger.info(f'\n{"="*60}')
        logger.info(f'开始爬取数据: {start_date} 到 {end_date}')
        logger.info(f'运行模式: {mode_desc}')
        logger.info(f'共需处理 {total_days} 天的数据')
        logger.info(f'{"="*60}\n')
        
        start_time = time.time()
        
        while current <= end:
            date_str = current.strftime('%Y%m%d')
            processed += 1
            if date_str in processed_dates:
                skipped += 1
                current += timedelta(days=1)
                continue
            
            logger.info(f'[{processed}/{total_days}] 处理日期: {date_str}')
            matches = self.parse_page(date_str)
            all_matches.extend(matches)
            
            if checkpoint_file:
                with open(checkpoint_file, 'a') as f:
                    f.write(f'{date_str}\n')
            if current < end:
                self._random_delay()
            if processed % 100 == 0:
                self._save_intermediate_results(all_matches, output_file)
            current += timedelta(days=1)
        
        self.save_all_csvs(all_matches, output_file)
        top5_count = len(self.filter_top5_matches(all_matches)) if self.top5_teams else 0
        
        elapsed_time = time.time() - start_time
        logger.info(f'\n{"="*60}')
        logger.info(f'✓ 爬取完成！')
        logger.info(f'  总天数: {total_days}, 跳过: {skipped}, 处理: {processed - skipped}')
        logger.info(f'  失败: {len(self.failed_dates)}, 全部比赛: {len(all_matches)} 条')
        logger.info(f'  五大联赛球队比赛: {top5_count} 条')
        logger.info(f'  耗时: {elapsed_time/60:.1f} 分钟')
        logger.info(f'  数据文件: {output_file}')
        if self.failed_dates:
            logger.warning(f'\n失败的日期: {self.failed_dates[:10]}')
        logger.info(f'{"="*60}\n')
    
    def _save_intermediate_results(self, data: List[Dict], output_file: str):
        if data:
            self.save_to_csv(data, output_file)
    
    def filter_top5_matches(self, data: List[Dict]) -> List[Dict]:
        if not self.top5_teams:
            return []
        filtered = []
        for match in data:
            home_team = match.get('主场球队', '').strip()
            away_team = match.get('客场球队', '').strip()
            home_clean = clean_team_name(home_team)
            away_clean = clean_team_name(away_team)
            if home_clean in self.top5_teams or away_clean in self.top5_teams:
                filtered.append(match)
        return filtered

    def save_to_csv(self, data: List[Dict], output_file: str):
        if not data:
            logger.warning('没有数据需要保存')
            return
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['date_str', 'full_start_time', 'match_id', '联赛', '赛事时间', '状态',
                      '主场球队', '比分', '客场球队', '半场', '亚让', '进球数', '数据']
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f'数据已保存: {len(data)} 条记录 -> {output_file}')

    def save_all_csvs(self, data: List[Dict], output_file: str):
        self.save_to_csv(data, output_file)
        if self.top5_teams:
            top5_matches = self.filter_top5_matches(data)
            if top5_matches:
                output_path = Path(output_file)
                top5_file = output_path.parent / f"{output_path.stem}_top5{output_path.suffix}"
                self.save_to_csv(top5_matches, str(top5_file))
    
    def close(self):
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
        except:
            pass


def main():
    parser = argparse.ArgumentParser(
        description='足球比赛数据爬虫工具 - V2版本 Selenium（支持历史和未来赛程）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 爬取历史比赛数据（过去日期）
  python win007_match_scraper_v2.py --start 20231201 --end 20231231 --output history_data.csv
  
  # 爬取未来赛程数据（今天及未来日期）
  python win007_match_scraper_v2.py --start 20251227 --end 20251228 --output future_data.csv

URL模式说明:
  - 历史比赛 (过去日期): https://bf.titan007.com/football/Over_{date}.htm
  - 未来赛程 (今天及未来): https://bf.titan007.com/football/Next_{date}.htm
"""
    )
    
    parser.add_argument('--start', type=str, required=True, help='开始日期 (格式: YYYYMMDD)')
    parser.add_argument('--end', type=str, required=True, help='结束日期 (格式: YYYYMMDD)')
    parser.add_argument('--output', '-o', type=str, default='football_data.csv', help='输出CSV文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点文件路径')
    parser.add_argument('--no-headless', action='store_true', help='显示浏览器窗口')
    parser.add_argument('--min-delay', type=float, default=1.0, help='最小延迟秒数')
    parser.add_argument('--max-delay', type=float, default=5.0, help='最大延迟秒数')
    parser.add_argument('--timeout', type=int, default=15, help='页面加载超时时间（秒）')
    parser.add_argument('--teams-csv', type=str, default='a0_sofascore_and_win007_teams.csv', help='五大联赛球队CSV')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    parser.add_argument('--local-file', type=str, default=None, help='解析本地HTML文件（测试用）')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        datetime.strptime(args.start, '%Y%m%d')
        datetime.strptime(args.end, '%Y%m%d')
    except ValueError:
        logger.error('日期格式不正确，请使用 YYYYMMDD 格式')
        return
    
    scraper = FootballScraperV2(
        headless=not args.no_headless,
        delay_range=(args.min_delay, args.max_delay),
        timeout=args.timeout,
        teams_csv=args.teams_csv
    )
    
    try:
        if args.local_file:
            logger.info(f'解析本地文件: {args.local_file}')
            matches = scraper.parse_local_file(args.local_file, args.start, is_future=True)
            scraper.save_all_csvs(matches, args.output)
            logger.info(f'完成! 解析到 {len(matches)} 条比赛数据')
        else:
            scraper.scrape_date_range(args.start, args.end, args.output, args.checkpoint)
    except KeyboardInterrupt:
        logger.info('\n\n用户中断操作')
    except Exception as e:
        logger.error(f'\n发生错误: {e}')
        import traceback
        traceback.print_exc()
    finally:
        scraper.close()


if __name__ == '__main__':
    main()
