#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
足球比赛数据爬虫工具 - Final Version
新增字段: date_str, full_start_time
新增功能: 输出五大联赛球队参赛比赛的单独CSV
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
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_top5_league_teams(csv_path: str = 'a0_sofascore_and_win007_teams.csv') -> Set[str]:
    """
    从CSV文件加载五大联赛球队名称列表

    Args:
        csv_path: CSV文件路径

    Returns:
        五大联赛球队中文名集合
    """
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
        logger.warning(f'球队CSV文件不存在: {csv_path}，将不输出五大联赛筛选结果')
    except Exception as e:
        logger.error(f'加载球队CSV失败: {e}')
    return teams


class FootballScraperFinal:
    """足球数据爬虫类 - 最终版本"""

    def __init__(self, headless=True, delay_range=(0, 1), timeout=15, teams_csv='a0_sofascore_and_win007_teams.csv'):
        """
        初始化爬虫

        Args:
            headless: 是否使用无头模式
            delay_range: 请求延迟范围（秒）
            timeout: 页面加载超时时间（秒）
            teams_csv: 五大联赛球队CSV文件路径
        """
        self.delay_range = delay_range
        self.timeout = timeout
        self.driver = self._init_driver(headless)
        self.failed_dates = []
        # 加载五大联赛球队名单
        self.top5_teams = load_top5_league_teams(teams_csv)
        
    def _init_driver(self, headless):
        """初始化Chrome驱动，配置反爬策略"""
        chrome_options = Options()
        
        if headless:
            chrome_options.add_argument('--headless=new')
        
        # 反爬策略配置
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # User-Agent池
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
        chrome_options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        # 其他优化选项
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--log-level=3')
        
        # 禁用图片加载
        prefs = {'profile.default_content_setting_values': {'images': 2}}
        chrome_options.add_experimental_option('prefs', prefs)
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            logger.error(f"Chrome驱动初始化失败: {e}")
            logger.info("尝试使用webdriver-manager自动下载...")
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # 修改webdriver属性
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['zh-CN', 'zh', 'en']
                });
            '''
        })
        
        return driver
    
    def _random_delay(self):
        """随机延迟"""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)
    
    def _format_score(self, score_text: str) -> str:
        """
        格式化比分：从 'm-n' 转换为 '[m:n]'
        
        Args:
            score_text: 原始比分文本，如 '3-1'
            
        Returns:
            格式化后的比分，如 '[3:1]'
        """
        if not score_text or score_text.strip() == '':
            return ''
        
        score_text = score_text.strip()
        
        # 匹配 m-n 格式（包括全角减号）
        match = re.search(r'(\d+)[-−–](\d+)', score_text)
        if match:
            return f'[{match.group(1)}:{match.group(2)}]'
        
        return score_text
    
    def _extract_match_id(self, html_text: str) -> Optional[str]:
        """
        从HTML文本中提取match_id
        
        Args:
            html_text: HTML文本内容
            
        Returns:
            match_id 或 None
        """
        patterns = [
            r'showgoallist\((\d+)\)',
            r'match[_-]?id["\']?\s*[:=]\s*["\']?(\d+)',
            r'id\s*=\s*["\']?(\d+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _parse_event_time(self, date_str: str, event_time_text: str) -> str:
        """
        解析赛事时间，生成完整的full_start_time
        
        重要逻辑：
        - 如果赛事时间显示的日期与date参数匹配，使用date的完整日期
        - 如果赛事时间显示的日期与date参数不匹配，以赛事时间的日期为准
        
        Args:
            date_str: 日期字符串，格式 YYYYMMDD，如 '20230101'
            event_time_text: 赛事时间文本，如 '1日01:00' 或 '2日01:00'
            
        Returns:
            完整时间字符串，格式 'YYYY-MM-DD HH:MM'
        """
        if not event_time_text or event_time_text.strip() == '':
            return ''
        
        try:
            # 解析基准日期
            base_date = datetime.strptime(date_str, '%Y%m%d')
            base_day = base_date.day
            
            # 清理赛事时间文本
            event_time_text = event_time_text.strip()
            
            # 匹配格式: "1日01:00", "2日01:00" 等
            match = re.search(r'(\d+)日\s*(\d{1,2}):(\d{2})', event_time_text)
            if match:
                event_day = int(match.group(1))  # 赛事时间中的日期
                hour = int(match.group(2))
                minute = int(match.group(3))
                
                # 计算日期差异
                day_diff = event_day - base_day
                
                # 生成结果日期
                result_date = base_date + timedelta(days=day_diff)
                result_time = result_date.replace(hour=hour, minute=minute, second=0)
                
                return result_time.strftime('%Y-%m-%d %H:%M')
            
            # 格式2: 只有时间 "01:00", "23:30" 等（当天时间）
            match2 = re.search(r'(\d{1,2}):(\d{2})', event_time_text)
            if match2:
                hour = int(match2.group(1))
                minute = int(match2.group(2))
                
                result_time = base_date.replace(hour=hour, minute=minute, second=0)
                return result_time.strftime('%Y-%m-%d %H:%M')
            
            # 如果无法解析，返回原文本
            return event_time_text
            
        except Exception as e:
            logger.debug(f'解析赛事时间失败 {date_str} {event_time_text}: {e}')
            return event_time_text
    
    def parse_page_with_beautifulsoup(self, html: str, date_str: str) -> List[Dict]:
        """
        使用BeautifulSoup解析HTML
        
        Args:
            html: HTML源代码
            date_str: 日期字符串，用于生成full_start_time
            
        Returns:
            解析后的数据列表
        """
        soup = BeautifulSoup(html, 'html.parser')
        matches = []
        
        # 查找所有可能包含比赛数据的表格行
        possible_selectors = [
            'tr[id^="tr"]',
            'tr.tr_match',
            'table#table tr',
            'table tr[onclick]',
            'tr[bgcolor]',
        ]
        
        rows = []
        for selector in possible_selectors:
            found_rows = soup.select(selector)
            if found_rows:
                rows = found_rows
                logger.debug(f"使用选择器 '{selector}' 找到 {len(rows)} 行")
                break
        
        if not rows:
            rows = soup.find_all('tr')
            logger.debug(f"使用通用选择器找到 {len(rows)} 行")
        
        for row in rows:
            try:
                row_html = str(row)
                
                # 提取match_id
                match_id = self._extract_match_id(row_html)
                
                if not match_id:
                    continue
                
                # 获取所有单元格
                cells = row.find_all(['td', 'th'])
                
                if len(cells) < 7:
                    continue
                
                # 提取文本并清理
                cell_texts = [cell.get_text(strip=True) for cell in cells]
                
                # 提取数据
                match_data = self._extract_match_data(cell_texts, match_id, date_str)
                
                if match_data:
                    matches.append(match_data)
                    
            except Exception as e:
                logger.debug(f'解析行数据失败: {e}')
                continue
        
        return matches
    
    def _extract_match_data(self, cells: List[str], match_id: str, date_str: str) -> Optional[Dict]:
        """
        从单元格数据中提取比赛信息
        
        Args:
            cells: 单元格文本列表
            match_id: 比赛ID
            date_str: 日期字符串，用于生成full_start_time
            
        Returns:
            比赛数据字典或None
        """
        try:
            # 尝试找到包含比分的单元格
            score_index = -1
            for i, cell in enumerate(cells):
                if re.search(r'\d+[-−–]\d+', cell):
                    score_index = i
                    break
            
            if score_index == -1:
                return None
            
            # 基于比分位置推断其他字段位置
            league = cells[max(0, score_index - 4)] if score_index >= 4 else ''
            event_time = cells[max(0, score_index - 3)] if score_index >= 3 else ''
            status = cells[max(0, score_index - 2)] if score_index >= 2 else ''
            home_team = cells[max(0, score_index - 1)] if score_index >= 1 else ''
            score = self._format_score(cells[score_index])
            away_team = cells[score_index + 1] if score_index + 1 < len(cells) else ''
            half_score = self._format_score(cells[score_index + 2]) if score_index + 2 < len(cells) else ''
            asian_handicap = cells[score_index + 3] if score_index + 3 < len(cells) else ''
            total_goals = cells[score_index + 4] if score_index + 4 < len(cells) else ''
            data = cells[score_index + 5] if score_index + 5 < len(cells) else ''
            
            # 生成full_start_time
            full_start_time = self._parse_event_time(date_str, event_time)
            
            match_data = {
                'date_str': date_str,
                'full_start_time': full_start_time,
                'match_id': match_id,
                '联赛': league,
                '赛事时间': event_time,
                '状态': status,
                '主场球队': home_team,
                '比分': score,
                '客场球队': away_team,
                '半场': half_score,
                '亚让': asian_handicap,
                '进球数': total_goals,
                '数据': data,
            }
            
            return match_data
            
        except Exception as e:
            logger.debug(f'提取比赛数据失败: {e}')
            return None
    
    def parse_page(self, date_str: str, retry_count: int = 2) -> List[Dict]:
        """
        解析单个日期的页面数据
        
        Args:
            date_str: 日期字符串，格式：YYYYMMDD
            retry_count: 重试次数
            
        Returns:
            解析后的数据列表
        """
        url = f'https://bf.titan007.com/football/Over_{date_str}.htm'
        
        for attempt in range(retry_count + 1):
            try:
                if attempt > 0:
                    logger.info(f'  重试 {attempt}/{retry_count}...')
                    time.sleep(random.uniform(3, 6))
                
                logger.info(f'正在访问: {url}')
                self.driver.get(url)
                
                # 等待页面加载
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # 等待动态内容
                time.sleep(random.uniform(2, 4))
                
                # 检查是否被拦截
                if "403" in self.driver.title or "访问被拒绝" in self.driver.page_source:
                    logger.warning(f'  访问被拒绝，可能被反爬虫拦截')
                    if attempt < retry_count:
                        continue
                    return []
                
                # 获取页面源代码
                page_source = self.driver.page_source
                
                # 使用BeautifulSoup解析
                matches = self.parse_page_with_beautifulsoup(page_source, date_str)
                
                logger.info(f'  ✓ 成功解析 {len(matches)} 条数据')
                return matches
                
            except TimeoutException:
                logger.warning(f'  ⚠ 页面加载超时: {url}')
                if attempt < retry_count:
                    continue
                self.failed_dates.append(date_str)
                return []
            except Exception as e:
                logger.error(f'  ✗ 访问失败 {url}: {e}')
                if attempt < retry_count:
                    continue
                self.failed_dates.append(date_str)
                return []
        
        return []
    
    def scrape_date_range(self, start_date: str, end_date: str, output_file: str, 
                          checkpoint_file: str = None):
        """
        爬取日期范围内的所有数据
        
        Args:
            start_date: 开始日期字符串 YYYYMMDD
            end_date: 结束日期字符串 YYYYMMDD
            output_file: 输出CSV文件路径
            checkpoint_file: 检查点文件路径
        """
        # 转换日期
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        # 读取检查点
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
        
        logger.info(f'\n{"="*60}')
        logger.info(f'开始爬取数据: {start_date} 到 {end_date}')
        logger.info(f'共需处理 {total_days} 天的数据')
        logger.info(f'{"="*60}\n')
        
        start_time = time.time()
        
        while current <= end:
            date_str = current.strftime('%Y%m%d')
            processed += 1
            
            # 跳过已处理的日期
            if date_str in processed_dates:
                skipped += 1
                current += timedelta(days=1)
                continue
            
            logger.info(f'[{processed}/{total_days}] 处理日期: {date_str}')
            
            matches = self.parse_page(date_str)
            all_matches.extend(matches)
            
            # 保存检查点
            if checkpoint_file:
                with open(checkpoint_file, 'a') as f:
                    f.write(f'{date_str}\n')
            
            # 随机延迟
            if current < end:
                self._random_delay()
            
            # 定期保存数据（每100天）
            if processed % 100 == 0:
                self._save_intermediate_results(all_matches, output_file)
                logger.info(f'  → 已保存中间结果到: {output_file}')
            
            current += timedelta(days=1)
        
        # 最终保存（包含全部数据和五大联赛筛选数据）
        self.save_all_csvs(all_matches, output_file)

        # 统计五大联赛数据
        top5_count = len(self.filter_top5_matches(all_matches)) if self.top5_teams else 0

        # 统计信息
        elapsed_time = time.time() - start_time
        logger.info(f'\n{"="*60}')
        logger.info(f'✓ 爬取完成！')
        logger.info(f'  总天数: {total_days}')
        logger.info(f'  跳过: {skipped}')
        logger.info(f'  处理: {processed - skipped}')
        logger.info(f'  失败: {len(self.failed_dates)}')
        logger.info(f'  全部比赛: {len(all_matches)} 条')
        logger.info(f'  五大联赛球队比赛: {top5_count} 条')
        logger.info(f'  耗时: {elapsed_time/60:.1f} 分钟')
        logger.info(f'  数据文件: {output_file}')
        
        if self.failed_dates:
            logger.warning(f'\n失败的日期 ({len(self.failed_dates)}):')
            for date in self.failed_dates[:10]:
                logger.warning(f'  - {date}')
            if len(self.failed_dates) > 10:
                logger.warning(f'  ... 还有 {len(self.failed_dates) - 10} 个')
        
        logger.info(f'{"="*60}\n')
    
    def _save_intermediate_results(self, data: List[Dict], output_file: str):
        """保存中间结果"""
        if data:
            self.save_to_csv(data, output_file)
    
    def filter_top5_matches(self, data: List[Dict]) -> List[Dict]:
        """
        筛选五大联赛球队参赛的比赛

        Args:
            data: 全部比赛数据列表

        Returns:
            五大联赛球队参赛的比赛列表
        """
        if not self.top5_teams:
            return []

        filtered = []
        for match in data:
            home_team = match.get('主场球队', '').strip()
            away_team = match.get('客场球队', '').strip()
            # 主队或客队是五大联赛球队则保留
            if home_team in self.top5_teams or away_team in self.top5_teams:
                filtered.append(match)

        return filtered

    def save_to_csv(self, data: List[Dict], output_file: str):
        """
        保存数据到CSV文件

        Args:
            data: 数据列表
            output_file: 输出文件路径
        """
        if not data:
            logger.warning('没有数据需要保存')
            return

        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入CSV - 字段顺序：date_str, full_start_time, match_id, ...
        fieldnames = ['date_str', 'full_start_time', 'match_id', '联赛', '赛事时间', '状态',
                      '主场球队', '比分', '客场球队', '半场', '亚让', '进球数', '数据']

        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        logger.info(f'数据已保存: {len(data)} 条记录')

    def save_all_csvs(self, data: List[Dict], output_file: str):
        """
        保存全部数据和五大联赛筛选数据到两个CSV

        Args:
            data: 全部比赛数据列表
            output_file: 主输出文件路径
        """
        # 保存全部数据
        self.save_to_csv(data, output_file)

        # 筛选并保存五大联赛球队比赛
        if self.top5_teams:
            top5_matches = self.filter_top5_matches(data)
            if top5_matches:
                # 生成五大联赛文件名：在原文件名基础上加 _top5
                output_path = Path(output_file)
                top5_file = output_path.parent / f"{output_path.stem}_top5{output_path.suffix}"
                self.save_to_csv(top5_matches, str(top5_file))
                logger.info(f'五大联赛球队比赛已保存: {len(top5_matches)} 条 -> {top5_file}')
    
    def close(self):
        """关闭浏览器驱动"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='足球比赛数据爬虫工具 - Final Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 爬取2023年1月1日到2025年11月11日的数据
  python football_scraper_final.py --start 20230101 --end 20251111 --output football_data.csv
  
  # 使用检查点支持断点续传
  python football_scraper_final.py --start 20230101 --end 20251111 --output data.csv --checkpoint checkpoint.txt
        """
    )
    
    parser.add_argument('--start', type=str, required=True,
                        help='开始日期 (格式: YYYYMMDD)')
    parser.add_argument('--end', type=str, required=True,
                        help='结束日期 (格式: YYYYMMDD)')
    parser.add_argument('--output', '-o', type=str, default='football_data.csv',
                        help='输出CSV文件路径 (默认: football_data.csv)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点文件路径（支持断点续传）')
    parser.add_argument('--no-headless', action='store_true',
                        help='显示浏览器窗口（用于调试）')
    parser.add_argument('--min-delay', type=float, default=1.0,
                        help='最小延迟秒数 (默认: 2)')
    parser.add_argument('--max-delay', type=float, default=5.0,
                        help='最大延迟秒数 (默认: 5)')
    parser.add_argument('--timeout', type=int, default=15,
                        help='页面加载超时时间（秒） (默认: 15)')
    parser.add_argument('--debug', action='store_true',
                        help='开启调试模式')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证日期格式
    try:
        datetime.strptime(args.start, '%Y%m%d')
        datetime.strptime(args.end, '%Y%m%d')
    except ValueError:
        logger.error('日期格式不正确，请使用 YYYYMMDD 格式')
        return
    
    # 创建爬虫实例
    scraper = FootballScraperFinal(
        headless=not args.no_headless,
        delay_range=(args.min_delay, args.max_delay),
        timeout=args.timeout
    )
    
    try:
        # 执行爬取
        scraper.scrape_date_range(
            args.start, 
            args.end, 
            args.output,
            args.checkpoint
        )
    except KeyboardInterrupt:
        logger.info('\n\n用户中断操作')
    except Exception as e:
        logger.error(f'\n发生错误: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        scraper.close()


if __name__ == '__main__':
    main()
