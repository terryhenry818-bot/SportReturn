#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
球探网全功能数据爬虫工具
支持：让球数据、大小球数据、对阵分析数据抓取
"""

import os
import sys
import time
import csv
import random
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Set
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import re


class TitanFullScraper:
    """球探网全功能数据爬虫类"""
    
    # 公司ID列表
    COMPANY_IDS = [1, 3, 8, 12, 14, 17, 22, 23, 24, 31, 35]
    
    def __init__(self, headless=True, delay_range=(1, 1)):
        """
        初始化爬虫
        
        Args:
            headless: 是否无头模式运行
            delay_range: 请求延迟范围(秒)
        """
        self.headless = headless
        self.delay_range = delay_range
        self.driver = None
        self.match_ids = []
        
    def load_match_ids_from_csv(self, csv_file):
        """
        从CSV文件加载match_id列表
        
        Args:
            csv_file: CSV文件路径
            
        Returns:
            match_id列表
        """
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            if 'match_id' not in df.columns:
                print(f"❌ CSV文件中未找到'match_id'列")
                return []
            
            # 提取唯一的match_id
            match_ids = df['match_id'].dropna().astype(int).unique().tolist()
            print(f"✓ 从 {csv_file} 加载了 {len(match_ids)} 个唯一的 match_id")
            return match_ids
        except Exception as e:
            print(f"❌ 加载CSV文件失败: {e}")
            return []
    
    def _init_driver(self):
        """初始化Selenium WebDriver，配置反爬策略"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        # 反爬虫配置
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # 随机User-Agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
        chrome_options.add_argument(f'user-agent={random.choice(user_agents)}')
        
        # 禁用自动化标识
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            # 修改webdriver属性
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                '''
            })
            print("✓ WebDriver初始化成功")
        except Exception as e:
            print(f"❌ 初始化WebDriver失败: {e}")
            print("请确保已安装Chrome浏览器和ChromeDriver")
            sys.exit(1)
    
    def _random_delay(self):
        """随机延迟，模拟人类行为"""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)
    
    def _format_score(self, score_text):
        """
        格式化比分为[m:n]格式
        
        Args:
            score_text: 原始比分文本
            
        Returns:
            格式化后的比分，如"[1:2]"
        """
        if not score_text or score_text == '-':
            return '-'
        
        # 处理各种可能的分隔符
        for sep in ['-', ':', '－', '：']:
            if sep in score_text:
                parts = score_text.split(sep)
                if len(parts) == 2:
                    try:
                        m, n = parts[0].strip(), parts[1].strip()
                        return f"[{m}:{n}]"
                    except:
                        pass
        
        return score_text
    
    def scrape_handicap(self, match_id, company_id):
        """
        抓取亚洲让球数据
        
        Args:
            match_id: 比赛ID
            company_id: 公司ID
            
        Returns:
            数据列表
        """
        url = f"https://vip.titan007.com/changeDetail/handicap.aspx?id={match_id}&companyid={company_id}&l=0"
        
        try:
            self.driver.get(url)
            self._random_delay()
            
            # 等待表格加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            
            # 获取页面HTML并解析
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # 查找数据表格
            table = soup.find('table', {'id': 'table1'}) or soup.find('table', {'class': 'tbl01'}) or soup.find('table')
            
            if not table:
                return []
            
            data = []
            rows = table.find_all('tr')[1:]  # 跳过表头
            
            for row in rows:
                cols = row.find_all(['td', 'th'])
                if len(cols) >= 7:
                    time_str = cols[0].get_text(strip=True)
                    score = self._format_score(cols[1].get_text(strip=True))
                    home_odds = cols[2].get_text(strip=True)
                    handicap = cols[3].get_text(strip=True)
                    away_odds = cols[4].get_text(strip=True)
                    change_time = cols[5].get_text(strip=True)
                    status = cols[6].get_text(strip=True) if len(cols) > 6 else ''
                    
                    data.append([
                        match_id, company_id, time_str, score, home_odds, 
                        handicap, away_odds, change_time, status
                    ])
            
            return data
            
        except TimeoutException:
            print(f"⚠️  Match {match_id}, Company {company_id}: 页面加载超时")
            return []
        except Exception as e:
            print(f"❌ Match {match_id}, Company {company_id}: {e}")
            return []
    
    def scrape_overunder(self, match_id, company_id):
        """
        抓取大小球数据
        
        Args:
            match_id: 比赛ID
            company_id: 公司ID
            
        Returns:
            数据列表
        """
        url = f"https://vip.titan007.com/changeDetail/overunder.aspx?id={match_id}&companyid={company_id}&l=0"
        
        try:
            self.driver.get(url)
            self._random_delay()
            
            # 等待表格加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            
            # 获取页面HTML并解析
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # 查找数据表格
            table = soup.find('table', {'id': 'table1'}) or soup.find('table', {'class': 'tbl01'}) or soup.find('table')
            
            if not table:
                return []
            
            data = []
            rows = table.find_all('tr')[1:]  # 跳过表头
            
            for row in rows:
                cols = row.find_all(['td', 'th'])
                if len(cols) >= 7:
                    time_str = cols[0].get_text(strip=True)
                    score = self._format_score(cols[1].get_text(strip=True))
                    over = cols[2].get_text(strip=True)
                    line = cols[3].get_text(strip=True)
                    under = cols[4].get_text(strip=True)
                    change_time = cols[5].get_text(strip=True)
                    status = cols[6].get_text(strip=True) if len(cols) > 6 else ''
                    
                    data.append([
                        match_id, company_id, time_str, score, over, 
                        line, under, change_time, status
                    ])
            
            return data
            
        except TimeoutException:
            print(f"⚠️  Match {match_id}, Company {company_id}: 页面加载超时")
            return []
        except Exception as e:
            print(f"❌ Match {match_id}, Company {company_id}: {e}")
            return []
    
    def scrape_analysis(self, match_id, output_dir='matches_ana'):
        """
        抓取对阵分析数据
        
        Args:
            match_id: 比赛ID
            output_dir: 输出目录
            
        Returns:
            (头部数据, 是否成功)
        """
        url = f"https://zq.titan007.com/analysis/{match_id}cn.htm"
        
        try:
            self.driver.get(url)
            self._random_delay()
            
            # 等待页面加载
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # ===== 解析头部信息 =====
            header_data = self._parse_analysis_header(soup, match_id)
            
            # ===== 解析详细数据并保存 =====
            self._parse_and_save_analysis_details(soup, match_id, output_dir)
            
            return header_data, True
            
        except Exception as e:
            print(f"❌ Match {match_id} 分析页面抓取失败: {e}")
            return None, False
    
    def _parse_analysis_header(self, soup, match_id):
        """解析对阵分析页面的头部信息"""
        try:
            header_data = {
                'match_id': match_id,
                'home_team_id': '',
                'home_team': '',
                'score': '',
                'away_team_id': '',
                'away_team': '',
                'weather': '',
                'temperature': ''
            }
            
            # 提取主队信息（从链接中提取ID）
            home_link = soup.find('a', href=re.compile(r'/cn/team/Summary/\d+\.html'))
            if home_link:
                match = re.search(r'/cn/team/Summary/(\d+)\.html', home_link['href'])
                if match:
                    header_data['home_team_id'] = match.group(1)
                header_data['home_team'] = home_link.get_text(strip=True)
            
            # 提取客队信息
            away_links = soup.find_all('a', href=re.compile(r'/cn/team/Summary/\d+\.html'))
            if len(away_links) >= 2:
                away_link = away_links[1]
                match = re.search(r'/cn/team/Summary/(\d+)\.html', away_link['href'])
                if match:
                    header_data['away_team_id'] = match.group(1)
                header_data['away_team'] = away_link.get_text(strip=True)
            
            # 提取比分
            score_elem = soup.find('strong', class_='cred')
            if score_elem:
                header_data['score'] = self._format_score(score_elem.get_text(strip=True))
            
            # 提取天气和温度
            weather_div = soup.find('div', class_='weather')
            if weather_div:
                weather_text = weather_div.get_text(strip=True)
                # 解析天气和温度
                if '℃' in weather_text:
                    parts = weather_text.split('℃')
                    if len(parts) >= 2:
                        header_data['temperature'] = parts[0].strip() + '℃'
                        header_data['weather'] = parts[1].strip()
            
            return header_data
            
        except Exception as e:
            print(f"⚠️  解析头部信息失败: {e}")
            return None
    
    def _parse_and_save_analysis_details(self, soup, match_id, output_dir):
        """解析并保存详细分析数据"""
        try:
            # 创建输出目录
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # 查找所有表格
            tables = soup.find_all('table')
            
            analysis_data = {
                '联赛积分排名': [],
                '数据对比_近10场': [],
                '阵容情况': [],
                '对赛往绩': [],
                '近期战绩_主队': [],
                '近期战绩_客队': [],
                '联赛盘路走势': [],
                '相同盘路': [],
                '入球数_上下半场入球分布': [],
                '半全场': [],
                '进球数_单双': [],
                '进球时间': [],
                '本赛季数据统计对比': []
            }
            
            # 遍历所有表格，尝试识别和提取数据
            for idx, table in enumerate(tables):
                rows = table.find_all('tr')
                if not rows:
                    continue
                
                # 提取表格数据
                table_data = []
                for row in rows:
                    cols = row.find_all(['td', 'th'])
                    row_data = [col.get_text(strip=True) for col in cols]
                    if row_data:
                        table_data.append(row_data)
                
                # 根据表头内容判断表格类型
                if table_data:
                    header = ' '.join(table_data[0])
                    
                    if '积分' in header or '排名' in header:
                        analysis_data['联赛积分排名'].extend(table_data)
                    elif '近10场' in header or '最近' in header:
                        analysis_data['数据对比_近10场'].extend(table_data)
                    elif '阵容' in header or '首发' in header:
                        analysis_data['阵容情况'].extend(table_data)
                    elif '往绩' in header or '交锋' in header:
                        analysis_data['对赛往绩'].extend(table_data)
                    elif '盘路' in header:
                        analysis_data['联赛盘路走势'].extend(table_data)
                    elif '入球' in header or '进球分布' in header:
                        analysis_data['入球数_上下半场入球分布'].extend(table_data)
                    elif '半全场' in header:
                        analysis_data['半全场'].extend(table_data)
                    elif '单双' in header:
                        analysis_data['进球数_单双'].extend(table_data)
            
            # 保存为CSV文件
            output_file = os.path.join(output_dir, f'{match_id}_analysis_data.csv')
            
            with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                
                for section_name, section_data in analysis_data.items():
                    if section_data:
                        writer.writerow([f'=== {section_name} ==='])
                        writer.writerows(section_data)
                        writer.writerow([])  # 空行分隔
            
            print(f"  ✓ 详细分析数据已保存: {output_file}")
            
        except Exception as e:
            print(f"⚠️  解析详细数据失败: {e}")
    
    def batch_scrape_handicap(self, match_ids, company_ids, analysis_dir):
        """批量抓取让球数据"""
        print("\n" + "="*60)
        print("开始抓取亚洲让球数据")
        print("="*60)
        print(f"Match IDs: {len(match_ids)} 个")
        print(f"Company IDs: {company_ids}")
        
        self._init_driver()
        
        total = len(match_ids) * len(company_ids)
        processed = 0
        total_records = 0
        
        for match_id in match_ids:
            for company_id in company_ids:
                output_file = analysis_dir + '/' + str(match_id) + '_handicap_live_data_cp' + str(company_id) + '.csv'
                f = open(output_file, 'w', newline='', encoding='utf-8-sig')
                writer = csv.writer(f)
                writer.writerow(['match_id', 'company_id', '时间', '比分', '主队', '盘口', '客队', '变化时间', '状态'])

                processed += 1

                data = self.scrape_handicap(match_id, company_id)
                
                if data:
                    writer.writerows(data)
                    total_records += len(data)
                
                progress = (processed / total) * 100
                print(f"进度: {processed}/{total} ({progress:.1f}%) | 记录数: {total_records}", end='\r')
                
                f.close()
                print(f"💾 文件: {output_file}")

                self._random_delay()
        
        self.driver.quit()
        print(f"\n\n✅ 让球数据抓取完成！")
        print(f"📊 总记录数: {total_records}")
        
    
    def batch_scrape_overunder(self, match_ids, company_ids, analysis_dir):
        """批量抓取大小球数据"""
        print("\n" + "="*60)
        print("开始抓取大小球数据")
        print("="*60)
        print(f"Match IDs: {len(match_ids)} 个")
        print(f"Company IDs: {company_ids}")
        
        self._init_driver()

            
        total = len(match_ids) * len(company_ids)
        processed = 0
        total_records = 0
        
        for match_id in match_ids:
            for company_id in company_ids:
                output_file = analysis_dir + '/' + str(match_id) + '_overunder_live_data_cp' +str(company_id)+ '.csv'
                f = open(output_file, 'w', newline='', encoding='utf-8-sig')
                writer = csv.writer(f)
                writer.writerow(['match_id', 'company_id', '时间', '比分', '大球', '盘口', '小球', '变化时间', '状态'])
                processed += 1
                data = self.scrape_overunder(match_id, company_id)
                
                if data:
                    writer.writerows(data)
                    total_records += len(data)
                
                progress = (processed / total) * 100
                print(f"进度: {processed}/{total} ({progress:.1f}%) | 记录数: {total_records}", end='\r')
                
                print(f"💾 文件: {output_file}")
                f.close()
                self._random_delay()
            
        
        self.driver.quit()
        print(f"\n\n✅ 大小球数据抓取完成！")
        print(f"📊 总记录数: {total_records}")
    
    def batch_scrape_analysis(self, match_ids, daily_str, daily_dir, analysis_dir):
        """批量抓取对阵分析数据"""
        print("\n" + "="*60)
        print("开始抓取对阵分析数据")
        print("="*60)
        print(f"Match IDs: {len(match_ids)} 个")
        
        self._init_driver()
        
        # 创建详细数据输出目录
        #Path(detail_dir).mkdir(parents=True, exist_ok=True)

        output_file = daily_dir + '/' + daily_str + '_matches_header.csv'
        
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['match_id', '主队ID', '主队', '比分', '客队ID', '客队', '天气', '温度'])
            
            processed = 0
            success_count = 0
            
            for match_id in match_ids:
                processed += 1
                header_data, success = self.scrape_analysis(match_id, analysis_dir)
                
                if success and header_data:
                    writer.writerow([
                        header_data['match_id'],
                        header_data['home_team_id'],
                        header_data['home_team'],
                        header_data['score'],
                        header_data['away_team_id'],
                        header_data['away_team'],
                        header_data['weather'],
                        header_data['temperature']
                    ])
                    success_count += 1
                
                progress = (processed / len(match_ids)) * 100
                print(f"进度: {processed}/{len(match_ids)} ({progress:.1f}%) | 成功: {success_count}", end='\r')
                
                self._random_delay()
        
        self.driver.quit()
        print(f"\n\n✅ 对阵分析数据抓取完成！")
        print(f"📊 成功抓取: {success_count}/{len(match_ids)}")
        print(f"💾 头部数据: {output_file}")
        print(f"💾 详细数据: {analysis_dir}/")
    
    def batch_scrape_all(self, match_ids, company_ids, daily_str, daily_dir, analysis_dir):
        """批量抓取所有数据"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("\n" + "="*60)
        print("批量抓取所有数据")
        print("="*60)
        print(f"Match IDs: {len(match_ids)} 个")
        print(f"Company IDs: {company_ids}")
        
        # 1. 抓取让球数据
        # handicap_file = f'{daily_str}_handicap_live_data.csv'
        self.batch_scrape_handicap(match_ids, company_ids, analysis_dir)
        
        print("\n" + "-"*60 + "\n")
        
        # 2. 抓取大小球数据
        # overunder_file = f'{daily_str}_overunder_live_data.csv'
        self.batch_scrape_overunder(match_ids, company_ids, analysis_dir)
        
        print("\n" + "-"*60 + "\n")
        
        # 3. 抓取对阵分析数据
        # self.batch_scrape_analysis(match_ids, daily_str, daily_dir, analysis_dir)
        
        print("\n" + "="*60)
        print("🎉 所有数据抓取完成！")
        print("="*60)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='球探网亚盘数据爬虫工具 (让球+大小球)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 从CSV加载match_id，抓取让球和大小球数据
  python b1_win007_asias2d_scraper.py --csv matches.csv --output-dir data/win007

  # 只抓取让球数据
  python b1_win007_asias2d_scraper.py --csv matches.csv --type handicap --output-dir data/win007

  # 只抓取大小球数据
  python b1_win007_asias2d_scraper.py --csv matches.csv --type overunder --output-dir data/win007

  # 自定义公司ID
  python b1_win007_asias2d_scraper.py --csv matches.csv --companies 1 3 8 --output-dir data/win007
        """
    )

    parser.add_argument(
        '--csv',
        required=True,
        help='包含match_id的CSV文件路径'
    )

    parser.add_argument(
        '--type',
        choices=['handicap', 'overunder', 'all'],
        default='all',
        help='抓取数据类型 (默认: all，抓取让球+大小球)'
    )

    parser.add_argument(
        '--companies',
        type=int,
        nargs='+',
        default=[3, 8],
        help='公司ID列表 (默认: 8)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='限制处理的match_id数量（用于测试）'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/win007',
        help='输出目录 (默认: data/win007)'
    )

    parser.add_argument(
        '--no-headless',
        action='store_true',
        help='显示浏览器窗口'
    )

    parser.add_argument(
        '--delay',
        type=float,
        nargs=2,
        default=[1, 1],
        metavar=('MIN', 'MAX'),
        help='请求延迟范围(秒) (默认: 1 1)'
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 打印欢迎信息
    print("\n" + "="*60)
    print("球探网亚盘数据爬虫工具 (让球+大小球)")
    print("="*60)
    print(f"CSV文件: {args.csv}")
    print(f"抓取类型: {args.type}")
    print(f"公司IDs: {args.companies}")
    print(f"输出目录: {args.output_dir}")
    print(f"运行模式: {'显示浏览器' if args.no_headless else '无头模式'}")
    print(f"请求延迟: {args.delay[0]}-{args.delay[1]}秒")
    print("="*60)

    # 创建爬虫实例
    scraper = TitanFullScraper(
        headless=not args.no_headless,
        delay_range=tuple(args.delay)
    )

    # 加载match_id
    match_ids = scraper.load_match_ids_from_csv(args.csv)

    if not match_ids:
        print("❌ 未能加载任何match_id，程序退出")
        sys.exit(1)

    # 限制数量（用于测试）
    if args.limit:
        match_ids = match_ids[:args.limit]
        print(f"⚠️  限制处理前 {args.limit} 个match_id")

    try:
        if args.type == 'handicap':
            scraper.batch_scrape_handicap(match_ids, args.companies, args.output_dir)
        elif args.type == 'overunder':
            scraper.batch_scrape_overunder(match_ids, args.companies, args.output_dir)
        else:  # all
            scraper.batch_scrape_handicap(match_ids, args.companies, args.output_dir)
            print("\n" + "-"*60 + "\n")
            scraper.batch_scrape_overunder(match_ids, args.companies, args.output_dir)
            print("\n" + "="*60)
            print("所有数据抓取完成！")
            print("="*60)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        if scraper.driver:
            scraper.driver.quit()
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        if scraper.driver:
            scraper.driver.quit()
        sys.exit(1)


if __name__ == '__main__':
    main()
