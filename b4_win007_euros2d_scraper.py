#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
球探网1x2d欧赔JS数据爬虫 - 完整24字段版本
game: 24个字段（包含概率）
gameDetail: 嵌套格式（^ 和 ; 分隔）
"""

import os
import sys
import time
import csv
import random
import argparse
import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Optional
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


class Odds1x2CompleteScraper:
    """1x2d JS数据爬虫 - 完整24字段版本"""
    
    def __init__(self, headless=True, delay_range=(2, 3), use_selenium=False, debug=False):
        """初始化爬虫"""
        self.headless = headless
        self.delay_range = delay_range
        self.use_selenium = use_selenium
        self.debug = debug
        self.driver = None
        self.session = None
        
        if not use_selenium:
            self._init_session()
    
    def _init_session(self):
        """初始化requests session"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Referer': 'https://live.titan007.com/',
        })
        print("✓ Session初始化成功")
    
    def _init_driver(self):
        """初始化Selenium"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            print("✓ WebDriver初始化成功")
        except Exception as e:
            print(f"❌ WebDriver初始化失败: {e}")
            sys.exit(1)
    
    def load_match_ids_from_csv(self, csv_file):
        """从CSV加载match_id"""
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            if 'match_id' not in df.columns:
                print(f"❌ CSV文件中未找到'match_id'列")
                return []
            
            match_ids = df['match_id'].dropna().astype(int).unique().tolist()
            print(f"✓ 从 {csv_file} 加载了 {len(match_ids)} 个唯一的 match_id")
            return match_ids
        except Exception as e:
            print(f"❌ 加载CSV失败: {e}")
            return []
    
    def _random_delay(self):
        """随机延迟"""
        time.sleep(random.uniform(*self.delay_range))
    
    def fetch_js_content(self, match_id) -> Optional[str]:
        """获取JS文件内容"""
        url = f"https://1x2d.titan007.com/{match_id}.js"
        print(url)
        if self.debug:
            print(f"\n{'='*60}")
            print(f"获取 Match {match_id}")
            print(f"URL: {url}")
        
        try:
            if self.use_selenium:
                if not self.driver:
                    self._init_driver()
                
                self.driver.get(url)
                time.sleep(1.5)
                
                try:
                    pre_element = self.driver.find_element(By.TAG_NAME, 'pre')
                    content = pre_element.text
                except:
                    try:
                        body_element = self.driver.find_element(By.TAG_NAME, 'body')
                        content = body_element.text
                    except:
                        page_source = self.driver.page_source
                        if '<pre>' in page_source:
                            content = re.search(r'<pre>(.*?)</pre>', page_source, re.DOTALL).group(1)
                        else:
                            content = page_source
            else:
                response = self.session.get(url, timeout=15)
                response.encoding = 'utf-8'
                content = response.text
            
            if self.debug:
                print(f"  内容长度: {len(content)}")
                print(f"  前200字符: {content[:200]}")
                
                debug_file = f'debug_match_{match_id}.js'
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✓ 已保存到: {debug_file}")
            
            return content
            
        except Exception as e:
            print(f"❌ Match {match_id}: 获取失败 - {e}")
            return None
    
    def parse_var_game(self, js_content, match_id) -> List[Dict]:
        """
        解析 var game=Array("str1","str2",...)
        每个字符串包含24个用|分隔的字段
        输出: match_id + 24个字段 = 25列
        """
        if not js_content:
            return []
        
        try:
            if self.debug:
                print(f"\n解析 var game=Array(...) [完整24字段版本]:")
            
            # 匹配 var game=Array(...)
            patterns = [
                r'var\s+game\s*=\s*Array\((.*?)\);',
                r'var\s+game\s*=\s*Array\((.*?)\)',
                r'(?:^|\n)\s*game\s*=\s*Array\((.*?)\);',
                r'(?:^|\n)\s*game\s*=\s*Array\((.*?)\)',
                r'game\s*=\s*Array\((.*?)\)',
            ]
            
            game_str = None
            
            for pattern in patterns:
                match = re.search(pattern, js_content, re.DOTALL | re.MULTILINE)
                if match:
                    game_str = match.group(1)
                    if self.debug:
                        print(f"  ✓ 匹配成功")
                        print(f"  ✓ 提取长度: {len(game_str)}")
                    break
            
            if not game_str:
                if self.debug:
                    print(f"  ✗ 未找到 game 变量")
                return []
            
            # 提取所有字符串
            string_pattern = r'''["']([^"']+?)["']'''
            data_strings = re.findall(string_pattern, game_str)
            
            if self.debug:
                print(f"  ✓ 找到 {len(data_strings)} 条记录")
            
            records = []
            
            for idx, data_str in enumerate(data_strings):
                # 按管道符分割成24个字段
                fields = data_str.split('|')
                
                if self.debug and idx < 3:
                    print(f"\n  记录 {idx+1}:")
                    print(f"    字段数: {len(fields)}")
                    print(f"    前10个字段: {fields[:10]}")
                
                # 确保至少有24个字段
                while len(fields) < 24:
                    fields.append('')
                
                # 构建记录: match_id + 24个字段
                # 完整的24字段表头:
                # match_id, num, eu_cp_id, eu_cp_name,
                # home_win_odd0, draw_odd0, away_win_odd0, home_win_p0, draw_p0, away_win_p0, return_rate0,
                # home_win_odd1, draw_odd1, away_win_odd1, home_win_p1, draw_p1, away_win_p1, return_rate1,
                # kelly_home, kelly_draw, kelly_away, datestr, eu_cp_name_ex, flag1, flag2
                record = {
                    'match_id': match_id,
                    'num': fields[0] if len(fields) > 0 else '',
                    'eu_cp_id': fields[1] if len(fields) > 1 else '',
                    'eu_cp_name': fields[2] if len(fields) > 2 else '',
                    'home_win_odd0': fields[3] if len(fields) > 3 else '',
                    'draw_odd0': fields[4] if len(fields) > 4 else '',
                    'away_win_odd0': fields[5] if len(fields) > 5 else '',
                    'home_win_p0': fields[6] if len(fields) > 6 else '',
                    'draw_p0': fields[7] if len(fields) > 7 else '',
                    'away_win_p0': fields[8] if len(fields) > 8 else '',
                    'return_rate0': fields[9] if len(fields) > 9 else '',
                    'home_win_odd1': fields[10] if len(fields) > 10 else '',
                    'draw_odd1': fields[11] if len(fields) > 11 else '',
                    'away_win_odd1': fields[12] if len(fields) > 12 else '',
                    'home_win_p1': fields[13] if len(fields) > 13 else '',
                    'draw_p1': fields[14] if len(fields) > 14 else '',
                    'away_win_p1': fields[15] if len(fields) > 15 else '',
                    'return_rate1': fields[16] if len(fields) > 16 else '',
                    'kelly_home': fields[17] if len(fields) > 17 else '',
                    'kelly_draw': fields[18] if len(fields) > 18 else '',
                    'kelly_away': fields[19] if len(fields) > 19 else '',
                    'datestr': fields[20] if len(fields) > 20 else '',
                    'eu_cp_name_ex': fields[21] if len(fields) > 21 else '',
                    'flag1': fields[22] if len(fields) > 22 else '',
                    'flag2': fields[23] if len(fields) > 23 else '',
                }
                records.append(record)
            
            if self.debug:
                print(f"\n  ✓ 解析出 {len(records)} 条记录")
                if records:
                    r = records[0]
                    print(f"  示例: {r['eu_cp_name']}")
                    print(f"    开盘: {r['home_win_odd0']}, {r['draw_odd0']}, {r['away_win_odd0']} (概率: {r['home_win_p0']}%, {r['draw_p0']}%, {r['away_win_p0']}%)")
                    print(f"    收盘: {r['home_win_odd1']}, {r['draw_odd1']}, {r['away_win_odd1']} (概率: {r['home_win_p1']}%, {r['draw_p1']}%, {r['away_win_p1']}%)")
            
            return records
            
        except Exception as e:
            if self.debug:
                print(f"  ✗ 解析失败: {e}")
                import traceback
                traceback.print_exc()
            return []
    
    def parse_var_gameDetail(self, js_content, match_id) -> List[Dict]:
        """
        解析 var gameDetail=Array("str1","str2",...)
        格式: eu_cp_id^home_win_odd|draw_odd|away_win_odd|date_str|kelly_home|kelly_draw|kelly_away;...
        输出: match_id, eu_cp_id, home_win_odd, draw_odd, away_win_odd, date_str, kelly_home, kelly_draw, kelly_away
        """
        if not js_content:
            return []
        
        try:
            if self.debug:
                print(f"\n解析 var gameDetail=Array(...) [嵌套格式版本]:")
            
            # 匹配 var gameDetail=Array(...)
            patterns = [
                r'var\s+gameDetail\s*=\s*Array\((.*?)\);',
                r'var\s+gameDetail\s*=\s*Array\((.*?)\)',
                r'(?:^|\n)\s*gameDetail\s*=\s*Array\((.*?)\);',
                r'(?:^|\n)\s*gameDetail\s*=\s*Array\((.*?)\)',
                r'gameDetail\s*=\s*Array\((.*?)\)',
            ]
            
            detail_str = None
            
            for pattern in patterns:
                match = re.search(pattern, js_content, re.DOTALL | re.MULTILINE)
                if match:
                    detail_str = match.group(1)
                    if self.debug:
                        print(f"  ✓ 匹配成功")
                        print(f"  ✓ 提取长度: {len(detail_str)}")
                    break
            
            if not detail_str:
                if self.debug:
                    print(f"  ✗ 未找到 gameDetail 变量")
                return []
            
            # 提取所有字符串
            string_pattern = r'''["']([^"']+?)["']'''
            data_strings = re.findall(string_pattern, detail_str)
            
            if self.debug:
                print(f"  ✓ 找到 {len(data_strings)} 个元素")
            
            records = []
            
            for idx, data_str in enumerate(data_strings):
                if self.debug and idx < 2:
                    print(f"\n  元素 {idx+1}:")
                    print(f"    长度: {len(data_str)}")
                    print(f"    前150字符: {data_str[:150]}...")
                
                # 格式: eu_cp_id^record1;record2;...;recordN
                # 其中每个record: home_win_odd|draw_odd|away_win_odd|date_str|kelly_home|kelly_draw|kelly_away
                
                if '^' not in data_str:
                    if self.debug and idx < 2:
                        print(f"    ⚠️  没有'^'分隔符，跳过")
                    continue
                
                # 分割出 eu_cp_id 和 走势记录
                parts = data_str.split('^', 1)
                if len(parts) != 2:
                    continue
                
                eu_cp_id = parts[0]
                trend_data = parts[1]
                
                if self.debug and idx < 2:
                    print(f"    eu_cp_id: {eu_cp_id}")
                    print(f"    走势数据长度: {len(trend_data)}")
                
                # 按分号分割多条走势记录
                trend_records = trend_data.split(';')
                
                if self.debug and idx < 2:
                    print(f"    走势记录数: {len(trend_records)}")
                
                for trend_idx, trend_record in enumerate(trend_records):
                    if not trend_record.strip():
                        continue
                    
                    # 按管道符分割字段
                    # 格式: home_win_odd|draw_odd|away_win_odd|date_str|kelly_home|kelly_draw|kelly_away
                    fields = trend_record.split('|')
                    
                    if len(fields) < 7:
                        continue
                    
                    if self.debug and idx < 2 and trend_idx < 3:
                        print(f"      走势{trend_idx+1}: {fields}")
                    
                    record = {
                        'match_id': match_id,
                        'eu_cp_id': eu_cp_id,
                        'home_win_odd': fields[0] if len(fields) > 0 else '',
                        'draw_odd': fields[1] if len(fields) > 1 else '',
                        'away_win_odd': fields[2] if len(fields) > 2 else '',
                        'date_str': fields[3] if len(fields) > 3 else '',
                        'kelly_home': fields[4] if len(fields) > 4 else '',
                        'kelly_draw': fields[5] if len(fields) > 5 else '',
                        'kelly_away': fields[6] if len(fields) > 6 else '',
                    }
                    records.append(record)
            
            if self.debug:
                print(f"\n  ✓ 解析出 {len(records)} 条走势记录")
            
            return records
            
        except Exception as e:
            if self.debug:
                print(f"  ✗ 解析失败: {e}")
                import traceback
                traceback.print_exc()
            return []
    
    def batch_scrape(self, match_ids, output_dir):
        """批量抓取"""
        print("\n" + "="*60)
        print("开始抓取1x2d欧赔JS数据（完整24字段版本）")
        print("="*60)
        print(f"Match IDs: {len(match_ids)} 个")
        print(f"模式: {'Selenium' if self.use_selenium else 'Requests'}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # main表头: match_id + 24个字段
        main_fieldnames = [
            'match_id', 'num', 'eu_cp_id', 'eu_cp_name',
            'home_win_odd0', 'draw_odd0', 'away_win_odd0', 'home_win_p0', 'draw_p0', 'away_win_p0', 'return_rate0',
            'home_win_odd1', 'draw_odd1', 'away_win_odd1', 'home_win_p1', 'draw_p1', 'away_win_p1', 'return_rate1',
            'kelly_home', 'kelly_draw', 'kelly_away', 'datestr', 'eu_cp_name_ex', 'flag1', 'flag2'
        ]


        
        processed = 0
        main_records = 0
        detail_records = 0
        success_count = 0
        
        for match_id in match_ids:
            main_output =  output_dir + '/' + str(match_id)+'_euro1x2_s2d_data.csv'
            
            main_file = open(main_output, 'w', newline='', encoding='utf-8-sig')
            
            main_writer = csv.DictWriter(main_file, fieldnames=main_fieldnames)
            
            main_writer.writeheader()

            processed += 1
            
            js_content = self.fetch_js_content(match_id)
            
            if js_content:
                # 解析开盘+收盘
                main_data = self.parse_var_game(js_content, match_id)
                if main_data:
                    main_writer.writerows(main_data)
                    main_file.flush()
                    main_records += len(main_data)
                
                # 解析走势

            if not self.debug:
                progress = (processed / len(match_ids)) * 100
                print(f"进度: {processed}/{len(match_ids)} ({progress:.1f}%) | "
                      f"成功: {success_count} | 开盘收盘: {main_records} | 走势: {detail_records}", 
                      end='\r')

            main_file.close()
            print(main_output, '[', processed, '],   开盘收盘:', main_records)
            self._random_delay()
        
        
        if self.use_selenium and self.driver:
            self.driver.quit()
        
        print(f"\n\n✅ 抓取完成！")
        print(f"📊 处理: {processed} | 成功: {success_count}")
        print(f"📊 开盘收盘记录: {main_records}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='球探网1x2d欧赔JS爬虫（完整24字段版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python b4_win007_euros2d_scraper.py --csv matches.csv --output-dir data/win007
  python b4_win007_euros2d_scraper.py --csv matches.csv --output-dir data/win007 --limit 10 --debug
        """
    )

    parser.add_argument('--csv', required=True, help='CSV文件路径')
    parser.add_argument('--output-dir', default='data/win007', help='输出目录 (默认: data/win007)')
    parser.add_argument('--limit', type=int, help='限制数量')
    parser.add_argument('--use-selenium', action='store_true', help='使用Selenium')
    parser.add_argument('--no-headless', action='store_true', help='显示浏览器')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--delay', type=float, nargs=2, default=[2, 3], help='延迟范围')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("1x2d欧赔JS爬虫（完整24字段版本）")
    print("="*60)
    print(f"CSV文件: {args.csv}")
    print(f"输出目录: {args.output_dir}")

    scraper = Odds1x2CompleteScraper(
        headless=not args.no_headless,
        delay_range=tuple(args.delay),
        use_selenium=args.use_selenium,
        debug=args.debug
    )

    match_ids = scraper.load_match_ids_from_csv(args.csv)

    if not match_ids:
        sys.exit(1)

    if args.limit:
        match_ids = match_ids[:args.limit]
        print(f"⚠️  限制处理前 {args.limit} 个")

    try:
        scraper.batch_scrape(match_ids, args.output_dir)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        if scraper.driver:
            scraper.driver.quit()
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        if scraper.driver:
            scraper.driver.quit()


if __name__ == '__main__':
    main()
