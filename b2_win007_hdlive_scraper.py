#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
亚洲让球赔率爬虫脚本
功能: 批量抓取亚洲让球赔率数据,解析初盘终盘以及所有历史记录
"""

import csv
import time
import os
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import argparse


class AsianOddsScraper:
    """亚洲让球赔率爬虫类"""
    
    def __init__(self, headless=True, delay=1):
        """
        初始化爬虫
        :param headless: 是否使用无头模式
        :param delay: 页面加载延迟(秒)
        """
        self.headless = headless
        self.delay = delay
        self.driver = None
        
    def setup_driver(self):
        """设置Selenium WebDriver"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        # 反爬虫策略设置
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # 随机User-Agent
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # 禁用图片加载以加快速度
        prefs = {
            'profile.managed_default_content_settings.images': 2,
            'permissions.default.stylesheet': 2
        }
        chrome_options.add_experimental_option('prefs', prefs)
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            '''
        })
        
    def load_match_ids(self, csv_file):
        """
        从CSV文件加载match_id列表
        :param csv_file: CSV文件路径
        :return: match_id列表
        """
        match_ids = []
        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'match_id' in row and row['match_id']:
                        match_ids.append(row['match_id'])
            print(f"✓ 成功加载 {len(match_ids)} 个 match_id")
            return match_ids
        except Exception as e:
            print(f"✗ 加载CSV文件失败: {e}")
            return []
    
    def fetch_page(self, match_id):
        """
        获取指定match_id的页面
        :param match_id: 比赛ID
        :return: 是否成功
        """
        url = f"https://vip.titan007.com/AsianOdds_n.aspx?id={match_id}&l=0"
        try:
            print(f"  正在访问: {url}")
            self.driver.get(url)
            time.sleep(self.delay)
            
            # 等待页面加载完成
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "odds"))
            )
            return True
        except TimeoutException:
            print(f"  ✗ 页面加载超时: {match_id}")
            return False
        except Exception as e:
            print(f"  ✗ 访问页面失败: {e}")
            return False
    
    def parse_main_table(self, match_id):
        """
        解析初盘和终盘表格 (table#odds)
        :param match_id: 比赛ID
        :return: 解析结果列表
        """
        results = []
        try:
            table = self.driver.find_element(By.ID, "odds")
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            for row in rows:
                try:
                    tds = row.find_elements(By.TAG_NAME, "td")
                    
                    # 至少需要12个td
                    if len(tds) < 12:
                        continue
                    
                    # 第2个td是公司名(index=1)
                    company = tds[1].text.strip()
                    
                    # 如果公司名为空,跳过此行
                    if not company:
                        continue
                    
                    # 过滤最大值、最小值等非博彩公司的行
                    if '最大值' in company or '最小值' in company or '平均值' in company:
                        continue
                    
                    # 清理company字段: 去掉"封"关键词和回车换行符
                    company = company.replace('封', '').replace('\n', '').replace('\r', '').strip()
                    
                    # 提取7个字段
                    
                    results.append({
                        'match_id': match_id,
                        'company': company,
                        '初盘-主队': tds[3].text.strip(),
                        '初盘盘口': tds[4].text.strip(),
                        '初盘-客队': tds[5].text.strip(),
                        '终盘-主队': tds[9].text.strip(),
                        '终盘盘口': tds[10].text.strip(),
                        '终盘-客队': tds[11].text.strip()
                    })
                except Exception as e:
                    continue
            
            print(f"  ✓ 解析初盘/终盘表格: {len(results)} 条记录")
            return results
            
        except NoSuchElementException:
            print(f"  ✗ 未找到初盘/终盘表格")
            return []
        except Exception as e:
            print(f"  ✗ 解析初盘/终盘表格失败: {e}")
            return []
    
    def clean_cell_text(self, text):
        """
        清理单元格文本,去除换行符,用空格分割
        :param text: 原始文本
        :return: 清理后的文本
        """
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', ' ', text)
        # 替换换行符为空格
        text = text.replace('\n', ' ').replace('\r', ' ')
        # 合并多个空格为一个
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def find_odds_change_table(self):
        """
        查找赔率变化表格(尝试多种方式)
        :return: table元素或None
        """
        # 尝试的ID列表
        possible_ids = ["oddsDetail", "oddsList", "table_detail", "detail", "changeList"]
        
        for table_id in possible_ids:
            try:
                table = self.driver.find_element(By.ID, table_id)
                print(f"  ✓ 通过ID '{table_id}' 找到表格")
                return table
            except:
                continue
        
        # 通过CSS选择器尝试
        possible_selectors = [
            "table[id*='detail']",
            "table[id*='odds']",
            "table.detail",
            "table.oddsList"
        ]
        
        for selector in possible_selectors:
            try:
                table = self.driver.find_element(By.CSS_SELECTOR, selector)
                print(f"  ✓ 通过选择器 '{selector}' 找到表格")
                return table
            except:
                continue
        
        # 通过内容特征查找
        try:
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                table_html = table.get_attribute('outerHTML')
                table_text = table.text
                # 检查是否包含赔率变化相关的关键词
                if any(keyword in table_text for keyword in ["变化时间", "时间", "盘口", "主队", "客队"]):
                    # 进一步检查是否是历史记录表(通常有多行数据)
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    if len(rows) > 5:  # 历史记录通常有多行
                        print(f"  ✓ 通过内容特征找到表格(共{len(rows)}行)")
                        return table
        except:
            pass
        
        print(f"  ✗ 未找到赔率变化表格")
        return None
    
    def parse_detail_table(self, match_id):
        """
        解析赛前+滚球所有记录表格 (table#oddsDetail)
        超优化版本: 使用JavaScript批量获取数据,性能提升10倍以上
        :param match_id: 比赛ID
        :return: 紧凑格式的记录列表
        """
        compact_results = []
        
        try:
            # 使用辅助方法查找表格
            table = self.find_odds_change_table()
            
            if not table:
                print(f"  ⚠ 未找到历史记录表格,可能该比赛没有历史记录")
                return []
            
            # 性能优化关键: 使用JavaScript一次性获取所有数据
            # 这比逐个调用get_attribute快10-20倍
            js_script = """
            var table = arguments[0];
            var result = {headers: [], rows: []};
            
            // 获取表头
            var thead = table.querySelector('thead');
            if (thead) {
                var ths = thead.querySelectorAll('th');
                for (var i = 0; i < ths.length; i++) {
                    result.headers.push(ths[i].textContent.trim());
                }
            } else {
                var firstRow = table.querySelector('tr');
                if (firstRow) {
                    var cells = firstRow.querySelectorAll('th, td');
                    for (var i = 0; i < cells.length; i++) {
                        result.headers.push(cells[i].textContent.trim());
                    }
                }
            }
            
            // 获取所有数据行
            var tbody = table.querySelector('tbody');
            var rows = tbody ? tbody.querySelectorAll('tr') : table.querySelectorAll('tr');
            
            // 跳过表头行(如果tbody不存在且第一行是表头)
            var startIdx = (tbody || result.headers.length === 0) ? 0 : 1;
            
            for (var i = startIdx; i < rows.length; i++) {
                var tds = rows[i].querySelectorAll('td');
                if (tds.length < 2) continue;
                
                var rowData = {
                    score: tds[tds.length - 2].textContent.trim(),
                    time: tds[tds.length - 1].textContent.trim(),
                    cells: []
                };
                
                // 获取所有单元格的innerHTML(除了最后两列)
                for (var j = 0; j < tds.length - 2; j++) {
                    rowData.cells.push(tds[j].innerHTML);
                }
                
                result.rows.push(rowData);
            }
            
            return result;
            """
            
            # 执行JavaScript,一次性获取所有数据
            data = self.driver.execute_script(js_script, table)
            
            headers = data.get('headers', [])
            rows = data.get('rows', [])
            
            print(f"  ✓ 表头字段数: {len(headers)}")
            print(f"  ✓ 数据行数: {len(rows)}")
            
            if not rows:
                print(f"  ⚠ 表格中没有数据行")
                return []
            
            # 在Python中处理数据(已经在内存中,无需再访问DOM)
            for row_data in rows:
                比分 = row_data['score']
                变化时间 = row_data['time']
                cells_html = row_data['cells']
                
                for i, cell_html in enumerate(cells_html):
                    cell_text = self.clean_cell_text(cell_html)
                    
                    # 如果单元格有内容
                    if cell_text:
                        # 确定公司名
                        if headers and i < len(headers):
                            company = headers[i]
                        else:
                            company = f"公司{i+1}"
                        
                        compact_results.append({
                            'match_id': match_id,
                            'company': company,
                            'handicap_odds': cell_text,
                            '比分': 比分,
                            '变化时间': 变化时间
                        })
            
            print(f"  ✓ 解析历史记录表格: {len(compact_results)} 条记录")
            return compact_results
            
        except Exception as e:
            print(f"  ✗ 解析历史记录表格失败: {e}")
            return []
    
    def save_to_csv(self, data, filename, fieldnames):
        """
        保存数据到CSV文件
        :param data: 数据列表
        :param filename: 文件名
        :param fieldnames: 字段名列表
        """
        try:
            with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            print(f"✓ 数据已保存到: {filename}")
        except Exception as e:
            print(f"✗ 保存CSV文件失败: {e}")
    
    def append_to_csv(self, data, filename, fieldnames):
        """
        追加数据到CSV文件(如果文件不存在则创建)
        :param data: 数据列表
        :param filename: 文件名
        :param fieldnames: 字段名列表
        """
        try:
            # 检查文件是否存在
            file_exists = os.path.isfile(filename)
            
            with open(filename, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # 如果文件不存在,写入表头
                if not file_exists:
                    writer.writeheader()
                
                # 追加数据
                writer.writerows(data)
            
            if file_exists:
                print(f"  ✓ 数据已追加到: {filename} ({len(data)} 条)")
            else:
                print(f"  ✓ 数据已保存到: {filename} ({len(data)} 条)")
        except Exception as e:
            print(f"  ✗ 保存CSV文件失败: {e}")
    
    def scrape_single_match(self, match_id, output_dir='output', save_html=False):
        """
        抓取单个比赛的数据
        :param match_id: 比赛ID
        :param output_dir: 输出目录
        :param save_html: 是否保存页面HTML用于调试
        """
        print(f"\n[{match_id}] 开始抓取...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取页面
        if not self.fetch_page(match_id):
            return False
        
        # 如果需要,保存页面HTML用于调试
        if save_html:
            try:
                html_content = self.driver.page_source
                html_file = os.path.join(output_dir, f"{match_id}_page.html")
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"  ✓ 页面HTML已保存: {html_file}")
            except Exception as e:
                print(f"  ⚠ 保存HTML失败: {e}")
        
        # 解析初盘/终盘表格
        main_data = self.parse_main_table(match_id)
        if main_data:
            # 追加到统一的asia_main.csv文件
            main_filename = output_dir + '/' + str(match_id) + '_handicap_s2d_data.csv'
            fieldnames = ['match_id', 'company', '初盘-主队', '初盘盘口', '初盘-客队', '终盘-主队', '终盘盘口', '终盘-客队']
            with open(main_filename, 'w', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(main_data)
        
        # 解析历史记录表格
        '''
        detail_data = self.parse_detail_table(match_id)
        if detail_data:
            # 追加到统一的asia_details.csv文件
            detail_filename = os.path.join(output_dir, "asia_details.csv")
            self.append_to_csv(
                detail_data,
                detail_filename,
                ['match_id', 'company', 'handicap_odds', '比分', '变化时间']
            )
        else:
            print(f"  ⚠ 没有解析到历史记录数据")
        '''
        
        print(f"[{match_id}] ✓ 完成")
        return True
    
    def scrape_batch(self, match_ids, output_dir='output', start_index=0, limit=None):
        """
        批量抓取多个比赛的数据
        :param match_ids: 比赛ID列表
        :param output_dir: 输出目录
        :param start_index: 起始索引
        :param limit: 限制抓取数量
        """
        if not match_ids:
            print("✗ 没有可抓取的match_id")
            return
        
        # 设置抓取范围
        end_index = len(match_ids) if limit is None else min(start_index + limit, len(match_ids))
        total = end_index - start_index
        
        print(f"\n{'='*60}")
        print(f"批量抓取任务")
        print(f"总数: {len(match_ids)} | 本次抓取: {total} (索引 {start_index} 到 {end_index-1})")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}\n")
        
        success_count = 0
        fail_count = 0
        
        for i, match_id in enumerate(match_ids[start_index:end_index], start=start_index+1):
            print(f"\n[进度: {i}/{end_index}]")
            
            try:
                if self.scrape_single_match(match_id, output_dir):
                    success_count += 1
                else:
                    fail_count += 1
                
                # 随机延迟,避免被封
                time.sleep(self.delay + (i % 3))
                
            except KeyboardInterrupt:
                print("\n\n用户中断抓取")
                break
            except Exception as e:
                print(f"✗ 抓取失败: {e}")
                fail_count += 1
        
        print(f"\n{'='*60}")
        print(f"批量抓取完成")
        print(f"成功: {success_count} | 失败: {fail_count}")
        print(f"{'='*60}\n")
    
    def close(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='亚洲让球赔率批量爬虫工具 (初盘终盘)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法 - 抓取所有比赛
  python b2_win007_hdlive_scraper.py --csv matches.csv --output-dir data/win007

  # 限制抓取数量
  python b2_win007_hdlive_scraper.py --csv matches.csv --limit 10 --output-dir data/win007

  # 显示浏览器窗口(调试用)
  python b2_win007_hdlive_scraper.py --csv matches.csv --no-headless --limit 5 --output-dir data/win007
        """
    )

    parser.add_argument('--csv', required=True, help='包含match_id的CSV文件路径')
    parser.add_argument('--output-dir', default='data/win007', help='输出目录 (默认: data/win007)')
    parser.add_argument('--start', '-s', type=int, default=0, help='起始索引 (默认: 0)')
    parser.add_argument('--limit', '-l', type=int, default=None, help='限制抓取数量 (默认: 全部)')
    parser.add_argument('--delay', '-d', type=float, default=2, help='页面加载延迟秒数 (默认: 2)')
    parser.add_argument('--no-headless', action='store_true', help='显示浏览器窗口(调试用)')

    args = parser.parse_args()

    # 检查CSV文件是否存在
    if not os.path.exists(args.csv):
        print(f"✗ 错误: 文件不存在 - {args.csv}")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建爬虫实例
    scraper = AsianOddsScraper(
        headless=not args.no_headless,
        delay=args.delay
    )

    try:
        # 设置WebDriver
        print("正在初始化浏览器...")
        scraper.setup_driver()
        print("✓ 浏览器初始化完成\n")

        # 加载match_id列表
        match_ids = scraper.load_match_ids(args.csv)

        if not match_ids:
            print("✗ 没有找到有效的match_id")
            return

        # 批量抓取
        scraper.scrape_batch(
            match_ids,
            output_dir=args.output_dir,
            start_index=args.start,
            limit=args.limit
        )

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
    finally:
        print("\n正在关闭浏览器...")
        scraper.close()
        print("✓ 完成")


if __name__ == '__main__':
    main()
