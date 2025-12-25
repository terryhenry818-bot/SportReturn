#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
亚洲大小球盘赔率爬虫工具 - 修复版 v1.1
功能：批量爬取指定match_id的大小球赔率数据
修复：
1. company列去除"封"和换行符
2. details表格使用JavaScript优化解析，提升性能和成功率
"""

import csv
import time
import os
import sys
import re
from typing import List, Dict, Optional
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


class OverUnderScraper:
    """大小球盘赔率爬虫类"""
    
    def __init__(self, headless: bool = True, timeout: int = 20):
        """
        初始化爬虫
        :param headless: 是否使用无头模式
        :param timeout: 页面加载超时时间（秒）
        """
        self.headless = headless
        self.timeout = timeout
        self.driver = None
        
    def init_driver(self):
        """初始化Selenium WebDriver"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        # 反爬虫策略：模拟真实浏览器
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # 设置User-Agent
        chrome_options.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        
        # 设置窗口大小
        chrome_options.add_argument('--window-size=1920,1080')
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # 执行CDP命令隐藏webdriver特征
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                '''
            })
            
            self.driver.set_page_load_timeout(self.timeout)
            print("✓ WebDriver初始化成功")
            
        except Exception as e:
            print(f"✗ WebDriver初始化失败: {e}")
            raise
    
    def load_match_ids(self, csv_file: str) -> List[str]:
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
                        match_ids.append(row['match_id'].strip())
            
            print(f"✓ 成功加载 {len(match_ids)} 个match_id")
            return match_ids
            
        except Exception as e:
            print(f"✗ 读取CSV文件失败: {e}")
            raise
    
    def fetch_page(self, match_id: str) -> bool:
        """
        访问指定match_id的页面
        :param match_id: 比赛ID
        :return: 是否成功加载
        """
        url = f"https://vip.titan007.com/OverDown_n.aspx?id={match_id}&l=0"
        
        try:
            self.driver.get(url)
            
            # 等待关键元素加载
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.ID, "odds"))
            )
            
            # 额外等待JavaScript执行
            time.sleep(2)
            
            return True
            
        except TimeoutException:
            print(f"  ⚠ match_id {match_id} 页面加载超时")
            return False
        except Exception as e:
            print(f"  ⚠ match_id {match_id} 访问失败: {e}")
            return False
    
    def clean_company_name(self, company: str) -> str:
        """
        清理公司名称：去除"封"和换行/回车符
        :param company: 原始公司名
        :return: 清理后的公司名
        """
        if not company:
            return ""
        
        # 去除"封"字
        company = company.replace('封', '')
        # 去除换行符
        company = company.replace('\n', '').replace('\r', '')
        # 去除多余空格
        company = re.sub(r'\s+', ' ', company)
        return company.strip()
    
    def parse_main_table(self, match_id: str) -> List[Dict]:
        """
        解析初盘和终盘表格 (table id='odds')
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
                    
                    if len(tds) < 12:
                        continue
                    
                    # 第2个td是公司名（索引为1）
                    company_raw = tds[1].text.strip()
                    
                    # 如果公司名为空，跳过这行
                    if not company_raw:
                        continue
                    
                    # 清理公司名：去除"封"和换行符
                    company = self.clean_company_name(company_raw)
                    
                    # 清理后仍为空，跳过
                    if not company:
                        continue
                    
                    # 过滤统计行
                    if any(keyword in company for keyword in ['最大值', '最小值', '平均值']):
                        continue
                    
                    # 解析各个字段
                    result = {
                        'match_id': match_id,
                        '公司名': company,
                        '初盘-大球': tds[3].text.strip(),
                        '初盘-盘': tds[4].text.strip(),
                        '初盘-小球': tds[5].text.strip(),
                        '终盘-大球': tds[9].text.strip(),
                        '终盘-盘': tds[10].text.strip(),
                        '终盘-小球': tds[11].text.strip()
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    continue
            
            return results
            
        except NoSuchElementException:
            print(f"  ⚠ match_id {match_id} 未找到odds表格")
            return []
        except Exception as e:
            print(f"  ⚠ match_id {match_id} 解析odds表格失败: {e}")
            return []
    
    def clean_cell_text(self, text: str) -> str:
        """
        清理单元格文本，去除换行符，用空格连接
        :param text: 原始文本
        :return: 清理后的文本
        """
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', ' ', text)
        # 替换换行符为空格
        text = text.replace('\n', ' ').replace('\r', ' ')
        # 替换多个空白字符为单个空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def find_detail_table(self):
        """
        查找赔率变化详细表格（尝试多种方式）
        :return: table元素或None
        """
        # 尝试的ID列表
        possible_ids = ["oddsDetail", "oddsList", "table_detail", "detail", "changeList"]
        
        for table_id in possible_ids:
            try:
                table = self.driver.find_element(By.ID, table_id)
                print(f"  ✓ 通过ID '{table_id}' 找到详细表格")
                return table
            except:
                continue
        
        # 通过CSS选择器尝试
        possible_selectors = [
            "table[id*='detail']",
            "table[id*='Detail']",
            "table[id*='odds']"
        ]
        
        for selector in possible_selectors:
            try:
                table = self.driver.find_element(By.CSS_SELECTOR, selector)
                print(f"  ✓ 通过选择器 '{selector}' 找到详细表格")
                return table
            except:
                continue
        
        # 通过内容特征查找
        try:
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                table_text = table.text
                # 检查是否包含关键词
                if any(keyword in table_text for keyword in ["变化时间", "比分"]):
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    # 历史记录通常有多行
                    if len(rows) > 3:
                        print(f"  ✓ 通过内容特征找到详细表格(共{len(rows)}行)")
                        return table
        except:
            pass
        
        print(f"  ⚠ 未找到详细记录表格")
        return None
    
    def parse_detail_table(self, match_id: str) -> List[Dict]:
        """
        解析赛前+滚球所有记录表格 (table#oddsDetail)
        使用JavaScript批量获取数据，性能提升10倍以上
        :param match_id: 比赛ID
        :return: 紧凑格式的记录列表
        """
        compact_results = []
        
        try:
            # 查找表格
            table = self.find_detail_table()
            
            if not table:
                print(f"  ⚠ 未找到详细记录表格，可能该比赛没有历史记录")
                return []
            
            # 性能优化关键：使用JavaScript一次性获取所有数据
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
            
            // 跳过表头行（如果tbody不存在且第一行是表头）
            var startIdx = (tbody || result.headers.length === 0) ? 0 : 1;
            
            for (var i = startIdx; i < rows.length; i++) {
                var tds = rows[i].querySelectorAll('td');
                if (tds.length < 2) continue;
                
                var rowData = {
                    score: tds[tds.length - 2].textContent.trim(),
                    time: tds[tds.length - 1].textContent.trim(),
                    cells: []
                };
                
                // 获取所有单元格的innerHTML（除了最后两列）
                for (var j = 0; j < tds.length - 2; j++) {
                    rowData.cells.push(tds[j].innerHTML);
                }
                
                result.rows.push(rowData);
            }
            
            return result;
            """
            
            # 执行JavaScript，一次性获取所有数据
            data = self.driver.execute_script(js_script, table)
            
            headers = data.get('headers', [])
            rows = data.get('rows', [])
            
            print(f"  ✓ 表头字段数: {len(headers)}")
            print(f"  ✓ 数据行数: {len(rows)}")
            
            if not rows:
                print(f"  ⚠ 表格中没有数据行")
                return []
            
            # 在Python中处理数据（已经在内存中，无需再访问DOM）
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
                            company_raw = headers[i]
                        else:
                            company_raw = f"公司{i+1}"
                        
                        # 清理公司名
                        company = self.clean_company_name(company_raw)
                        
                        compact_results.append({
                            'match_id': match_id,
                            'company': company,
                            'overunder_odds': cell_text,
                            '比分': 比分,
                            '变化时间': 变化时间
                        })
            
            return compact_results
            
        except Exception as e:
            print(f"  ⚠ 解析详细记录表格失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def save_main_csv(self, data: List[Dict], output_file: str):
        """
        保存初盘/终盘数据到CSV
        :param data: 数据列表
        :param output_file: 输出文件路径
        """
        if not data:
            return
        
        fieldnames = [
            'match_id', '公司名', '初始时间-大球', '初始时间-盘', '初始时间-小球',
            'wholeOdds-大球', 'wholeOdds-盘', 'wholeOdds-小球'
        ]
        
        file_exists = os.path.exists(output_file)
        
        with open(output_file, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(data)
    
    def save_detail_csv(self, data: List[Dict], output_file: str):
        """
        保存详细记录数据到CSV
        :param data: 数据列表
        :param output_file: 输出文件路径
        """
        if not data:
            return
        
        fieldnames = ['match_id', 'company', 'overunder_odds', '比分', '变化时间']
        
        file_exists = os.path.exists(output_file)
        
        with open(output_file, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(data)
    
    def scrape(self, match_ids: List[str], output_dir: str = '.', 
               delay: float = 2.0, batch_size: int = 10):
        """
        批量爬取数据
        :param match_ids: match_id列表
        :param output_dir: 输出目录
        :param delay: 请求延迟（秒）
        :param batch_size: 每批处理数量（用于进度显示和保存）
        """
        if not self.driver:
            self.init_driver()
        

        
        total = len(match_ids)
        success_count = 0
        failed_count = 0
        
        print(f"\n开始爬取 {total} 个match_id的数据...")
        print("=" * 60)
        
        for idx, match_id in enumerate(match_ids, 1):
            output_file = os.path.join(output_dir, f'{match_id}_overunder_s2d_data.csv')

            print(f"\n[{idx}/{total}] 正在处理 match_id: {match_id}")
            
            # 访问页面
            if not self.fetch_page(match_id):
                failed_count += 1
                time.sleep(delay)
                continue
            
            # 解析初盘/终盘表格
            main_data = self.parse_main_table(match_id)
            if main_data:
                fieldnames = ['match_id', '公司名', '初盘-大球', '初盘-盘', '初盘-小球', '终盘-大球', '终盘-盘', '终盘-小球']
                with open(output_file, 'w', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(main_data)
                
            
            # 解析详细记录表格
            '''
            detail_data = self.parse_detail_table(match_id)
            if detail_data:
                self.save_detail_csv(detail_data, detail_output)
                print(f"  ✓ 详细记录: 解析到 {len(detail_data)} 条记录")
            else:
                print(f"  ⚠ 详细记录: 无数据")
            '''
            
            success_count += 1
            
            # 请求延迟
            if idx < total:
                time.sleep(delay)
        
        print("\n" + "=" * 60)
        print(f"爬取完成！")
        print(f"  总数: {total}")
        print(f"  成功: {success_count}")
        print(f"  失败: {failed_count}")
        print(f"\n输出文件:")
    
    def close(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()
            print("\n✓ 浏览器已关闭")


def main():
    """主函数：命令行交互界面"""
    import argparse

    parser = argparse.ArgumentParser(
        description='亚洲大小球盘赔率爬虫工具 (初盘终盘)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python b3_win007_ovlive_scraper.py --csv matches.csv --output-dir data/win007
  python b3_win007_ovlive_scraper.py --csv matches.csv --output-dir data/win007 --delay 3 --visible
        """
    )

    parser.add_argument(
        '--csv',
        required=True,
        help='输入CSV文件路径（包含match_id列）'
    )

    parser.add_argument(
        '--output-dir',
        default='data/win007',
        help='输出目录路径（默认: data/win007）'
    )

    parser.add_argument(
        '-d', '--delay',
        type=float,
        default=2.0,
        help='请求延迟时间（秒，默认: 2.0）'
    )

    parser.add_argument(
        '-t', '--timeout',
        type=int,
        default=20,
        help='页面加载超时时间（秒，默认: 20）'
    )

    parser.add_argument(
        '--visible',
        action='store_true',
        help='显示浏览器窗口（默认: 无头模式）'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='限制爬取数量（用于测试）'
    )

    args = parser.parse_args()

    # 打印配置信息
    print("=" * 60)
    print("亚洲大小球盘赔率爬虫工具 (初盘终盘)")
    print("=" * 60)
    print(f"输入文件: {args.csv}")
    print(f"输出目录: {args.output_dir}")
    print(f"请求延迟: {args.delay}秒")
    print(f"超时时间: {args.timeout}秒")
    print(f"浏览器模式: {'可见' if args.visible else '无头'}")
    if args.limit:
        print(f"爬取限制: {args.limit}个")
    print("=" * 60)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化爬虫
    scraper = OverUnderScraper(
        headless=not args.visible,
        timeout=args.timeout
    )

    try:
        # 加载match_id列表
        match_ids = scraper.load_match_ids(args.csv)

        # 限制数量（如果指定）
        if args.limit and args.limit > 0:
            match_ids = match_ids[:args.limit]
            print(f"✓ 已限制爬取数量为 {len(match_ids)} 个")

        # 开始爬取
        scraper.scrape(
            match_ids=match_ids,
            output_dir=args.output_dir,
            delay=args.delay
        )

    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断操作")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close()


if __name__ == '__main__':
    main()
