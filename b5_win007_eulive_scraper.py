#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
欧洲赔率滚球数据爬虫工具
支持批量抓取titan007.com的欧赔数据并输出为CSV
"""

import time
import csv
import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EuroOddsScraper:
    """欧洲赔率数据爬虫类"""
    
    def __init__(self, headless=True, wait_timeout=15):
        """
        初始化爬虫
        
        Args:
            headless: 是否使用无头模式
            wait_timeout: 页面加载超时时间(秒)
        """
        self.headless = headless
        self.wait_timeout = wait_timeout
        self.driver = None
        
    def setup_driver(self):
        """设置Chrome驱动"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        # 反爬虫设置
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        # 设置User-Agent
        chrome_options.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
        
        # 禁用自动化标识
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            # 执行CDP命令,隐藏webdriver特征
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                '''
            })
            logger.info("Chrome驱动初始化成功")
        except Exception as e:
            logger.error(f"Chrome驱动初始化失败: {e}")
            raise
            
    def close_driver(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()
            logger.info("浏览器已关闭")
    
    def fetch_page(self, url: str, retry_times=3) -> bool:
        """
        获取页面
        
        Args:
            url: 目标URL
            retry_times: 重试次数
            
        Returns:
            是否成功加载页面
        """
        for attempt in range(retry_times):
            try:
                logger.info(f"正在访问: {url} (尝试 {attempt + 1}/{retry_times})")
                self.driver.get(url)
                
                # 等待表格加载
                WebDriverWait(self.driver, self.wait_timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
                
                # 随机延迟,模拟人类行为
                time.sleep(2 + attempt * 0.5)
                return True
                
            except TimeoutException:
                logger.warning(f"页面加载超时 (尝试 {attempt + 1}/{retry_times})")
                if attempt < retry_times - 1:
                    time.sleep(3)
                    continue
            except Exception as e:
                logger.error(f"访问页面出错: {e}")
                if attempt < retry_times - 1:
                    time.sleep(3)
                    continue
        
        return False
    
    def parse_table_data(self, match_id, company_id) -> List[Dict[str, str]]:
        """
        解析页面中的欧赔表格数据
        
        Returns:
            包含所有行数据的列表
        """
        data_rows = []
        
        try:
            # 查找所有表格
            tables = self.driver.find_elements(By.TAG_NAME, "table")
            
            if not tables:
                logger.warning("未找到表格元素")
                return data_rows
            
            # 通常第一个表格是我们需要的数据表格
            target_table = tables[0]
            
            # 查找所有行(跳过表头)
            rows = target_table.find_elements(By.TAG_NAME, "tr")
            
            logger.info(f"找到 {len(rows)} 行数据(包含表头)")
            
            for i, row in enumerate(rows):
                # 跳过表头行
                if i == 0:
                    continue
                    
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    # 确保至少有7列数据
                    if len(cells) < 7:
                        continue
                    
                    # 提取各列数据
                    row_data = {
                        'match_id': match_id,
                        'company_id': company_id,
                        '时间': cells[0].text.strip(),
                        '比分': cells[1].text.strip(),
                        '主队胜': cells[2].text.strip(),
                        '和局': cells[3].text.strip(),
                        '客队胜': cells[4].text.strip(),
                        '变化时间': cells[5].text.strip(),
                        '状态': cells[6].text.strip()
                    }
                    
                    # 过滤空行
                    if any(row_data.values()):
                        data_rows.append(row_data)
                        
                except Exception as e:
                    logger.warning(f"解析第 {i} 行数据失败: {e}")
                    continue
            
            logger.info(f"成功解析 {len(data_rows)} 条有效数据")
            
        except NoSuchElementException:
            logger.error("未找到表格元素")
        except Exception as e:
            logger.error(f"解析表格数据出错: {e}")
        
        return data_rows
    
    def save_to_csv(self, data: List[Dict[str, str]], filepath: str):
        """
        保存数据到CSV文件
        
        Args:
            data: 数据列表
            filepath: 输出文件路径
        """
        if not data:
            logger.warning(f"没有数据可保存: {filepath}")
            return
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 写入CSV
            with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
                fieldnames = ['match_id', 'company_id', '时间', '比分', '主队胜', '和局', '客队胜', '变化时间', '状态']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"数据已保存到: {filepath} ({len(data)} 条记录)")
            
        except Exception as e:
            logger.error(f"保存CSV文件失败: {e}")
    
    def scrape_match(self, match_id: str, company_id: str, output_dir: str) -> bool:
        """
        抓取单个比赛的欧赔数据
        
        Args:
            match_id: 比赛ID
            company_id: 公司ID
            output_dir: 输出目录
            
        Returns:
            是否成功
        """
        url = f"https://vip.titan007.com/changeDetail/1x2.aspx?id={match_id}&companyid={company_id}&l=0"
        
        logger.info(f"开始抓取 match_id={match_id}, company_id={company_id}")
        
        # 获取页面
        if not self.fetch_page(url):
            logger.error(f"无法加载页面: match_id={match_id}, company_id={company_id}")
            return False
        
        # 解析数据
        data = self.parse_table_data(match_id, company_id)
        
        if not data:
            logger.warning(f"未获取到数据: match_id={match_id}, company_id={company_id}")
            return False
        
        # 保存数据
        filename = f"{match_id}_euro1x2_live_data_cp{company_id}.csv"
        filepath = os.path.join(output_dir, filename)
        self.save_to_csv(data, filepath)
        
        return True


def load_match_ids(csv_file: str) -> List[str]:
    """
    从CSV文件加载match_id列表
    
    Args:
        csv_file: CSV文件路径
        
    Returns:
        match_id列表
    """
    match_ids = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'match_id' in row and row['match_id']:
                    match_ids.append(row['match_id'].strip())
        
        logger.info(f"从 {csv_file} 加载了 {len(match_ids)} 个match_id")
        
    except Exception as e:
        logger.error(f"读取CSV文件失败: {e}")
        raise
    
    return match_ids


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='欧洲赔率滚球数据批量爬虫工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 基本用法
  python b5_win007_eulive_scraper.py --csv matches.csv --output-dir data/win007

  # 指定公司ID
  python b5_win007_eulive_scraper.py --csv matches.csv --output-dir data/win007 -c 3,8

  # 限制抓取数量
  python b5_win007_eulive_scraper.py --csv matches.csv --output-dir data/win007 -l 10
        '''
    )

    parser.add_argument(
        '--csv',
        required=True,
        help='输入CSV文件路径(包含match_id列)'
    )

    parser.add_argument(
        '--output-dir',
        default='data/win007',
        help='输出目录路径(默认: data/win007)'
    )

    parser.add_argument(
        '-c', '--companies',
        default='3,8',
        help='公司ID列表,逗号分隔(默认: 3,8)'
    )

    parser.add_argument(
        '-l', '--limit',
        type=int,
        default=None,
        help='限制处理的match_id数量(默认: 全部)'
    )

    parser.add_argument(
        '--no-headless',
        action='store_true',
        help='显示浏览器窗口(默认为无头模式)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=15,
        help='页面加载超时时间,秒(默认: 15)'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=4.0,
        help='每次请求之间的延迟时间,秒(默认: 1)'
    )

    args = parser.parse_args()

    # 解析公司ID列表
    company_ids = [cid.strip() for cid in args.companies.split(',')]

    # 加载match_id列表
    logger.info("=" * 60)
    logger.info("欧洲赔率滚球数据爬虫启动")
    logger.info("=" * 60)

    try:
        match_ids = load_match_ids(args.csv)
    except Exception as e:
        logger.error(f"无法加载match_id: {e}")
        return

    # 应用数量限制
    if args.limit and args.limit > 0:
        match_ids = match_ids[:args.limit]
        logger.info(f"限制处理数量为: {args.limit}")

    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 初始化爬虫
    scraper = EuroOddsScraper(
        headless=not args.no_headless,
        wait_timeout=args.timeout
    )
    
    try:
        scraper.setup_driver()
        
        # 统计信息
        total_tasks = len(match_ids) * len(company_ids)
        success_count = 0
        fail_count = 0
        
        logger.info(f"开始处理: {len(match_ids)} 个match_id × {len(company_ids)} 个company_id = {total_tasks} 个任务")
        logger.info(f"公司ID列表: {company_ids}")
        logger.info("-" * 60)
        
        start_time = time.time()
        
        # 遍历所有match_id和company_id组合
        for idx, match_id in enumerate(match_ids, 1):
            for company_id in company_ids:
                task_num = (idx - 1) * len(company_ids) + company_ids.index(company_id) + 1
                
                logger.info(f"\n[{task_num}/{total_tasks}] 处理中...")
                
                # 抓取数据
                success = scraper.scrape_match(match_id, company_id, output_dir)
                
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                
                # 延迟,避免请求过快
                if task_num < total_tasks:
                    time.sleep(args.delay)
        
        # 最终统计
        elapsed_time = time.time() - start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("抓取完成!")
        logger.info(f"总任务数: {total_tasks}")
        logger.info(f"成功: {success_count}")
        logger.info(f"失败: {fail_count}")
        logger.info(f"耗时: {elapsed_time:.2f} 秒")
        logger.info(f"输出目录: {output_dir}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("\n用户中断操作")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
    finally:
        scraper.close_driver()


if __name__ == '__main__':
    main()
