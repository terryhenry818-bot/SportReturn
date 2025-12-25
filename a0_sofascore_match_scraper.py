#!/usr/bin/env python3
"""
SofaScore Football Match Scraper
Scrapes daily football match data from SofaScore using Selenium with headless Chrome.
"""

import os
import re
import csv
import json
import time
import argparse
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class SofaScoreScraper:
    """Scraper for SofaScore football match data."""

    BASE_URL = "https://www.sofascore.com/football"

    def __init__(self, headless=True, wait_timeout=20):
        """
        Initialize the scraper with Chrome WebDriver.

        Args:
            headless: Whether to run Chrome in headless mode
            wait_timeout: Maximum wait time for page elements (seconds)
        """
        self.wait_timeout = wait_timeout
        self.driver = self._init_driver(headless)
        self.matches = []

    def _init_driver(self, headless):
        """Initialize Chrome WebDriver with appropriate options."""
        chrome_options = Options()

        if headless:
            chrome_options.add_argument("--headless=new")

        # Common options for stability
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        # Disable images and CSS for faster loading
        prefs = {
            "profile.managed_default_content_settings.images": 2,
        }
        chrome_options.add_experimental_option("prefs", prefs)

        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)

        return driver

    def _get_weekday_chinese(self, date_str):
        """Convert date string to Chinese weekday."""
        weekdays = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday"
        }
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return weekdays[date_obj.weekday()]

    def _parse_match_data_from_api(self, date_str):
        """
        Parse match data by intercepting API calls.
        SofaScore loads data via API, so we extract from network requests.
        """
        matches = []

        try:
            # Execute JavaScript to get match data from the page's React state or API
            script = """
            return new Promise((resolve) => {
                // Try to find match data in the page
                const matchElements = document.querySelectorAll('[data-testid="event_cell"]');
                if (matchElements.length > 0) {
                    resolve(matchElements.length);
                } else {
                    resolve(0);
                }
            });
            """

            # Wait for the page to fully load
            time.sleep(3)

            # Try to extract match data from the page
            matches = self._extract_matches_from_page(date_str)

        except Exception as e:
            print(f"Error parsing match data: {e}")

        return matches

    def _extract_matches_from_page(self, date_str):
        """Extract match information from the loaded page."""
        matches = []
        weekday = self._get_weekday_chinese(date_str)

        try:
            # Find all match event elements
            # SofaScore uses various selectors for match events
            event_selectors = [
                "a[data-testid='event_cell']",
                "div[class*='event']",
                "a[href*='/football/']"
            ]

            match_elements = []
            for selector in event_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        match_elements = elements
                        break
                except:
                    continue

            # If standard selectors don't work, try getting data from script tags
            if not match_elements:
                matches = self._extract_from_script_data(date_str, weekday)
                return matches

            print(f"  Found {len(match_elements)} match elements on page")

            for elem in match_elements:
                try:
                    match_data = self._parse_match_element(elem, date_str, weekday)
                    if match_data:
                        matches.append(match_data)
                except Exception as e:
                    continue

        except Exception as e:
            print(f"  Error extracting matches: {e}")
            # Fallback to API extraction
            matches = self._extract_from_api(date_str, weekday)

        return matches

    def _extract_from_script_data(self, date_str, weekday):
        """Extract match data from embedded script tags or window objects."""
        matches = []

        try:
            # Try to get data from __NEXT_DATA__ or similar
            script = """
            // Try to find React state data
            if (window.__NEXT_DATA__) {
                return JSON.stringify(window.__NEXT_DATA__);
            }

            // Try to find embedded JSON data
            const scripts = document.querySelectorAll('script[type="application/json"]');
            for (let script of scripts) {
                try {
                    return script.textContent;
                } catch(e) {}
            }

            return null;
            """

            data = self.driver.execute_script(script)
            if data:
                json_data = json.loads(data)
                # Parse the JSON structure to extract matches
                matches = self._parse_json_match_data(json_data, date_str, weekday)

        except Exception as e:
            print(f"  Could not extract from script data: {e}")

        return matches

    def _extract_from_api(self, date_str, weekday):
        """Extract match data by calling SofaScore API directly."""
        matches = []

        try:
            # SofaScore API endpoint for scheduled events
            api_url = f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{date_str}"

            script = f"""
            return fetch('{api_url}')
                .then(response => response.json())
                .catch(error => null);
            """

            data = self.driver.execute_script(script)

            if data and 'events' in data:
                for event in data['events']:
                    match = self._parse_api_event(event, date_str, weekday)
                    if match:
                        matches.append(match)

        except Exception as e:
            print(f"  API extraction failed: {e}")

        return matches

    def _parse_api_event(self, event, date_str, weekday):
        """Parse a single event from API response."""
        try:
            # Check if match is finished
            status = event.get('status', {})
            status_type = status.get('type', '')

            if status_type != 'finished':
                return None

            # Extract basic info
            match_id = event.get('id', '')

            # Teams
            home_team_data = event.get('homeTeam', {})
            away_team_data = event.get('awayTeam', {})

            home_team = home_team_data.get('name', '')
            away_team = away_team_data.get('name', '')
            home_team_id = home_team_data.get('id', '')
            away_team_id = away_team_data.get('id', '')

            # Score
            home_score = event.get('homeScore', {})
            away_score = event.get('awayScore', {})

            home_goals = home_score.get('current', 0)
            away_goals = away_score.get('current', 0)
            home_ht = home_score.get('period1', 0)
            away_ht = away_score.get('period1', 0)

            # Tournament/Competition
            tournament = event.get('tournament', {})
            competition = tournament.get('name', '')

            # Season
            season_data = event.get('season', {})
            season = season_data.get('name', season_data.get('year', ''))

            # Round
            round_info = event.get('roundInfo', {})
            round_num = round_info.get('round', '')

            # Time - 从时间戳提取实际日期和时间
            start_timestamp = event.get('startTimestamp', 0)
            if start_timestamp:
                match_time = datetime.fromtimestamp(start_timestamp)
                time_str = match_time.strftime("%H:%M")
                actual_date = match_time.strftime("%Y-%m-%d")  # 提取实际比赛日期
            else:
                time_str = ""
                actual_date = date_str  # 无时间戳时回退使用查询日期

            # Build match URL
            # Format: https://www.sofascore.com/football/match/{slug}/{customId}#id:{match_id}
            slug = event.get('slug', '')
            custom_id = event.get('customId', '')
            if slug and custom_id:
                match_url = f"https://www.sofascore.com/football/match/{slug}/{custom_id}#id:{match_id}"
            elif slug:
                match_url = f"https://www.sofascore.com/football/match/{slug}#id:{match_id}"
            else:
                match_url = ""

            # 根据实际日期计算星期
            actual_weekday = self._get_weekday_chinese(actual_date)

            return {
                'match_id': match_id,
                'date': actual_date,
                'time': time_str,
                'weekday': actual_weekday,
                'competition': competition,
                'season': season,
                'round': round_num,
                'venue': '',  # Will be determined based on team perspective
                'opponent': '',  # Will be determined based on team perspective
                'home_team': home_team,
                'away_team': away_team,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'home_ht': home_ht,
                'away_ht': away_ht,
                'team_goals': home_goals,  # Default to home team perspective
                'opponent_goals': away_goals,
                'result': self._determine_result(home_goals, away_goals),
                'status': 'finished',
                'match_url': match_url
            }

        except Exception as e:
            print(f"  Error parsing API event: {e}")
            return None

    def _parse_match_element(self, elem, date_str, weekday):
        """Parse a single match element from the page."""
        try:
            # Get href for match URL and ID
            href = elem.get_attribute('href') or ''

            # Extract match ID from URL
            match_id = ''
            if '#id:' in href:
                match_id = href.split('#id:')[-1]
            elif '/id:' in href:
                match_id = href.split('/id:')[-1]

            # Get text content
            text = elem.text.strip()

            if not text:
                return None

            # Try to parse team names and score
            # Format is usually: "Team1 score1 - score2 Team2" or similar
            lines = text.split('\n')

            if len(lines) < 2:
                return None

            # Basic structure parsing
            home_team = lines[0].strip() if len(lines) > 0 else ''
            away_team = lines[-1].strip() if len(lines) > 1 else ''

            # Try to find score
            score_pattern = r'(\d+)\s*[-:]\s*(\d+)'
            score_match = re.search(score_pattern, text)

            home_goals = 0
            away_goals = 0
            if score_match:
                home_goals = int(score_match.group(1))
                away_goals = int(score_match.group(2))

            return {
                'match_id': match_id,
                'date': date_str,
                'time': '',
                'weekday': weekday,
                'competition': '',
                'season': '',
                'round': '',
                'venue': '',
                'opponent': '',
                'home_team': home_team,
                'away_team': away_team,
                'home_team_id': '',
                'away_team_id': '',
                'home_goals': home_goals,
                'away_goals': away_goals,
                'home_ht': '',
                'away_ht': '',
                'team_goals': home_goals,
                'opponent_goals': away_goals,
                'result': self._determine_result(home_goals, away_goals),
                'status': 'finished',
                'match_url': href
            }

        except Exception as e:
            return None

    def _parse_json_match_data(self, json_data, date_str, weekday):
        """Parse match data from JSON structure."""
        matches = []

        def find_events(obj, events_list):
            """Recursively find events in JSON structure."""
            if isinstance(obj, dict):
                if 'events' in obj and isinstance(obj['events'], list):
                    events_list.extend(obj['events'])
                for value in obj.values():
                    find_events(value, events_list)
            elif isinstance(obj, list):
                for item in obj:
                    find_events(item, events_list)

        events = []
        find_events(json_data, events)

        for event in events:
            match = self._parse_api_event(event, date_str, weekday)
            if match:
                matches.append(match)

        return matches

    def _determine_result(self, home_goals, away_goals):
        """Determine match result from home team perspective."""
        try:
            home = int(home_goals) if home_goals else 0
            away = int(away_goals) if away_goals else 0

            if home > away:
                return "胜"
            elif home < away:
                return "负"
            else:
                return "平"
        except:
            return ""

    def scrape_date(self, date_str):
        """
        Scrape all finished football matches for a specific date.

        Args:
            date_str: Date in YYYY-MM-DD format

        Returns:
            List of match dictionaries
        """
        url = f"{self.BASE_URL}/{date_str}"
        print(f"Scraping: {url}")

        try:
            self.driver.get(url)

            # Wait for page to load
            time.sleep(5)

            # Try multiple methods to extract data
            matches = []

            # Method 1: Try API extraction first (most reliable)
            matches = self._extract_from_api(date_str, self._get_weekday_chinese(date_str))

            if not matches:
                # Method 2: Try page parsing
                matches = self._extract_matches_from_page(date_str)

            print(f"  Found {len(matches)} finished matches")
            return matches

        except TimeoutException:
            print(f"  Timeout loading page for {date_str}")
            return []
        except Exception as e:
            print(f"  Error scraping {date_str}: {e}")
            return []

    def scrape_date_range(self, start_date, end_date):
        """
        Scrape matches for a range of dates.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of all match dictionaries
        """
        all_matches = []

        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            matches = self.scrape_date(date_str)
            all_matches.extend(matches)

            # Add delay between requests to be respectful
            time.sleep(2)

            current += timedelta(days=1)

        self.matches = all_matches
        return all_matches

    def load_team_ids_from_csv(self, team_list_file):
        """
        Load team IDs from a CSV file.

        Args:
            team_list_file: Path to CSV file containing team IDs

        Returns:
            Set of team IDs (as strings for comparison)
        """
        team_ids = set()

        try:
            with open(team_list_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try common column names for team ID
                    team_id = row.get('sofascore_team_id') or row.get('id') or row.get('teamId')
                    if team_id:
                        team_ids.add(str(team_id).strip())

            print(f"Loaded {len(team_ids)} team IDs from {team_list_file}")

        except FileNotFoundError:
            print(f"Warning: Team list file not found: {team_list_file}")
        except Exception as e:
            print(f"Error loading team list: {e}")

        return team_ids

    def filter_matches_by_teams(self, team_ids):
        """
        Filter matches where home_team_id or away_team_id is in the team ID set.

        Args:
            team_ids: Set of team IDs to filter by

        Returns:
            List of filtered matches
        """
        if not team_ids:
            return []

        filtered = []
        for match in self.matches:
            home_id = str(match.get('home_team_id', '')).strip()
            away_id = str(match.get('away_team_id', '')).strip()

            if home_id in team_ids or away_id in team_ids:
                filtered.append(match)

        return filtered

    def _write_csv(self, matches, filename):
        """
        Write matches to a CSV file.

        Args:
            matches: List of match dictionaries
            filename: Output CSV filename
        """
        # Define CSV columns matching the original format
        fieldnames = [
            'match_id', 'date', 'time', 'weekday', 'competition', 'season',
            'round', 'venue', 'opponent', 'home_team', 'away_team',
            'home_team_id', 'away_team_id',
            'home_goals', 'away_goals', 'home_ht', 'away_ht',
            'team_goals', 'opponent_goals', 'result', 'status', 'match_url'
        ]

        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for match in matches:
                # Ensure all fields exist
                row = {field: match.get(field, '') for field in fieldnames}
                writer.writerow(row)

        return len(matches)

    def save_to_csv(self, all_matches_file="sofascore_all_matches.csv",
                    top5_matches_file=None, team_list_file=None):
        """
        Save scraped matches to CSV files.

        Args:
            all_matches_file: Output CSV filename for all matches
            top5_matches_file: Output CSV filename for top 5 team matches (optional)
            team_list_file: Path to CSV file containing team IDs for filtering (optional)
        """
        if not self.matches:
            print("No matches to save")
            return

        # Save all matches
        all_count = self._write_csv(self.matches, all_matches_file)
        print(f"Saved {all_count} matches to {all_matches_file}")

        # Save top 5 team matches if team list file is provided
        if top5_matches_file and team_list_file:
            team_ids = self.load_team_ids_from_csv(team_list_file)
            if team_ids:
                filtered_matches = self.filter_matches_by_teams(team_ids)
                top5_count = self._write_csv(filtered_matches, top5_matches_file)
                print(f"Saved {top5_count} top 5 team matches to {top5_matches_file}")

    def close(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Scrape SofaScore football matches')
    parser.add_argument('--start-date', '-s', default='2025-12-05',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', '-e', default='2025-12-24',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', default='sofascore_all_matches.csv',
                        help='Output CSV filename for all matches')
    parser.add_argument('--output-top5', '-t', default='sofascore_top5_teams_matches.csv',
                        help='Output CSV filename for top 5 team matches')
    parser.add_argument('--team-list', '-l', default='a0_sofascore_and_win007_teams.csv',
                        help='CSV file containing team IDs for filtering')
    parser.add_argument('--no-headless', action='store_true',
                        help='Run Chrome in visible mode (not headless)')

    args = parser.parse_args()

    print(f"SofaScore Football Match Scraper")
    print(f"================================")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output files:")
    print(f"  - All matches: {args.output}")
    print(f"  - Top 5 team matches: {args.output_top5}")
    print(f"  - Team list file: {args.team_list}")
    print()

    scraper = SofaScoreScraper(headless=not args.no_headless)

    try:
        scraper.scrape_date_range(args.start_date, args.end_date)
        scraper.save_to_csv(args.output, args.output_top5, args.team_list)
    finally:
        scraper.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
