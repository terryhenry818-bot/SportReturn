#!/bin/bash
# =============================================================================
# 未来比赛预测流水线
# =============================================================================
# 用法:
#   ./run_prediction_pipeline.sh 20251227 20251228
#   ./run_prediction_pipeline.sh 20251227 20251228 --skip-scrape  # 跳过抓取，只做预测
# =============================================================================

set -e  # 出错即停止

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <开始日期> <结束日期> [选项]"
    echo "  日期格式: YYYYMMDD (如 20251227)"
    echo "  选项:"
    echo "    --skip-scrape    跳过抓取步骤"
    echo ""
    echo "示例:"
    echo "  $0 20251227 20251228"
    echo "  $0 20251227 20251228 --skip-scrape"
    exit 1
fi

START_DATE=$1
END_DATE=$2
SKIP_SCRAPE=false

# 解析选项
for arg in "${@:3}"; do
    case $arg in
        --skip-scrape)
            SKIP_SCRAPE=true
            ;;
    esac
done

# 目录设置
DATA_DIR="data"
MATCHES_DIR="${DATA_DIR}/matches"
SOFASCORE_DIR="${DATA_DIR}/sofascore"
WIN007_DIR="${DATA_DIR}/win007"
OUTPUT_DIR="output"

# 文件名
SOFASCORE_ALL="${MATCHES_DIR}/sofascore_${START_DATE}_${END_DATE}.csv"
SOFASCORE_TOP5="${MATCHES_DIR}/sofascore_${START_DATE}_${END_DATE}_top5.csv"
WIN007_ALL="${MATCHES_DIR}/win007_${START_DATE}_${END_DATE}.csv"
WIN007_TOP5="${MATCHES_DIR}/win007_${START_DATE}_${END_DATE}_top5.csv"
MAPPED_MATCHES="${MATCHES_DIR}/mapped_matches_${START_DATE}_${END_DATE}.csv"
UPCOMING_WIDE_TABLE="upcoming_wide_table.csv"

# 创建目录
mkdir -p "${MATCHES_DIR}" "${SOFASCORE_DIR}" "${WIN007_DIR}" "${OUTPUT_DIR}"

echo "=============================================="
echo "未来比赛预测流水线"
echo "=============================================="
echo "日期范围: ${START_DATE} - ${END_DATE}"
echo "跳过抓取: ${SKIP_SCRAPE}"
echo "=============================================="

if [ "$SKIP_SCRAPE" = false ]; then

    # =============================================================================
    # Step 0: SofaScore比赛列表抓取
    # =============================================================================
    info "Step 0: 抓取SofaScore比赛列表..."

    # 转换日期格式: YYYYMMDD -> YYYY-MM-DD
    START_DATE_FMT="${START_DATE:0:4}-${START_DATE:4:2}-${START_DATE:6:2}"
    END_DATE_FMT="${END_DATE:0:4}-${END_DATE:4:2}-${END_DATE:6:2}"

    python3 a0_sofascore_match_scraper.py \
        --start-date "${START_DATE_FMT}" \
        --end-date "${END_DATE_FMT}" \
        --output "${SOFASCORE_ALL}" \
        --output-top5 "${SOFASCORE_TOP5}" \
        --team-list "a0_sofascore_and_win007_teams.csv"

    success "SofaScore比赛列表抓取完成"
    echo "  全量: ${SOFASCORE_ALL}"
    echo "  五大联赛: ${SOFASCORE_TOP5}"

    # =============================================================================
    # Step 1: Win007比赛列表抓取
    # =============================================================================
    info "Step 1: 抓取Win007比赛列表..."

    python3 a0_win007_match_scraper.py \
        --start "${START_DATE}" \
        --end "${END_DATE}" \
        --output "${WIN007_ALL}"

    success "Win007比赛列表抓取完成"
    echo "  全量: ${WIN007_ALL}"
    echo "  五大联赛: ${WIN007_TOP5}"

    # =============================================================================
    # Step 2: 比赛映射
    # =============================================================================
    info "Step 2: 映射SofaScore和Win007比赛..."

    python3 a0_zmapping_matches.py \
        --sofascore-csv "${SOFASCORE_TOP5}" \
        --win007-csv "${WIN007_TOP5}" \
        --mapping-csv "a0_sofascore_and_win007_teams.csv" \
        --output "${MAPPED_MATCHES}"

    success "比赛映射完成: ${MAPPED_MATCHES}"

    # =============================================================================
    # Step 3: SofaScore API数据抓取 (未来比赛跳过，因为没有比赛数据)
    # =============================================================================
    info "Step 3: 跳过SofaScore API数据抓取 (未来比赛无比赛统计数据)"

    # =============================================================================
    # Step 4: Win007亚盘数据抓取 (让球+大小球)
    # =============================================================================
    info "Step 4: 抓取Win007亚盘数据 (让球+大小球)..."

    python3 b1_win007_aslive_scraper.py \
        --csv "${WIN007_TOP5}" \
        --type all \
        --output-dir "${WIN007_DIR}"

    success "Win007亚盘数据抓取完成"

    # =============================================================================
    # Step 5: Win007欧赔初终盘数据
    # =============================================================================
    info "Step 5: 抓取Win007欧赔初终盘数据..."

    python3 b4_win007_euros2d_scraper.py \
        --csv "${WIN007_TOP5}" \
        --output-dir "${WIN007_DIR}"

    success "Win007欧赔初终盘数据抓取完成"

    # =============================================================================
    # Step 6: 特征工程 (构建 upcoming_wide_table)
    # =============================================================================
    info "Step 6: 执行特征工程 (构建 upcoming_wide_table)..."

    python3 c0_feature_engineering.py \
        --sofascore-dir "${SOFASCORE_DIR}" \
        --win007-dir "${WIN007_DIR}" \
        --matches-file "${MAPPED_MATCHES}" \
        --output-dir "${OUTPUT_DIR}"

    # 复制特征表到根目录作为 upcoming_wide_table.csv
    if [ -f "${OUTPUT_DIR}/inc_wide_table.csv" ]; then
        cp "${OUTPUT_DIR}/inc_wide_table.csv" "${UPCOMING_WIDE_TABLE}"
        success "特征工程完成: ${UPCOMING_WIDE_TABLE}"
    else
        warn "特征工程输出文件不存在: ${OUTPUT_DIR}/inc_wide_table.csv"
    fi

fi  # end if SKIP_SCRAPE

# =============================================================================
# Step 7: 运行预测脚本并发送邮件
# =============================================================================
info "Step 7: 运行预测脚本并发送邮件..."

python3 predict_upcoming_matches.py

success "预测完成"

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "=============================================="
success "预测流水线执行完成！"
echo "=============================================="
echo "输出文件:"
echo "  比赛映射: ${MAPPED_MATCHES}"
echo "  特征表: ${UPCOMING_WIDE_TABLE}"
echo "  所有预测: upcoming_predictions.csv"
echo "  高价值投注: upcoming_high_value_bets.csv"
echo "=============================================="
