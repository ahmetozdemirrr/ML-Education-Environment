#!/bin/bash

# ML Cache Tester - Bash Edition 🚀
# Test all ML models with cache system
#
# DEPENDENCIES (install before running):
#   sudo apt install curl jq bc
#   # or on macOS: brew install curl jq bc
#   # or on CentOS/RHEL: sudo yum install curl jq bc

set -euo pipefail

# Configuration
BASE_URL="http://localhost:8000"
MAX_PARALLEL=3
VERBOSE=true
LOG_FILE="ml_test_$(date +%Y%m%d_%H%M%S).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
SUCCESSFUL_TESTS=0
FAILED_TESTS=0
CACHE_HITS=0

# Arrays for results
declare -a TEST_RESULTS=()
declare -a TIMING_RESULTS=()

# Utility Functions
print_header() {
    echo -e "${BLUE}${BOLD}================================================================================================${NC}"
    echo -e "${WHITE}${BOLD} $1 ${NC}"
    echo -e "${BLUE}${BOLD}================================================================================================${NC}"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
    echo "[INFO] $1" >> "$LOG_FILE" 2>/dev/null || true
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    echo "[SUCCESS] $1" >> "$LOG_FILE" 2>/dev/null || true
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[WARN] $1" >> "$LOG_FILE" 2>/dev/null || true
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $1" >> "$LOG_FILE" 2>/dev/null || true
}

print_progress() {
    echo -e "${PURPLE}[PROGRESS]${NC} $1"
}

# Check dependencies
check_dependencies() {
    local deps=("curl" "jq")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            print_error "$dep is required but not installed"
            echo "Install with: sudo apt install $dep"
            exit 1
        fi
    done
    print_success "All dependencies available"
}

# Health check
health_check() {
    print_info "Checking backend health..."

    if curl -s -f "$BASE_URL/health" > /dev/null; then
        print_success "Backend is healthy"
        return 0
    else
        print_error "Backend is not accessible at $BASE_URL"
        return 1
    fi
}

# Get cache stats
get_cache_stats() {
    curl -s "$BASE_URL/cache-stats" | jq -r '.cache_stats'
}

# Clear cache
clear_cache() {
    print_info "Clearing cache..."

    local response
    response=$(curl -s -w "%{http_code}" -X DELETE "$BASE_URL/clear-cache")
    local status_code="${response: -3}"

    if [[ "$status_code" == "200" ]]; then
        print_success "Cache cleared successfully"
        return 0
    else
        print_error "Failed to clear cache (HTTP $status_code)"
        return 1
    fi
}

# Send HTTP request with timing
send_request() {
    local endpoint="$1"
    local payload="$2"
    local test_name="$3"

    print_info "Sending request to $endpoint..."

    local start_time=$(date +%s.%3N)

    local response
    local status_code

    # Add timeout and better error handling
    response=$(timeout 120 curl -s -w "%{http_code}" \
        --connect-timeout 30 \
        --max-time 120 \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$BASE_URL$endpoint" 2>/dev/null || echo "000TIMEOUT")

    local end_time=$(date +%s.%3N)

    # Handle timeout
    if [[ "$response" == "000TIMEOUT" ]]; then
        print_error "Request timed out after 120 seconds"
        echo "408|{\"error\": \"timeout\"}|120.0"
        return
    fi

    # Calculate elapsed time (use awk instead of bc for better compatibility)
    local elapsed
    elapsed=$(awk "BEGIN {printf \"%.3f\", $end_time - $start_time}")

    status_code="${response: -3}"
    local body="${response%???}"

    print_info "Response received: HTTP $status_code in ${elapsed}s"

    # Store timing result
    TIMING_RESULTS+=("$test_name:$elapsed:$status_code")

    echo "$status_code|$body|$elapsed"
}

# Test single configuration
test_single_config() {
    local algorithm="$1"
    local dataset="$2"
    local params="$3"
    local metrics="$4"
    local mode="$5"

    ((TOTAL_TESTS++))

    local config_id="${algorithm,,}_${dataset}_$(date +%s)_$"
    config_id=${config_id// /_}

    # Build JSON payload using simple template (avoid complex jq)
    local payload
    case "$algorithm" in
        "Decision Tree")
            payload="{
                \"algorithm\": \"Decision Tree\",
                \"dataset\": \"$dataset\",
                \"params\": {
                    \"Criterion\": \"Gini\",
                    \"Max Depth\": \"\",
                    \"Min Samples Split\": 2,
                    \"Min Samples Leaf\": 1,
                    \"selectedMetrics\": $metrics,
                    \"frontend_config_id\": \"$config_id\"
                },
                \"global_settings\": {
                    \"useCrossValidation\": false,
                    \"cvFolds\": 5,
                    \"useTrainTestSplit\": true,
                    \"testSplitRatio\": 0.2,
                    \"randomSeedType\": \"fixed\",
                    \"applyFeatureScaling\": true,
                    \"scalerType\": \"standard\",
                    \"randomSeed\": 42
                }
            }"
            ;;
        "Logistic Regression")
            payload="{
                \"algorithm\": \"Logistic Regression\",
                \"dataset\": \"$dataset\",
                \"params\": {
                    \"Penalty\": \"L2\",
                    \"C (Reg. Strength)\": 1.0,
                    \"Solver\": \"lbfgs\",
                    \"Max Iterations\": 100,
                    \"selectedMetrics\": $metrics,
                    \"frontend_config_id\": \"$config_id\"
                },
                \"global_settings\": {
                    \"useCrossValidation\": false,
                    \"cvFolds\": 5,
                    \"useTrainTestSplit\": true,
                    \"testSplitRatio\": 0.2,
                    \"randomSeedType\": \"fixed\",
                    \"applyFeatureScaling\": true,
                    \"scalerType\": \"standard\",
                    \"randomSeed\": 42
                }
            }"
            ;;
        "SVM")
            payload="{
                \"algorithm\": \"SVM\",
                \"dataset\": \"$dataset\",
                \"params\": {
                    \"C (Reg. Param)\": 1.0,
                    \"Kernel\": \"RBF\",
                    \"Gamma\": \"Scale\",
                    \"Degree (Poly Kernel)\": 3,
                    \"selectedMetrics\": $metrics,
                    \"frontend_config_id\": \"$config_id\"
                },
                \"global_settings\": {
                    \"useCrossValidation\": false,
                    \"cvFolds\": 5,
                    \"useTrainTestSplit\": true,
                    \"testSplitRatio\": 0.2,
                    \"randomSeedType\": \"fixed\",
                    \"applyFeatureScaling\": true,
                    \"scalerType\": \"standard\",
                    \"randomSeed\": 42
                }
            }"
            ;;
        "K-Nearest Neighbor")
            payload="{
                \"algorithm\": \"K-Nearest Neighbor\",
                \"dataset\": \"$dataset\",
                \"params\": {
                    \"N Neighbors\": 5,
                    \"Weights\": \"Uniform\",
                    \"Algorithm\": \"auto\",
                    \"Metric\": \"minkowski\",
                    \"selectedMetrics\": $metrics,
                    \"frontend_config_id\": \"$config_id\"
                },
                \"global_settings\": {
                    \"useCrossValidation\": false,
                    \"cvFolds\": 5,
                    \"useTrainTestSplit\": true,
                    \"testSplitRatio\": 0.2,
                    \"randomSeedType\": \"fixed\",
                    \"applyFeatureScaling\": true,
                    \"scalerType\": \"standard\",
                    \"randomSeed\": 42
                }
            }"
            ;;
        *)
            print_error "Unknown algorithm: $algorithm"
            return 1
            ;;
    esac

    local endpoint
    if [[ "$mode" == "train" ]]; then
        endpoint="/train"
    else
        endpoint="/evaluate"
    fi

    local test_name="$algorithm/$dataset/$mode"
    print_progress "Testing: $test_name with metrics: $(echo "$metrics" | jq -r 'join(", ")')"

    # Send request
    local result
    result=$(send_request "$endpoint" "$payload" "$test_name")

    IFS='|' read -r status_code body elapsed <<< "$result"

    if [[ "$status_code" == "200" ]]; then
        ((SUCCESSFUL_TESTS++))

        # Check if from cache
        local from_cache
        from_cache=$(echo "$body" | jq -r '.from_cache // false')

        if [[ "$from_cache" == "true" ]]; then
            ((CACHE_HITS++))
            print_success "$test_name completed in ${elapsed}s (${GREEN}CACHE HIT${NC})"
        else
            print_success "$test_name completed in ${elapsed}s (${RED}CACHE MISS${NC})"
        fi

        # Show detailed results if verbose
        if [[ "$VERBOSE" == true ]]; then
            if [[ "$mode" == "train" ]]; then
                local training_metrics
                training_metrics=$(echo "$body" | jq -r '.training_metrics')
                if [[ "$training_metrics" != "null" ]]; then
                    local fit_time memory throughput
                    fit_time=$(echo "$training_metrics" | jq -r '.fit_time_seconds // 0')
                    memory=$(echo "$training_metrics" | jq -r '.memory_usage_mb // 0')
                    throughput=$(echo "$training_metrics" | jq -r '.training_throughput_samples_per_sec // 0')

                    echo -e "    ${CYAN}Fit Time: ${fit_time}s | Memory: ${memory}MB | Throughput: ${throughput} samples/s${NC}"
                fi
            elif [[ "$mode" == "evaluate" ]]; then
                local eval_metrics
                eval_metrics=$(echo "$body" | jq -r '.metrics')
                if [[ "$eval_metrics" != "null" ]] && [[ "$eval_metrics" != "{}" ]]; then
                    echo -n -e "    ${CYAN}Metrics: "
                    echo "$eval_metrics" | jq -r 'to_entries[] | "\(.key): \(.value)"' | tr '\n' ' '
                    echo -e "${NC}"
                fi
            fi
        fi

        TEST_RESULTS+=("SUCCESS:$test_name:$elapsed:$from_cache")

    else
        ((FAILED_TESTS++))
        print_error "$test_name failed (HTTP $status_code)"

        if [[ "$VERBOSE" == true ]]; then
            echo "    Response: $(echo "$body" | head -c 200)..."
        fi

        TEST_RESULTS+=("FAILED:$test_name:$elapsed:false")
    fi
}

# Run cache hit/miss test
test_cache_functionality() {
    print_header "🔄 CACHE HIT/MISS TEST"

    local algorithm="Decision Tree"
    local dataset="iris"
    local params='{"Criterion": "Gini", "Max Depth": "", "Min Samples Split": 2, "Min Samples Leaf": 1}'
    local metrics='["Accuracy"]'

    print_info "First request (expecting CACHE MISS):"
    test_single_config "$algorithm" "$dataset" "$params" "$metrics" "train"

    sleep 1

    print_info "Second request (expecting CACHE HIT):"
    test_single_config "$algorithm" "$dataset" "$params" "$metrics" "train"

    # Calculate speedup
    local first_time second_time
    first_time=$(echo "${TIMING_RESULTS[-2]}" | cut -d: -f2)
    second_time=$(echo "${TIMING_RESULTS[-1]}" | cut -d: -f2)

    # Calculate speedup using awk instead of bc
    if [[ -n "$first_time" ]] && [[ -n "$second_time" ]] && [[ "$first_time" != "0" ]] && [[ "$second_time" != "0" ]]; then
        local speedup
        speedup=$(awk "BEGIN {printf \"%.1f\", $first_time / $second_time}")
        print_success "Cache speedup: ${speedup}x (${first_time}s → ${second_time}s)"
    fi
}

# Run comprehensive test
test_all_models() {
    print_header "🧪 COMPREHENSIVE MODEL TEST"

    # Define model configurations
    declare -A MODEL_CONFIGS=(
        ["Decision Tree"]='{"Criterion": "Gini", "Max Depth": "", "Min Samples Split": 2, "Min Samples Leaf": 1}'
        ["Logistic Regression"]='{"Penalty": "L2", "C (Reg. Strength)": 1.0, "Solver": "lbfgs", "Max Iterations": 100}'
        ["SVM"]='{"C (Reg. Param)": 1.0, "Kernel": "RBF", "Gamma": "Scale", "Degree (Poly Kernel)": 3}'
        ["K-Nearest Neighbor"]='{"N Neighbors": 5, "Weights": "Uniform", "Algorithm": "auto", "Metric": "minkowski"}'
    )

    # Define datasets (fast ones for testing)
    local datasets=("iris" "two_moons_data" "wine_data")

    # Define metric combinations
    local metric_combinations=(
        '["Accuracy"]'
        '["Accuracy", "Precision"]'
        '["Accuracy", "F1-Score"]'
        '["Accuracy", "Precision", "Recall", "F1-Score"]'
    )

    local total_combinations=$((${#MODEL_CONFIGS[@]} * ${#datasets[@]} * ${#metric_combinations[@]} * 2))
    print_info "Total test combinations: $total_combinations"

    local current=0

    # Test all combinations
    for algorithm in "${!MODEL_CONFIGS[@]}"; do
        for dataset in "${datasets[@]}"; do
            for metrics in "${metric_combinations[@]}"; do
                # Test training
                ((current++))
                echo -e "${PURPLE}Progress: $current/$total_combinations${NC}"
                test_single_config "$algorithm" "$dataset" "${MODEL_CONFIGS[$algorithm]}" "$metrics" "train"

                # Test evaluation
                ((current++))
                echo -e "${PURPLE}Progress: $current/$total_combinations${NC}"
                test_single_config "$algorithm" "$dataset" "${MODEL_CONFIGS[$algorithm]}" "$metrics" "evaluate"

                # Small delay to avoid overwhelming
                sleep 0.1
            done
        done
    done
}

# Parallel test runner (optional)
test_parallel() {
    print_header "⚡ PARALLEL TEST EXECUTION"
    print_warning "Parallel testing disabled for now (can overwhelm backend)"
    # Implementation would use background jobs with job control
}

# Print comprehensive summary
print_summary() {
    print_header "📊 TEST SUMMARY"

    echo -e "${WHITE}${BOLD}Overall Statistics:${NC}"
    echo -e "  Total tests: ${TOTAL_TESTS}"
    echo -e "  ${GREEN}Successful: ${SUCCESSFUL_TESTS}${NC}"
    echo -e "  ${RED}Failed: ${FAILED_TESTS}${NC}"
    echo -e "  ${CYAN}Cache hits: ${CACHE_HITS}${NC}"

    if [[ $TOTAL_TESTS -gt 0 ]]; then
        local success_rate cache_hit_rate
        success_rate=$(awk "BEGIN {printf \"%.1f\", $SUCCESSFUL_TESTS * 100 / $TOTAL_TESTS}")
        cache_hit_rate=$(awk "BEGIN {printf \"%.1f\", $CACHE_HITS * 100 / $TOTAL_TESTS}")

        echo -e "  ${GREEN}Success rate: ${success_rate}%${NC}"
        echo -e "  ${CYAN}Cache hit rate: ${cache_hit_rate}%${NC}"
    fi

    echo ""
    echo -e "${WHITE}${BOLD}Performance Analysis:${NC}"

    # Find fastest and slowest tests
    local fastest_time=999999
    local slowest_time=0
    local fastest_test=""
    local slowest_test=""

    for result in "${TIMING_RESULTS[@]}"; do
        IFS=':' read -r test_name elapsed status_code <<< "$result"

        if [[ "$status_code" == "200" ]]; then
            if (( $(awk "BEGIN {print ($elapsed < $fastest_time)}") )); then
                fastest_time="$elapsed"
                fastest_test="$test_name"
            fi

            if (( $(awk "BEGIN {print ($elapsed > $slowest_time)}") )); then
                slowest_time="$elapsed"
                slowest_test="$test_name"
            fi
        fi
    done

    if [[ -n "$fastest_test" ]]; then
        echo -e "  ${GREEN}Fastest: $fastest_test (${fastest_time}s)${NC}"
        echo -e "  ${RED}Slowest: $slowest_test (${slowest_time}s)${NC}"
    fi

    # Cache statistics
    echo ""
    echo -e "${WHITE}${BOLD}Cache Statistics:${NC}"
    local cache_stats
    cache_stats=$(get_cache_stats)

    if [[ "$cache_stats" != "null" ]]; then
        echo "$cache_stats" | jq -r '
            "  Total entries: \(.total_entries // 0)",
            "  Training entries: \(.training_entries // 0)",
            "  Evaluation entries: \(.evaluation_entries // 0)",
            "  Cache file size: \(.cache_file_size_mb // 0) MB",
            "  Unique algorithms: \(.unique_algorithms // 0)",
            "  Unique datasets: \(.unique_datasets // 0)"
        '
    else
        echo -e "  ${YELLOW}Cache stats unavailable${NC}"
    fi

    # Save detailed results to file
    local results_file="test_results_$(date +%Y%m%d_%H%M%S).json"
    {
        echo "{"
        echo "  \"summary\": {"
        echo "    \"total_tests\": $TOTAL_TESTS,"
        echo "    \"successful_tests\": $SUCCESSFUL_TESTS,"
        echo "    \"failed_tests\": $FAILED_TESTS,"
        echo "    \"cache_hits\": $CACHE_HITS"
        echo "  },"
        echo "  \"results\": ["

        local first=true
        for result in "${TEST_RESULTS[@]}"; do
            [[ "$first" == true ]] && first=false || echo "    ,"
            IFS=':' read -r status test_name elapsed from_cache <<< "$result"
            echo -n "    {\"status\": \"$status\", \"test\": \"$test_name\", \"elapsed\": $elapsed, \"from_cache\": $from_cache}"
        done

        echo ""
        echo "  ]"
        echo "}"
    } > "$results_file"

    echo ""
    echo -e "${CYAN}Detailed results saved to: $results_file${NC}"
    echo -e "${CYAN}Test log saved to: $LOG_FILE${NC}"
}

# Main function
main() {
    print_header "🚀 BASH ML CACHE TESTER"
    echo -e "${CYAN}Backend URL: $BASE_URL${NC}"
    echo -e "${CYAN}Log file: $LOG_FILE${NC}"
    echo ""

    # Check dependencies
    check_dependencies

    # Health check
    if ! health_check; then
        exit 1
    fi

    # Clear cache for clean start
    clear_cache

    echo ""

    # Run cache functionality test
    test_cache_functionality

    echo ""

    # Run comprehensive test
    test_all_models

    echo ""

    # Print summary
    print_summary

    echo ""
    print_header "✅ TESTING COMPLETED!"
    echo -e "${GREEN}${BOLD}All tests finished successfully!${NC}"
}

# Run if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
