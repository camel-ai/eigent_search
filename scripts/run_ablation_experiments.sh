#!/bin/bash
# ========= Copyright 2025 @ CAMEL-AI.org. All Rights Reserved. =========
# Ablation Experiment Runner Script
#
# This script runs ablation experiments for the Q+ search agent.
#
# Available ablations:
#   - none:              Full Q+ (no ablation)
#   - fixed_10_results:  Only select_query_and_search with fixed 10 results
#   - no_query_tools:    Search + analyze + extract (no query refinement)
#   - query_tools_only:  Search + query tools (no analyze/extract)
#
# Usage:
#   ./scripts/run_ablation_experiments.sh [eval_script] [--test-all | -n NUM] [--resume]
#
# Examples:
#   ./scripts/run_ablation_experiments.sh                              # Default: simpleqa, 5 questions
#   ./scripts/run_ablation_experiments.sh simpleqa_verified_eval       # SimpleQA, 5 questions
#   ./scripts/run_ablation_experiments.sh frames_eval --test-all       # FRAMES, all questions
#   ./scripts/run_ablation_experiments.sh frames_eval -n 10            # FRAMES, 10 questions
#   ./scripts/run_ablation_experiments.sh --test-all                   # SimpleQA, all questions
#   ./scripts/run_ablation_experiments.sh -n 20                        # SimpleQA, 20 questions
#   ./scripts/run_ablation_experiments.sh --resume                     # Resume from default directories
#   ./scripts/run_ablation_experiments.sh frames_eval --test-all --resume  # Resume FRAMES
# =========================================================================

set -e  # Exit on error

# ===========================================
# CONFIGURATION - Modify these as needed
# ===========================================

# Which ablations to run (space-separated list)
# Options: none, fixed_10_results, fixed_10_results_eigent_prompt, no_query_tools, query_tools_only
ABLATIONS_TO_RUN="fixed_10_results fixed_10_results_eigent_prompt no_query_tools query_tools_only"

# Model and worker configuration
MODEL="azure-gpt-4.1-mini"
#MODEL="gpt-4.1-mini"
WORKERS=5
AGENT_TYPE="eigent_search_q+"

# Default test mode (can be overridden by command line arguments)
DEFAULT_TEST_ALL=false
DEFAULT_NUM_QUESTIONS=5

# ===========================================
# DEFAULT RESUME DIRECTORIES
# Set these to the directories you want to resume from for each ablation
# Format: results/<benchmark>_eval_<timestamp>
# Leave empty ("") to start fresh for that ablation
# ===========================================

# SimpleQA Verified resume directories
declare -A SIMPLEQA_VERIFIED_RESUME_DIRS=(
    ["fixed_10_results"]=""
    ["fixed_10_results_eigent_prompt"]=""
    ["no_query_tools"]=""
    ["query_tools_only"]=""
)

# Frames resume directories
declare -A FRAMES_RESUME_DIRS=(
    ["fixed_10_results"]=""
    ["fixed_10_results_eigent_prompt"]=""
    ["no_query_tools"]=""
    ["query_tools_only"]=""
)

# BrowseComp resume directories
declare -A BROWSECOMP_RESUME_DIRS=(
    ["fixed_10_results"]=""
    ["fixed_10_results_eigent_prompt"]=""
    ["no_query_tools"]=""
    ["query_tools_only"]=""
)

# WebWalker resume directories
declare -A WEBWALKER_RESUME_DIRS=(
    ["fixed_10_results"]=""
    ["fixed_10_results_eigent_prompt"]=""
    ["no_query_tools"]=""
    ["query_tools_only"]=""
)

# XBench resume directories
declare -A XBENCH_RESUME_DIRS=(
    ["fixed_10_results"]=""
    ["fixed_10_results_eigent_prompt"]=""
    ["no_query_tools"]=""
    ["query_tools_only"]=""
)

# ===========================================
# END CONFIGURATION
# ===========================================

# Initialize variables
EVAL_SCRIPT=""
TEST_ALL="${DEFAULT_TEST_ALL}"
NUM_QUESTIONS="${DEFAULT_NUM_QUESTIONS}"
USE_RESUME=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-all)
            TEST_ALL=true
            shift
            ;;
        --resume)
            USE_RESUME=true
            shift
            ;;
        -n)
            TEST_ALL=false
            NUM_QUESTIONS="$2"
            shift 2
            ;;
        *)
            # First non-option argument is the eval script
            if [[ -z "${EVAL_SCRIPT}" ]]; then
                EVAL_SCRIPT="$1"
            else
                echo "Error: Unknown argument '$1'"
                echo "Usage: ./scripts/run_ablation_experiments.sh [eval_script] [--test-all | -n NUM] [--resume]"
                exit 1
            fi
            shift
            ;;
    esac
done

# Function to get resume directory for a given eval script and ablation
get_resume_dir() {
    local eval_script="$1"
    local ablation="$2"

    case "${eval_script}" in
        simpleqa_verified_eval)
            echo "${SIMPLEQA_VERIFIED_RESUME_DIRS[$ablation]:-}"
            ;;
        frames_eval)
            echo "${FRAMES_RESUME_DIRS[$ablation]:-}"
            ;;
        browsecomp_eval)
            echo "${BROWSECOMP_RESUME_DIRS[$ablation]:-}"
            ;;
        webwalker_eval)
            echo "${WEBWALKER_RESUME_DIRS[$ablation]:-}"
            ;;
        xbench_eval)
            echo "${XBENCH_RESUME_DIRS[$ablation]:-}"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Default eval script if not provided
EVAL_SCRIPT="${EVAL_SCRIPT:-simpleqa_verified_eval}"

# Remove .py extension if provided
EVAL_SCRIPT="${EVAL_SCRIPT%.py}"

# Validate the script exists
if [[ ! -f "scripts/${EVAL_SCRIPT}.py" ]]; then
    echo "Error: scripts/${EVAL_SCRIPT}.py not found!"
    echo "Available eval scripts:"
    ls scripts/*_eval.py 2>/dev/null | sed 's/scripts\//  - /g' | sed 's/.py//g'
    exit 1
fi

# Setup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${EVAL_SCRIPT}_ablation_${TIMESTAMP}"

# Create log directory
mkdir -p "${LOG_DIR}"

# Convert ablations string to array
read -r -a ABLATION_ARRAY <<< "${ABLATIONS_TO_RUN}"
TOTAL_ABLATIONS=${#ABLATION_ARRAY[@]}

# Build test mode argument
if [[ "${TEST_ALL}" == "true" ]]; then
    TEST_MODE_ARG="--test-all"
    TEST_MODE_DESC="all questions (--test-all)"
else
    TEST_MODE_ARG="-n ${NUM_QUESTIONS}"
    TEST_MODE_DESC="${NUM_QUESTIONS} questions (-n ${NUM_QUESTIONS})"
fi

echo "=========================================="
echo "Starting Ablation Experiments"
echo "Eval Script: ${EVAL_SCRIPT}.py"
echo "Timestamp: ${TIMESTAMP}"
echo "Model: ${MODEL}"
echo "Workers: ${WORKERS}"
echo "Test Mode: ${TEST_MODE_DESC}"
echo "Ablations to run: ${ABLATIONS_TO_RUN}"
echo "Log Directory: ${LOG_DIR}"
echo "=========================================="

# Arrays to store PIDs, ablation names, and log files
declare -a PIDS
declare -a ABLATION_NAMES
declare -a LOG_FILES

# Run each ablation
COUNTER=0
for ABLATION in "${ABLATION_ARRAY[@]}"; do
    COUNTER=$((COUNTER + 1))

    # Sleep between ablations to ensure unique timestamps (skip for first ablation)
    if [[ ${COUNTER} -gt 1 ]]; then
        echo "Sleeping 20 seconds before next ablation..."
        sleep 20
    fi

    # Get fresh timestamp for each ablation
    ABLATION_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    echo "[${COUNTER}/${TOTAL_ABLATIONS}] Starting: ${ABLATION}..."

    LOG_FILE="${LOG_DIR}/${EVAL_SCRIPT}_qplus_${ABLATION}_${ABLATION_TIMESTAMP}.log"

    # Build resume argument if --resume flag is set and resume directory exists
    RESUME_ARG=""
    if [[ "${USE_RESUME}" == "true" ]]; then
        RESUME_DIR=$(get_resume_dir "${EVAL_SCRIPT}" "${ABLATION}")
        if [[ -n "${RESUME_DIR}" && -d "${RESUME_DIR}" ]]; then
            RESUME_ARG="--resume-from ${RESUME_DIR}"
            echo "  Resuming from: ${RESUME_DIR}"
        elif [[ -n "${RESUME_DIR}" ]]; then
            echo "  Warning: Resume directory not found: ${RESUME_DIR}"
            echo "  Starting fresh instead."
        else
            echo "  No resume directory configured for ${ABLATION}, starting fresh."
        fi
    fi

    nohup python "scripts/${EVAL_SCRIPT}.py" \
        -a "${AGENT_TYPE}" \
        -m "${MODEL}" \
        -w "${WORKERS}" \
        ${TEST_MODE_ARG} \
        --ablation "${ABLATION}" \
        ${RESUME_ARG} \
        > "${LOG_FILE}" 2>&1 &

    PID=$!
    PIDS+=("${PID}")
    ABLATION_NAMES+=("${ABLATION}")
    LOG_FILES+=("${LOG_FILE}")

    echo "  PID: ${PID}"
    echo "  Log: ${LOG_FILE}"
done

echo ""
echo "=========================================="
echo "All experiments started!"
echo "=========================================="
echo ""
echo "Process IDs:"
for i in "${!PIDS[@]}"; do
    printf "  %-20s ${PIDS[$i]}\n" "${ABLATION_NAMES[$i]}:"
done
echo ""
echo "Monitor progress with:"
for LOG_FILE in "${LOG_FILES[@]}"; do
    echo "  tail -f ${LOG_FILE}"
done
echo ""
echo "Check if processes are running:"
echo "  ps aux | grep ${EVAL_SCRIPT}"
echo ""
echo "Kill all experiments:"
echo "  kill ${PIDS[*]}"
echo "=========================================="

# Save PIDs to a file for easy reference
echo "${PIDS[*]}" > "${LOG_DIR}/pids.txt"
echo "PIDs saved to: ${LOG_DIR}/pids.txt"
