#!/bin/bash
# Demonstration script for MT_DEBUG JSON output parsing
# This script shows how to use the JSON debug output for analysis

echo "MT_DEBUG JSON Output Analysis Demo"
echo "==================================="
echo ""

# Check if the program is compiled with JSON debug
if [ ! -f ./test_mt_debug ]; then
    echo "Error: test_mt_debug not found. Please build with MT_DEBUG_JSON first:"
    echo "  CFLAGS=\"-DMT_DEBUG -DMT_DEBUG_JSON\" make test_mt_debug"
    exit 1
fi

# Run the test and capture JSON output
echo "Running test_mt_debug and capturing JSON output..."
./test_mt_debug 2>mt_debug_raw.json 1>/dev/null

if [ ! -f mt_debug_raw.json ]; then
    echo "Error: Failed to capture JSON output"
    exit 1
fi

# Extract only valid JSON lines (timing, queue, load_balance)
grep -E '"type":"(timing|queue|load_balance)"' mt_debug_raw.json > mt_debug.json

echo "JSON output saved to mt_debug.json"
echo ""

# Check if jq is available
if command -v jq &> /dev/null; then
    echo "Analyzing with jq..."
    echo ""
    
    echo "=== Queue Depth ==="
    grep '"type":"queue"' mt_debug.json | jq '.depth' | sort -n | tail -1 | xargs echo "Maximum queue depth:"
    echo ""
    
    echo "=== Timing Analysis by Operation ==="
    grep '"type":"timing"' mt_debug.json | jq -s '
        group_by(.operation) | 
        map({
            operation: .[0].operation, 
            count: length, 
            min_us: (map(.duration_us) | min),
            max_us: (map(.duration_us) | max),
            avg_us: (map(.duration_us) | add / length)
        })' | jq -r '.[] | "\(.operation): \(.count) calls, avg=\(.avg_us | floor)us, min=\(.min_us | floor)us, max=\(.max_us | floor)us"'
    echo ""
    
    echo "=== Load Balance Analysis ==="
    grep '"type":"load_balance"' mt_debug.json | jq -r '
        if .warning then 
            "⚠ WARNING: \(.variance_pct | floor)% variance (min=\(.min_us | floor)us, max=\(.max_us | floor)us, avg=\(.avg_us | floor)us, threads=\(.thread_count))" 
        else 
            "✓ OK: \(.variance_pct | floor)% variance (threads=\(.thread_count))"
        end'
    echo ""
    
    echo "=== Per-Thread Execution Time ==="
    grep '"type":"timing"' mt_debug.json | jq -s '
        group_by(.tid) | 
        map({
            thread: .[0].tid,
            total_us: (map(.duration_us) | add),
            tasks: length
        }) | sort_by(.total_us) | reverse | .[0:5]' | jq -r '.[] | "Thread \(.thread): \(.tasks) tasks, total=\(.total_us | floor)us"'
    
else
    echo "jq not found. Install jq for advanced JSON analysis."
    echo "Raw JSON examples:"
    echo ""
    echo "=== Timing Events (first 5) ==="
    head -5 mt_debug.json
    echo ""
    echo "=== Load Balance Events ==="
    grep '"type":"load_balance"' mt_debug.json
fi

echo ""
echo "Structured JSON output available in mt_debug.json"
echo "Raw output (including trace/thread messages) in mt_debug_raw.json"
echo "You can parse it with tools like jq, Python, or any JSON parser."
