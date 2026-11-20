#!/usr/bin/env bash

require_idle_gpu() {
    local device=${CUDA_VISIBLE_DEVICES:-0}
    if [[ "$device" == *,* ]]; then
        echo "Fig6 requires exactly one visible GPU; got CUDA_VISIBLE_DEVICES=$device" >&2
        return 1
    fi
    local processes
    processes=$(nvidia-smi -i "$device" --query-compute-apps=pid --format=csv,noheader) || return 1
    if [[ -n "${processes//[[:space:]]/}" ]]; then
        printf 'GPU %s has compute processes; refusing to start a timing phase:\n%s\n' "$device" "$processes" >&2
        return 1
    fi
}

run_guarded() {
    local name=$1
    shift
    require_idle_gpu || return 1
    "$@" > "$OUT/logs/$name.log" 2>&1 &
    local pid=$!
    local last_clean
    last_clean=$(date -Is)
    while kill -0 "$pid" 2>/dev/null; do
        sleep 2
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        local processes
        if ! processes=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-compute-apps=pid --format=csv,noheader); then
            echo "GPU monitor failed in $LABEL/$name; stopping only benchmark PID $pid" >&2
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" || true
            return 1
        fi
        local other
        other=$(printf '%s\n' "$processes" | awk -v own="$pid" 'NF && $1 != own {print $1}')
        if [[ -n "$other" ]]; then
            printf '%s phase=%s last_clean=%s own_pid=%s other_pids=%s\n' "$(date -Is)" "$name" "$last_clean" "$pid" "$other" >> "$OUT/logs/gpu_contention.log"
            echo "GPU contention detected in $LABEL/$name; stopping only benchmark PID $pid" >&2
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" || true
            return 1
        fi
        last_clean=$(date -Is)
    done
    wait "$pid"
}
