# Server lifecycle for the inference stack. Sourced by the PBS job scripts.
#
#     source supek/serve.sh
#     start_servers "$LOGDIR"   # brings up vLLM then optillm, or exits non-zero
#     ...
#     stop_servers              # idempotent; also wired to EXIT by start_servers
#
# Readiness is checked, never assumed: each server must answer HTTP before the
# next step runs. Every wait is bounded, so a server that dies or wedges fails
# the job in minutes with its log quoted, rather than idling on a GPU until
# walltime.
#
# shellcheck shell=bash

VLLM_PID=""
OPTILLM_PID=""
SERVER_LOGDIR=""

# How long each server gets to answer. vLLM's budget covers loading weights off
# Lustre and profiling the KV cache; optillm only has to bind a port.
: "${VLLM_READY_TIMEOUT:=900}"
: "${OPTILLM_READY_TIMEOUT:=300}"

die() {
    echo "ERROR: $*" >&2
    stop_servers
    exit 1
}

# Poll a URL until it answers, the deadline passes, or the server process dies.
wait_for_http() {
    local name="$1" url="$2" pid="$3" timeout="$4" log="$5"
    local deadline=$((SECONDS + timeout))

    echo "Waiting for $name at $url (up to ${timeout}s) ..."
    while true; do
        if curl -sf --max-time 5 "$url" >/dev/null 2>&1; then
            echo "$name is up after $((SECONDS - deadline + timeout))s."
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "--- last 60 lines of $log ---" >&2
            tail -n 60 "$log" >&2 || true
            die "$name exited before becoming ready (see $log)"
        fi
        if (( SECONDS >= deadline )); then
            echo "--- last 60 lines of $log ---" >&2
            tail -n 60 "$log" >&2 || true
            die "$name did not become ready within ${timeout}s (see $log)"
        fi
        sleep 5
    done
}

stop_servers() {
    for pid in "$OPTILLM_PID" "$VLLM_PID"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    # Give them a moment to close the GPU cleanly, then insist.
    sleep 10
    for pid in "$OPTILLM_PID" "$VLLM_PID"; do
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    done
    VLLM_PID=""
    OPTILLM_PID=""
}

start_servers() {
    SERVER_LOGDIR="${1:-.}"
    mkdir -p "$SERVER_LOGDIR"

    [ -f "$SIF" ]        || die "container image not found: $SIF (run supek/build_image.sh)"
    [ -d "$MODEL_DIR" ]  || die "model not staged: $MODEL_DIR (run supek/stage_model.sh)"
    [ -f "$OPTILLM_DIR/optillm.py" ] \
        || die "optillm checkout not found at $OPTILLM_DIR (see supek/README.md - it must be copied to Lustre, not pip-installed)"

    # Tear the stack down however the job ends - success, failure, or qdel.
    trap stop_servers EXIT INT TERM

    echo "Starting vLLM: $MODEL_DIR as '$SERVED_MODEL_NAME'"
    echo "  max-model-len=$MAX_MODEL_LEN max-num-seqs=$MAX_NUM_SEQS" \
         "gpu-memory-utilization=$GPU_MEMORY_UTILIZATION tp=$TENSOR_PARALLEL_SIZE"
    apptainer exec --nv "$SIF" \
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_DIR" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --host "$BIND_HOST" --port "$VLLM_PORT" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        > "$SERVER_LOGDIR/vllm.log" 2>&1 &
    VLLM_PID=$!

    wait_for_http "vLLM" "http://127.0.0.1:$VLLM_PORT/v1/models" \
        "$VLLM_PID" "$VLLM_READY_TIMEOUT" "$SERVER_LOGDIR/vllm.log"

    # Confirm it is serving under the name the harness will ask for, not just
    # that something answered on the port.
    if ! curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" | grep -q "\"$SERVED_MODEL_NAME\""; then
        curl -s "http://127.0.0.1:$VLLM_PORT/v1/models" >&2
        die "vLLM is up but is not serving '$SERVED_MODEL_NAME'"
    fi

    echo "Starting optillm (router device: $OPTILLM_ROUTER_DEVICE)"
    # Run optillm from the staged checkout, not from the image: the router's
    # configurable device and the optillm_router_approach response field are
    # local modifications that a PyPI install does not have.
    apptainer exec --nv "$SIF" \
        python "$OPTILLM_DIR/optillm.py" \
            --base_url "http://127.0.0.1:$VLLM_PORT/v1" \
            --host "$BIND_HOST" \
            --port "$OPTILLM_PORT" \
            --model "$SERVED_MODEL_NAME" \
        > "$SERVER_LOGDIR/optillm.log" 2>&1 &
    OPTILLM_PID=$!

    wait_for_http "optillm" "http://127.0.0.1:$OPTILLM_PORT/v1/models" \
        "$OPTILLM_PID" "$OPTILLM_READY_TIMEOUT" "$SERVER_LOGDIR/optillm.log"

    echo "Inference stack ready: optillm :$OPTILLM_PORT -> vLLM :$VLLM_PORT"
}

# One real chat completion through the whole stack. Catches the failure modes a
# port check cannot: a model that loads but cannot generate, or an optillm that
# accepts connections but cannot reach vLLM.
smoke_test() {
    echo "Smoke-testing a chat completion through optillm ..."
    local response
    response=$(curl -sf --max-time 300 \
        "http://127.0.0.1:$OPTILLM_PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"$SERVED_MODEL_NAME\",
             \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the word OK.\"}],
             \"max_tokens\": 16}") \
        || die "smoke test request failed (see $SERVER_LOGDIR/optillm.log)"

    echo "$response" | head -c 500
    echo
    echo "$response" | grep -q '"content"' \
        || die "smoke test returned no completion content"
    echo "Smoke test passed."
}
