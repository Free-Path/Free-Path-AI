#!/bin/bash

# 설정
PID_FILE="server.pid"
LOG_FILE="server.out"

start_server() {
    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Server is already running (PID: $pid)"
            return
        fi
        rm "$PID_FILE"
    fi

    echo "Starting server..."
    nohup python scripts/run_server.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Server started with PID: $(cat $PID_FILE)"
}

stop_server() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Server is not running (PID file not found)"
        return
    fi

    pid=$(cat "$PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Stopping server (PID: $pid)"
        kill "$pid"
        rm "$PID_FILE"
        echo "Server stopped"
    else
        echo "Server is not running (stale PID file)"
        rm "$PID_FILE"
    fi
}

check_status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Server is not running (PID file not found)"
        return 1
    fi

    pid=$(cat "$PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Server is running (PID: $pid)"
        return 0
    else
        echo "Server is not running (stale PID file)"
        rm "$PID_FILE"
        return 1
    fi
}

case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 2
        start_server
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac