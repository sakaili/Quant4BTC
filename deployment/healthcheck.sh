#!/bin/bash
# Health check and auto-restart script for Quant4BTC

SERVICE_NAME="quant4btc"
LOG_FILE="/home/ubuntu/quant4BTC/logs/healthcheck.log"
RESTART_LOG="/home/ubuntu/quant4BTC/logs/restart.log"

# Check if service is running
if ! systemctl is-active --quiet "$SERVICE_NAME"; then
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] Service $SERVICE_NAME is down! Attempting to restart..." >> "$LOG_FILE"

    # Try to restart
    sudo systemctl start "$SERVICE_NAME"
    sleep 5

    # Check if restart was successful
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo "[$TIMESTAMP] Service restarted successfully" >> "$LOG_FILE"
        echo "[$TIMESTAMP] Service auto-restarted" >> "$RESTART_LOG"
    else
        echo "[$TIMESTAMP] Failed to restart service!" >> "$LOG_FILE"
        # You can add email notification here
    fi
else
    # Service is running, log success (optional, comment out if too verbose)
    # TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    # echo "[$TIMESTAMP] Service is running normally" >> "$LOG_FILE"
    :
fi
