#!/bin/bash
#
# This is the correct, final autossh tunnel script.
# It automatically reconnects and keeps the connection alive.
# It correctly forwards all ports to the COMPUTE_NODE_HOST.
#

export AUTOSSH_PIDFILE="/tmp/robot_brain.pid"
export AUTOSSH_GATETIME=0

# --- CONFIGURATION: VERIFY THESE THREE VARIABLES ---
REMOTE_USER_HOST="jayinakshav_iitp@paramrudra.iitp.ac.in"
REMOTE_PORT=4422
COMPUTE_NODE_HOST="ragpu003" # The specific GPU node your server is running on

# --- AUTOSSH COMMAND ---
echo "Starting robust, auto-reconnecting SSH tunnels to $COMPUTE_NODE_HOST..."
echo "PID file will be at ${AUTOSSH_PIDFILE}"

autossh -M 0 -N \
    -o "ServerAliveInterval 30" \
    -o "ServerAliveCountMax 3" \
    -p $REMOTE_PORT \
    -L 5000:$COMPUTE_NODE_HOST:5000 \
    -L 9001:$COMPUTE_NODE_HOST:9001 \
    -L 9002:$COMPUTE_NODE_HOST:9002 \
    -L 9003:$COMPUTE_NODE_HOST:9003 \
    -L 9004:$COMPUTE_NODE_HOST:9004 \
    -L 8080:$COMPUTE_NODE_HOST:8080 \
    $REMOTE_USER_HOST

echo "Tunnel process terminated."
