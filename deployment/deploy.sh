#!/bin/bash
# Deployment script for Ubuntu ECS server

set -e  # Exit on error

echo "======================================"
echo "Quant4BTC Deployment Script"
echo "======================================"

# Configuration
APP_DIR="/home/ubuntu/quant4BTC"
SERVICE_NAME="quant4btc"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="$APP_DIR/logs"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    print_warning "Run as: ./deploy.sh"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p "$LOG_DIR"
mkdir -p "$APP_DIR/deployment"
print_success "Directories created"

# Check if .env exists
echo ""
echo "Checking configuration..."
if [ ! -f "$APP_DIR/.env" ]; then
    if [ -f "$APP_DIR/.env.template" ]; then
        print_warning ".env file not found"
        echo "Copying from template..."
        cp "$APP_DIR/.env.template" "$APP_DIR/.env"
        print_warning "Please edit .env file with your configuration"
        print_warning "At minimum, set:"
        echo "  - BINANCE_API_KEY"
        echo "  - BINANCE_SECRET"
        echo "  - USE_DEMO (true for testnet, false for production)"
        echo ""
        read -p "Press Enter to continue after editing .env, or Ctrl+C to exit..."
    else
        print_error ".env.template not found"
        exit 1
    fi
else
    print_success ".env file found"
fi

# Setup Python virtual environment
echo ""
echo "Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
source "$VENV_DIR/bin/activate"

echo ""
echo "Installing Python dependencies..."
if [ -f "$APP_DIR/requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r "$APP_DIR/requirements.txt"
    print_success "Dependencies installed"
else
    print_warning "requirements.txt not found, installing minimal dependencies..."
    pip install --upgrade pip
    pip install ccxt pandas numpy python-dotenv
    print_success "Minimal dependencies installed"
fi

# Test configuration loading
echo ""
echo "Testing configuration..."
cd "$APP_DIR"
python3 -c "from config import Config; cfg = Config(); print(f'Strategy: {cfg.strategy_name}, Symbol: {cfg.symbol}')" && print_success "Configuration loaded successfully" || print_error "Configuration test failed"

# Setup systemd service
echo ""
echo "Setting up systemd service..."
SERVICE_FILE="$APP_DIR/deployment/quant4btc.service"

if [ ! -f "$SERVICE_FILE" ]; then
    print_error "Service file not found: $SERVICE_FILE"
    exit 1
fi

# Update service file with current user and paths
sed -i "s|User=ubuntu|User=$USER|g" "$SERVICE_FILE"
sed -i "s|Group=ubuntu|Group=$USER|g" "$SERVICE_FILE"
sed -i "s|/home/ubuntu/quant4BTC|$APP_DIR|g" "$SERVICE_FILE"

sudo cp "$SERVICE_FILE" "/etc/systemd/system/$SERVICE_NAME.service"
sudo systemctl daemon-reload
print_success "Systemd service installed"

# Enable and start service
echo ""
read -p "Do you want to enable and start the service now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo systemctl enable "$SERVICE_NAME.service"
    print_success "Service enabled to start on boot"

    sudo systemctl restart "$SERVICE_NAME.service"
    print_success "Service started"

    echo ""
    echo "Checking service status..."
    sleep 2
    sudo systemctl status "$SERVICE_NAME.service" --no-pager
else
    print_warning "Service not started"
    echo "To start manually, run:"
    echo "  sudo systemctl enable $SERVICE_NAME.service"
    echo "  sudo systemctl start $SERVICE_NAME.service"
fi

echo ""
echo "======================================"
echo "Deployment Summary"
echo "======================================"
echo "App Directory: $APP_DIR"
echo "Log Directory: $LOG_DIR"
echo "Service Name: $SERVICE_NAME"
echo ""
echo "Useful commands:"
echo "  Start service:   sudo systemctl start $SERVICE_NAME"
echo "  Stop service:    sudo systemctl stop $SERVICE_NAME"
echo "  Restart service: sudo systemctl restart $SERVICE_NAME"
echo "  View status:     sudo systemctl status $SERVICE_NAME"
echo "  View logs:       sudo journalctl -u $SERVICE_NAME -f"
echo "  View app log:    tail -f $LOG_DIR/bot.log"
echo "  View errors:     tail -f $LOG_DIR/bot.error.log"
echo ""
print_success "Deployment completed!"
