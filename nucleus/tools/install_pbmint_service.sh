#!/bin/bash
# Install PBMINT as a system service

set -e

echo "Installing PBMINT - Pluribus Documentation Department Iteration Operator..."

# Copy service file to systemd directory
sudo cp /pluribus/nucleus/deploy/systemd/pluribus-pbmint.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable pluribus-pbmint.service
sudo systemctl start pluribus-pbmint.service

echo "PBMINT service installed and running!"
echo "Check status with: systemctl status pluribus-pbmint.service"
echo "View logs with: journalctl -u pluribus-pbmint.service -f"