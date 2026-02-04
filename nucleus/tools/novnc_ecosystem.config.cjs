/**
 * PM2 Ecosystem Configuration for noVNC services
 *
 * Manages:
 * - novnc-vnc: TigerVNC server on display :1 (port 5901)
 * - novnc-websockify: websockify bridge (port 6080 -> 5901)
 *
 * Usage:
 *   pm2 start novnc_ecosystem.config.cjs
 *   pm2 logs novnc-websockify
 *   pm2 status
 *
 * Access noVNC at: https://kroma.live/vnc/vnc.html
 * VNC Password: see /pluribus/.pluribus/vnc_password.txt
 */

module.exports = {
  apps: [
    {
      name: 'novnc-vnc',
      script: '/pluribus/nucleus/tools/vncserver_wrapper.sh',
      args: ':1 -geometry 1920x1080 -depth 24 -rfbport 5901 -localhost no -PasswordFile /pluribus/.pluribus/vnc_passwd -UseBlacklist 0 -fg',
      interpreter: '/bin/bash',
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 5000,
      stop_exit_codes: [0],
      // Environment
      env: {
        HOME: '/root',
        USER: 'root',
      },
      // Logs
      error_file: '/tmp/pm2-novnc-vnc-error.log',
      out_file: '/tmp/pm2-novnc-vnc-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      // Health check
      min_uptime: '10s',
      max_memory_restart: '500M',
    },
    {
      name: 'novnc-websockify',
      script: '/pluribus/nucleus/tools/websockify_wrapper.sh',
      // Use bash wrapper script to run websockify
      interpreter: '/bin/bash',
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 3000,
      stop_exit_codes: [0],
      // Wait for VNC to be ready
      wait_ready: false,
      // Logs
      error_file: '/tmp/pm2-novnc-websockify-error.log',
      out_file: '/tmp/pm2-novnc-websockify-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      // Health check
      min_uptime: '5s',
      max_memory_restart: '200M',
    },
  ],
};
