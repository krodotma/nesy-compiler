/**
 * PM2 Ecosystem Configuration for World Router & Nucleus Core
 *
 * Usage:
 *   pm2 start ecosystem.world_router.config.js
 *   pm2 stop world-router
 *   pm2 logs world-router
 *   pm2 restart world-router
 */
module.exports = {
  apps: [
    {
      name: "world-router",
      script: "python3",
      args: [
        "/pluribus/nucleus/tools/world_router.py",
        "--port",
        "8080",
        "--bus-dir",
        "/pluribus/.pluribus/bus",
      ],
      cwd: "/pluribus/nucleus/tools",
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_memory_restart: "500M",
      env: {
        PYTHONDONTWRITEBYTECODE: "1",
        PYTHONUNBUFFERED: "1",
        PLURIBUS_BUS_DIR: "/pluribus/.pluribus/bus",
        PLURIBUS_ACTOR: "world-router",
        DISPLAY: ":1",
      },
      // Logging
      error_file: "/var/log/pluribus/world-router-error.log",
      out_file: "/var/log/pluribus/world-router-out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
      merge_logs: true,
      // Restart behavior
      restart_delay: 1000,
      max_restarts: 10,
      min_uptime: 5000,
      // Graceful shutdown
      kill_timeout: 5000,
      listen_timeout: 10000,
    },
    {
      // Superworker Injection Daemon (The Hands)
      // Bridges UI 'worker.spawn' events to backend processes.
      name: "superworker-daemon",
      script: "python3",
      args: [
        "/pluribus/nucleus/tools/superworker_injection.py"
      ],
      cwd: "/pluribus/nucleus/tools",
      interpreter: "none",
      autorestart: true,
      watch: false,
      env: {
        PYTHONDONTWRITEBYTECODE: "1",
        PYTHONUNBUFFERED: "1",
        PLURIBUS_BUS_DIR: "/pluribus/.pluribus/bus",
        PLURIBUS_ACTOR: "superworker-d",
      },
      error_file: "/var/log/pluribus/superworker-error.log",
      out_file: "/var/log/pluribus/superworker-out.log",
    },
    {
      // World Router with Identity Hub enabled
      name: "world-router-identity",
      script: "python3",
      args: [
        "/pluribus/nucleus/tools/world_router.py",
        "--port",
        "8081",
        "--identity-hub",
        "--bus-dir",
        "/pluribus/.pluribus/bus",
      ],
      cwd: "/pluribus/nucleus/tools",
      interpreter: "none",
      autorestart: false,  // Don't auto-start this variant
      watch: false,
      env: {
        PYTHONDONTWRITEBYTECODE: "1",
        PYTHONUNBUFFERED: "1",
        PLURIBUS_BUS_DIR: "/pluribus/.pluribus/bus",
        DISPLAY: ":1",
      },
      error_file: "/var/log/pluribus/world-router-identity-error.log",
      out_file: "/var/log/pluribus/world-router-identity-out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
    },
    {
      // World Router with Native CUA (requires Playwright)
      name: "world-router-cua-native",
      script: "python3",
      args: [
        "/pluribus/nucleus/tools/world_router.py",
        "--port",
        "8082",
        "--cua-native",
        "--bus-dir",
        "/pluribus/.pluribus/bus",
      ],
      cwd: "/pluribus/nucleus/tools",
      interpreter: "none",
      autorestart: false,  // Don't auto-start this variant
      watch: false,
      env: {
        PYTHONDONTWRITEBYTECODE: "1",
        PYTHONUNBUFFERED: "1",
        PLURIBUS_BUS_DIR: "/pluribus/.pluribus/bus",
        DISPLAY: ":1",
      },
      error_file: "/var/log/pluribus/world-router-cua-native-error.log",
      out_file: "/var/log/pluribus/world-router-cua-native-out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
    },
  ],
};