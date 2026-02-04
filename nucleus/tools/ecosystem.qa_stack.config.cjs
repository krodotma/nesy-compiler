/**
 * PM2 Ecosystem Configuration for the QA Stack
 *
 * Manages:
 * - qa-observer: bus IR normalization + hygiene guard
 * - qa-tool-queue: tool backpressure manager (emit-only by default)
 *
 * Usage:
 *   pm2 start ecosystem.qa_stack.config.cjs
 *   pm2 logs qa-observer
 *   pm2 logs qa-tool-queue
 */

const path = require('path');

const pluribusRoot = process.env.PLURIBUS_ROOT || '/pluribus';
const busDir = process.env.PLURIBUS_BUS_DIR || '/pluribus/.pluribus/bus';
const python = process.env.PYTHON || 'python3';

const toolsDir = path.join(pluribusRoot, 'nucleus', 'tools');
const qaOutputDir = path.join(pluribusRoot, '.pluribus', 'index', 'qa');
const queuePath = path.join(pluribusRoot, '.pluribus', 'index', 'tool_queue', 'queue.ndjson');

module.exports = {
  apps: [
    {
      name: 'qa-observer',
      script: python,
      args: [
        path.join(toolsDir, 'qa_observer.py'),
        '--emit-bus',
        '--max-bus-bytes', '524288000', // 500MB
        '--rotate-seconds', '86400', // 24h
        '--rotate-check-s', '30',
      ],
      cwd: pluribusRoot,
      env: {
        PLURIBUS_ROOT: pluribusRoot,
        PLURIBUS_BUS_DIR: busDir,
        QA_OUTPUT_DIR: qaOutputDir,
      },
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 3000,
      error_file: '/tmp/pm2-qa-observer-error.log',
      out_file: '/tmp/pm2-qa-observer-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      min_uptime: '5s',
      max_memory_restart: '200M',
    },
    {
      name: 'qa-tool-queue',
      script: python,
      args: [
        path.join(toolsDir, 'qa_tool_queue.py'),
        'run',
        '--emit-bus',
        '--max-active', '1',
        '--poll-s', '2.0',
        '--execute',
        '--queue-path', queuePath,
      ],
      cwd: pluribusRoot,
      env: {
        PLURIBUS_ROOT: pluribusRoot,
        PLURIBUS_BUS_DIR: busDir,
      },
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 3000,
      error_file: '/tmp/pm2-qa-tool-queue-error.log',
      out_file: '/tmp/pm2-qa-tool-queue-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      min_uptime: '5s',
      max_memory_restart: '200M',
    },
    {
      name: 'qa-action-executor',
      script: python,
      args: [
        path.join(toolsDir, 'qa_action_executor.py'),
        '--emit-bus',
        '--allow-root',
        '--mode', 'queue',
        '--queue-path', queuePath,
      ],
      cwd: pluribusRoot,
      env: {
        PLURIBUS_ROOT: pluribusRoot,
        PLURIBUS_BUS_DIR: busDir,
      },
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 3000,
      error_file: '/tmp/pm2-qa-action-executor-error.log',
      out_file: '/tmp/pm2-qa-action-executor-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      min_uptime: '5s',
      max_memory_restart: '200M',
    },
  ],
};
