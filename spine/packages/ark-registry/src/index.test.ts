/**
 * @ark/registry tests
 *
 * Comprehensive tests for Service Registry
 * Target: 90%+ coverage
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { Ring } from '@ark/core';
import {
  ServiceRegistry,
  createRegistry,
  nowIsoUtc,
  defaultActor,
  ensureDir,
  appendNdjson,
  iterNdjson,
  findRhizomeRoot,
  VERSION,
  type ServiceDef,
  type ServiceInstance,
  type CreateServiceOptions,
  type ServiceKind,
  type RestartPolicy,
  type HealthStatus,
} from './index.js';

describe('ServiceRegistry', () => {
  let tmpDir: string;
  let registry: ServiceRegistry;

  beforeEach(() => {
    // Create temp directory
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ark-registry-test-'));

    // Create .pluribus directory structure
    const pluribusDir = path.join(tmpDir, '.pluribus');
    fs.mkdirSync(pluribusDir, { recursive: true });
    fs.writeFileSync(
      path.join(pluribusDir, 'rhizome.json'),
      '{}',
      'utf-8'
    );

    registry = createRegistry({ root: tmpDir });
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  describe('utility functions', () => {
    describe('nowIsoUtc', () => {
      it('returns ISO timestamp without milliseconds', () => {
        const iso = nowIsoUtc();
        expect(iso).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
      });

      it('returns current time', () => {
        const before = Date.now();
        const iso = nowIsoUtc();
        const after = Date.now();

        const parsed = new Date(iso).getTime();
        expect(parsed).toBeGreaterThanOrEqual(before - 1000);
        expect(parsed).toBeLessThanOrEqual(after + 1000);
      });
    });

    describe('defaultActor', () => {
      it('returns PLURIBUS_ACTOR if set', () => {
        const original = process.env.PLURIBUS_ACTOR;
        process.env.PLURIBUS_ACTOR = 'test-actor';

        expect(defaultActor()).toBe('test-actor');

        if (original) {
          process.env.PLURIBUS_ACTOR = original;
        } else {
          delete process.env.PLURIBUS_ACTOR;
        }
      });

      it('falls back to USER env var', () => {
        const originalPluribusActor = process.env.PLURIBUS_ACTOR;
        const originalUser = process.env.USER;

        delete process.env.PLURIBUS_ACTOR;
        process.env.USER = 'test-user';

        expect(defaultActor()).toBe('test-user');

        if (originalPluribusActor) {
          process.env.PLURIBUS_ACTOR = originalPluribusActor;
        }
        if (originalUser) {
          process.env.USER = originalUser;
        }
      });
    });

    describe('ensureDir', () => {
      it('creates directory if it does not exist', () => {
        const newDir = path.join(tmpDir, 'a', 'b', 'c');
        expect(fs.existsSync(newDir)).toBe(false);

        ensureDir(newDir);

        expect(fs.existsSync(newDir)).toBe(true);
      });

      it('does nothing if directory already exists', () => {
        const existingDir = path.join(tmpDir, 'existing');
        fs.mkdirSync(existingDir);

        ensureDir(existingDir);

        expect(fs.existsSync(existingDir)).toBe(true);
      });
    });

    describe('appendNdjson', () => {
      it('appends JSON object as a line', () => {
        const filePath = path.join(tmpDir, 'test.ndjson');
        const obj = { id: 'test', value: 42 };

        appendNdjson(filePath, obj);

        const content = fs.readFileSync(filePath, 'utf-8');
        expect(content).toBe('{"id":"test","value":42}\n');
      });

      it('creates parent directories if needed', () => {
        const filePath = path.join(tmpDir, 'nested', 'dir', 'test.ndjson');

        appendNdjson(filePath, { id: 'test' });

        expect(fs.existsSync(filePath)).toBe(true);
      });

      it('appends multiple lines', () => {
        const filePath = path.join(tmpDir, 'multi.ndjson');

        appendNdjson(filePath, { line: 1 });
        appendNdjson(filePath, { line: 2 });
        appendNdjson(filePath, { line: 3 });

        const content = fs.readFileSync(filePath, 'utf-8');
        const lines = content.trim().split('\n');
        expect(lines.length).toBe(3);
      });
    });

    describe('iterNdjson', () => {
      it('yields nothing for nonexistent file', () => {
        const results = Array.from(iterNdjson('/nonexistent/file.ndjson'));
        expect(results.length).toBe(0);
      });

      it('yields parsed JSON objects', () => {
        const filePath = path.join(tmpDir, 'read.ndjson');
        fs.writeFileSync(
          filePath,
          '{"id":"a"}\n{"id":"b"}\n{"id":"c"}\n',
          'utf-8'
        );

        const results = Array.from(iterNdjson(filePath));

        expect(results.length).toBe(3);
        expect(results[0]).toEqual({ id: 'a' });
        expect(results[1]).toEqual({ id: 'b' });
        expect(results[2]).toEqual({ id: 'c' });
      });

      it('skips empty lines', () => {
        const filePath = path.join(tmpDir, 'sparse.ndjson');
        fs.writeFileSync(filePath, '{"id":"a"}\n\n{"id":"b"}\n\n\n', 'utf-8');

        const results = Array.from(iterNdjson(filePath));

        expect(results.length).toBe(2);
      });

      it('skips malformed JSON lines', () => {
        const filePath = path.join(tmpDir, 'malformed.ndjson');
        fs.writeFileSync(
          filePath,
          '{"id":"good1"}\nnot json\n{"id":"good2"}\n{broken\n',
          'utf-8'
        );

        const results = Array.from(iterNdjson(filePath));

        expect(results.length).toBe(2);
        expect(results[0]).toEqual({ id: 'good1' });
        expect(results[1]).toEqual({ id: 'good2' });
      });
    });

    describe('findRhizomeRoot', () => {
      it('finds rhizome root when .pluribus/rhizome.json exists', () => {
        const root = findRhizomeRoot(tmpDir);
        expect(root).toBe(tmpDir);
      });

      it('finds root from nested directory', () => {
        const nestedDir = path.join(tmpDir, 'a', 'b', 'c');
        fs.mkdirSync(nestedDir, { recursive: true });

        const root = findRhizomeRoot(nestedDir);
        expect(root).toBe(tmpDir);
      });

      it('returns null when no rhizome.json exists', () => {
        const noRhizomeDir = fs.mkdtempSync(path.join(os.tmpdir(), 'no-rhizome-'));
        try {
          const root = findRhizomeRoot(noRhizomeDir);
          expect(root).toBeNull();
        } finally {
          fs.rmSync(noRhizomeDir, { recursive: true, force: true });
        }
      });
    });
  });

  describe('init', () => {
    it('creates services directory', async () => {
      await registry.init();

      const servicesDir = path.join(tmpDir, '.pluribus', 'services');
      expect(fs.existsSync(servicesDir)).toBe(true);
    });

    it('creates pid directory', async () => {
      await registry.init();

      const pidDir = path.join(tmpDir, '.pluribus', 'services', 'pids');
      expect(fs.existsSync(pidDir)).toBe(true);
    });

    it('creates registry.ndjson file', async () => {
      await registry.init();

      expect(fs.existsSync(registry.getRegistryPath())).toBe(true);
    });

    it('creates instances.ndjson file', async () => {
      await registry.init();

      expect(fs.existsSync(registry.getInstancesPath())).toBe(true);
    });
  });

  describe('registerService', () => {
    beforeEach(async () => {
      await registry.init();
    });

    it('registers a service with minimal options', () => {
      const serviceId = registry.registerService({
        name: 'test-service',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      expect(serviceId).toBeDefined();
      expect(typeof serviceId).toBe('string');
      expect(serviceId.length).toBeGreaterThan(0);
    });

    it('uses provided ID if given', () => {
      const serviceId = registry.registerService({
        id: 'custom-id',
        name: 'test-service',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      expect(serviceId).toBe('custom-id');
    });

    it('sets default values correctly', () => {
      const serviceId = registry.registerService({
        name: 'test-service',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const svc = registry.getService(serviceId);
      expect(svc).toBeDefined();
      expect(svc!.description).toBe('');
      expect(svc!.dependsOn).toEqual([]);
      expect(svc!.env).toEqual({});
      expect(svc!.args).toEqual([]);
      expect(svc!.tags).toEqual([]);
      expect(svc!.autoStart).toBe(false);
      expect(svc!.restartPolicy).toBe('never');
      expect(svc!.lineage).toBe('orphan');
      expect(svc!.gates).toEqual({});
      expect(svc!.omegaMotif).toBe(false);
      expect(svc!.cmpScore).toBe(0);
      expect(svc!.ring).toBe(Ring.User);
    });

    it('stores all provided options', () => {
      const options: CreateServiceOptions = {
        id: 'full-service',
        name: 'Full Service',
        kind: 'port',
        entryPoint: 'scripts/server.py',
        description: 'A fully configured service',
        port: 8080,
        dependsOn: ['dep-1', 'dep-2'],
        env: { NODE_ENV: 'production' },
        args: ['--verbose', '--port', '8080'],
        tags: ['web', 'api'],
        autoStart: true,
        restartPolicy: 'always',
        healthCheck: 'http://localhost:8080/health',
        lineage: 'core.web',
        gates: { E: 'api_responsive', L: 'liveness' },
        omegaMotif: true,
        cmpScore: 0.75,
        ring: Ring.Service,
      };

      const serviceId = registry.registerService(options);
      const svc = registry.getService(serviceId);

      expect(svc).toBeDefined();
      expect(svc!.name).toBe('Full Service');
      expect(svc!.kind).toBe('port');
      expect(svc!.entryPoint).toBe('scripts/server.py');
      expect(svc!.description).toBe('A fully configured service');
      expect(svc!.port).toBe(8080);
      expect(svc!.dependsOn).toEqual(['dep-1', 'dep-2']);
      expect(svc!.env).toEqual({ NODE_ENV: 'production' });
      expect(svc!.args).toEqual(['--verbose', '--port', '8080']);
      expect(svc!.tags).toEqual(['web', 'api']);
      expect(svc!.autoStart).toBe(true);
      expect(svc!.restartPolicy).toBe('always');
      expect(svc!.healthCheck).toBe('http://localhost:8080/health');
      expect(svc!.lineage).toBe('core.web');
      expect(svc!.gates).toEqual({ E: 'api_responsive', L: 'liveness' });
      expect(svc!.omegaMotif).toBe(true);
      expect(svc!.cmpScore).toBe(0.75);
      expect(svc!.ring).toBe(Ring.Service);
    });

    it('sets createdIso timestamp', () => {
      const before = new Date().toISOString();
      const serviceId = registry.registerService({
        name: 'test',
        kind: 'process',
        entryPoint: 'test.py',
      });
      const after = new Date().toISOString();

      const svc = registry.getService(serviceId);
      expect(svc!.createdIso).toBeDefined();
      expect(new Date(svc!.createdIso).getTime()).toBeGreaterThanOrEqual(
        new Date(before).getTime() - 1000
      );
      expect(new Date(svc!.createdIso).getTime()).toBeLessThanOrEqual(
        new Date(after).getTime() + 1000
      );
    });

    it('persists to NDJSON file', () => {
      registry.registerService({
        name: 'persisted',
        kind: 'process',
        entryPoint: 'test.py',
      });

      const content = fs.readFileSync(registry.getRegistryPath(), 'utf-8');
      expect(content).toContain('service_def');
      expect(content).toContain('persisted');
    });
  });

  describe('unregisterService', () => {
    beforeEach(async () => {
      await registry.init();
    });

    it('removes service from registry', () => {
      const serviceId = registry.registerService({
        name: 'to-remove',
        kind: 'process',
        entryPoint: 'test.py',
      });

      expect(registry.getService(serviceId)).toBeDefined();

      const result = registry.unregisterService(serviceId);

      expect(result).toBe(true);
      expect(registry.getService(serviceId)).toBeUndefined();
    });

    it('returns false for unknown service', () => {
      const result = registry.unregisterService('unknown-id');
      expect(result).toBe(false);
    });

    it('appends unregister record to NDJSON', () => {
      const serviceId = registry.registerService({
        name: 'to-remove',
        kind: 'process',
        entryPoint: 'test.py',
      });

      registry.unregisterService(serviceId);

      const content = fs.readFileSync(registry.getRegistryPath(), 'utf-8');
      expect(content).toContain('service_unregister');
    });
  });

  describe('listServices', () => {
    beforeEach(async () => {
      await registry.init();
    });

    it('returns empty array when no services registered', () => {
      const services = registry.listServices();
      expect(services).toEqual([]);
    });

    it('returns all registered services', () => {
      registry.registerService({ name: 'svc-1', kind: 'process', entryPoint: 'a.py' });
      registry.registerService({ name: 'svc-2', kind: 'port', entryPoint: 'b.py' });
      registry.registerService({ name: 'svc-3', kind: 'composition', entryPoint: 'c.py' });

      const services = registry.listServices();

      expect(services.length).toBe(3);
      expect(services.map((s) => s.name)).toContain('svc-1');
      expect(services.map((s) => s.name)).toContain('svc-2');
      expect(services.map((s) => s.name)).toContain('svc-3');
    });
  });

  describe('getByTag', () => {
    beforeEach(async () => {
      await registry.init();
      registry.registerService({
        name: 'web-service',
        kind: 'port',
        entryPoint: 'web.py',
        tags: ['web', 'api'],
      });
      registry.registerService({
        name: 'worker-service',
        kind: 'process',
        entryPoint: 'worker.py',
        tags: ['worker', 'background'],
      });
      registry.registerService({
        name: 'api-worker',
        kind: 'process',
        entryPoint: 'api-worker.py',
        tags: ['worker', 'api'],
      });
    });

    it('returns services with matching tag', () => {
      const workers = registry.getByTag('worker');
      expect(workers.length).toBe(2);
      expect(workers.every((s) => s.tags.includes('worker'))).toBe(true);
    });

    it('returns empty array for unknown tag', () => {
      const services = registry.getByTag('nonexistent');
      expect(services).toEqual([]);
    });
  });

  describe('getByKind', () => {
    beforeEach(async () => {
      await registry.init();
      registry.registerService({ name: 'p1', kind: 'port', entryPoint: 'p1.py' });
      registry.registerService({ name: 'p2', kind: 'port', entryPoint: 'p2.py' });
      registry.registerService({ name: 'proc1', kind: 'process', entryPoint: 'proc1.py' });
      registry.registerService({ name: 'comp1', kind: 'composition', entryPoint: 'comp1.py' });
    });

    it('returns services of specified kind', () => {
      const ports = registry.getByKind('port');
      expect(ports.length).toBe(2);
      expect(ports.every((s) => s.kind === 'port')).toBe(true);
    });

    it('returns all process services', () => {
      const processes = registry.getByKind('process');
      expect(processes.length).toBe(1);
    });

    it('returns all composition services', () => {
      const compositions = registry.getByKind('composition');
      expect(compositions.length).toBe(1);
    });
  });

  describe('getByLineage', () => {
    beforeEach(async () => {
      await registry.init();
      registry.registerService({
        name: 'core-tui',
        kind: 'process',
        entryPoint: 'tui.py',
        lineage: 'core.tui',
      });
      registry.registerService({
        name: 'core-bus',
        kind: 'process',
        entryPoint: 'bus.py',
        lineage: 'core.bus',
      });
      registry.registerService({
        name: 'mcp-host',
        kind: 'process',
        entryPoint: 'host.py',
        lineage: 'mcp.host',
      });
    });

    it('returns services matching lineage prefix', () => {
      const coreServices = registry.getByLineage('core');
      expect(coreServices.length).toBe(2);
      expect(coreServices.every((s) => s.lineage.startsWith('core'))).toBe(true);
    });

    it('returns exact lineage match', () => {
      const tui = registry.getByLineage('core.tui');
      expect(tui.length).toBe(1);
      expect(tui[0].name).toBe('core-tui');
    });
  });

  describe('getOmegaMotifs', () => {
    beforeEach(async () => {
      await registry.init();
      registry.registerService({
        name: 'omega-service',
        kind: 'process',
        entryPoint: 'omega.py',
        omegaMotif: true,
      });
      registry.registerService({
        name: 'regular-service',
        kind: 'process',
        entryPoint: 'regular.py',
        omegaMotif: false,
      });
    });

    it('returns only omega motif services', () => {
      const omegas = registry.getOmegaMotifs();
      expect(omegas.length).toBe(1);
      expect(omegas[0].name).toBe('omega-service');
    });
  });

  describe('getByRing', () => {
    beforeEach(async () => {
      await registry.init();
      registry.registerService({
        name: 'kernel',
        kind: 'process',
        entryPoint: 'kernel.py',
        ring: Ring.Kernel,
      });
      registry.registerService({
        name: 'service',
        kind: 'process',
        entryPoint: 'service.py',
        ring: Ring.Service,
      });
      registry.registerService({
        name: 'user1',
        kind: 'process',
        entryPoint: 'user1.py',
        ring: Ring.User,
      });
      registry.registerService({
        name: 'user2',
        kind: 'process',
        entryPoint: 'user2.py',
        ring: Ring.User,
      });
    });

    it('returns services at specified ring', () => {
      const userServices = registry.getByRing(Ring.User);
      expect(userServices.length).toBe(2);
      expect(userServices.every((s) => s.ring === Ring.User)).toBe(true);
    });

    it('returns kernel ring services', () => {
      const kernelServices = registry.getByRing(Ring.Kernel);
      expect(kernelServices.length).toBe(1);
      expect(kernelServices[0].name).toBe('kernel');
    });
  });

  describe('canAccessService', () => {
    beforeEach(async () => {
      await registry.init();
      registry.registerService({
        id: 'kernel-svc',
        name: 'kernel',
        kind: 'process',
        entryPoint: 'kernel.py',
        ring: Ring.Kernel,
      });
      registry.registerService({
        id: 'user-svc',
        name: 'user',
        kind: 'process',
        entryPoint: 'user.py',
        ring: Ring.User,
      });
    });

    it('kernel can access all services', () => {
      expect(registry.canAccessService('kernel-svc', Ring.Kernel)).toBe(true);
      expect(registry.canAccessService('user-svc', Ring.Kernel)).toBe(true);
    });

    it('user cannot access kernel services', () => {
      expect(registry.canAccessService('kernel-svc', Ring.User)).toBe(false);
    });

    it('user can access user-level services', () => {
      expect(registry.canAccessService('user-svc', Ring.User)).toBe(true);
    });

    it('returns false for unknown service', () => {
      expect(registry.canAccessService('unknown', Ring.Kernel)).toBe(false);
    });
  });

  describe('startService', () => {
    beforeEach(async () => {
      await registry.init();

      // Create a dummy entry point file
      const scriptsDir = path.join(tmpDir, 'scripts');
      fs.mkdirSync(scriptsDir, { recursive: true });
      fs.writeFileSync(path.join(scriptsDir, 'test.py'), '# test', 'utf-8');
    });

    it('creates a running instance for valid service', async () => {
      const serviceId = registry.registerService({
        name: 'startable',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await registry.startService(serviceId);

      expect(instance).not.toBeNull();
      expect(instance!.status).toBe('running');
      expect(instance!.serviceId).toBe(serviceId);
    });

    it('returns null for unknown service', async () => {
      const instance = await registry.startService('unknown');
      expect(instance).toBeNull();
    });

    it('returns error instance for missing entry point', async () => {
      const serviceId = registry.registerService({
        name: 'missing-entry',
        kind: 'process',
        entryPoint: 'scripts/nonexistent.py',
      });

      const instance = await registry.startService(serviceId);

      expect(instance).not.toBeNull();
      expect(instance!.status).toBe('error');
      expect(instance!.error).toContain('Entry point not found');
    });

    it('uses port override when provided', async () => {
      const serviceId = registry.registerService({
        name: 'port-svc',
        kind: 'port',
        entryPoint: 'scripts/test.py',
        port: 8080,
      });

      const instance = await registry.startService(serviceId, 9090);

      expect(instance!.port).toBe(9090);
    });

    it('uses default port from service def', async () => {
      const serviceId = registry.registerService({
        name: 'port-svc',
        kind: 'port',
        entryPoint: 'scripts/test.py',
        port: 8080,
      });

      const instance = await registry.startService(serviceId);

      expect(instance!.port).toBe(8080);
    });

    it('persists instance to NDJSON', async () => {
      const serviceId = registry.registerService({
        name: 'persist-test',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      await registry.startService(serviceId);

      const content = fs.readFileSync(registry.getInstancesPath(), 'utf-8');
      expect(content).toContain('service_instance');
      expect(content).toContain(serviceId);
    });
  });

  describe('stopService', () => {
    beforeEach(async () => {
      await registry.init();

      const scriptsDir = path.join(tmpDir, 'scripts');
      fs.mkdirSync(scriptsDir, { recursive: true });
      fs.writeFileSync(path.join(scriptsDir, 'test.py'), '# test', 'utf-8');
    });

    it('stops a running instance', async () => {
      const serviceId = registry.registerService({
        name: 'stoppable',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await registry.startService(serviceId);
      const stopped = await registry.stopService(instance!.instanceId);

      expect(stopped).toBe(true);

      const updated = registry.getInstance(instance!.instanceId);
      expect(updated!.status).toBe('stopped');
      expect(updated!.health).toBe('stopped');
    });

    it('returns false for unknown instance', async () => {
      const stopped = await registry.stopService('unknown');
      expect(stopped).toBe(false);
    });

    it('persists stop action to NDJSON', async () => {
      const serviceId = registry.registerService({
        name: 'persist-stop',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await registry.startService(serviceId);
      await registry.stopService(instance!.instanceId);

      const content = fs.readFileSync(registry.getInstancesPath(), 'utf-8');
      expect(content).toContain('"action":"stopped"');
    });
  });

  describe('updateHealth', () => {
    beforeEach(async () => {
      await registry.init();

      const scriptsDir = path.join(tmpDir, 'scripts');
      fs.mkdirSync(scriptsDir, { recursive: true });
      fs.writeFileSync(path.join(scriptsDir, 'test.py'), '# test', 'utf-8');
    });

    it('updates health status', async () => {
      const serviceId = registry.registerService({
        name: 'health-test',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await registry.startService(serviceId);
      const result = registry.updateHealth(instance!.instanceId, 'healthy');

      expect(result).toBe(true);

      const updated = registry.getInstance(instance!.instanceId);
      expect(updated!.health).toBe('healthy');
      expect(updated!.lastHealthIso).toBeDefined();
    });

    it('sets error message when unhealthy', async () => {
      const serviceId = registry.registerService({
        name: 'unhealthy-test',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await registry.startService(serviceId);
      registry.updateHealth(instance!.instanceId, 'unhealthy', 'Connection refused');

      const updated = registry.getInstance(instance!.instanceId);
      expect(updated!.health).toBe('unhealthy');
      expect(updated!.error).toBe('Connection refused');
      expect(updated!.status).toBe('error');
    });

    it('sets status to stopped when health is stopped', async () => {
      const serviceId = registry.registerService({
        name: 'stopped-test',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await registry.startService(serviceId);
      registry.updateHealth(instance!.instanceId, 'stopped');

      const updated = registry.getInstance(instance!.instanceId);
      expect(updated!.status).toBe('stopped');
    });

    it('returns false for unknown instance', () => {
      const result = registry.updateHealth('unknown', 'healthy');
      expect(result).toBe(false);
    });
  });

  describe('checkHealth', () => {
    beforeEach(async () => {
      await registry.init();

      const scriptsDir = path.join(tmpDir, 'scripts');
      fs.mkdirSync(scriptsDir, { recursive: true });
      fs.writeFileSync(path.join(scriptsDir, 'test.py'), '# test', 'utf-8');
    });

    it('returns unknown for unknown instance', async () => {
      const health = await registry.checkHealth('unknown');
      expect(health).toBe('unknown');
    });

    it('returns healthy for running instance without health check', async () => {
      const serviceId = registry.registerService({
        name: 'no-health-check',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await registry.startService(serviceId);
      const health = await registry.checkHealth(instance!.instanceId);

      expect(health).toBe('healthy');
    });
  });

  describe('getRunningInstances', () => {
    beforeEach(async () => {
      await registry.init();

      const scriptsDir = path.join(tmpDir, 'scripts');
      fs.mkdirSync(scriptsDir, { recursive: true });
      fs.writeFileSync(path.join(scriptsDir, 'test.py'), '# test', 'utf-8');
    });

    it('returns running instances for a service', async () => {
      const serviceId = registry.registerService({
        name: 'multi-instance',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      await registry.startService(serviceId);
      await registry.startService(serviceId);

      const running = registry.getRunningInstances(serviceId);
      expect(running.length).toBe(2);
    });

    it('excludes stopped instances', async () => {
      const serviceId = registry.registerService({
        name: 'mixed-instances',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const inst1 = await registry.startService(serviceId);
      await registry.startService(serviceId);
      await registry.stopService(inst1!.instanceId);

      const running = registry.getRunningInstances(serviceId);
      expect(running.length).toBe(1);
    });
  });

  describe('getStats', () => {
    beforeEach(async () => {
      await registry.init();

      const scriptsDir = path.join(tmpDir, 'scripts');
      fs.mkdirSync(scriptsDir, { recursive: true });
      fs.writeFileSync(path.join(scriptsDir, 'test.py'), '# test', 'utf-8');
    });

    it('returns correct statistics', async () => {
      registry.registerService({
        name: 'port1',
        kind: 'port',
        entryPoint: 'scripts/test.py',
        lineage: 'core.web',
        omegaMotif: true,
      });
      registry.registerService({
        name: 'proc1',
        kind: 'process',
        entryPoint: 'scripts/test.py',
        lineage: 'core.worker',
      });
      registry.registerService({
        name: 'proc2',
        kind: 'process',
        entryPoint: 'scripts/test.py',
        lineage: 'mcp.host',
        omegaMotif: true,
      });

      const svcId = registry.listServices()[0].id;
      await registry.startService(svcId);

      const stats = registry.getStats();

      expect(stats.serviceCount).toBe(3);
      expect(stats.instanceCount).toBe(1);
      expect(stats.runningCount).toBe(1);
      expect(stats.byKind.port).toBe(1);
      expect(stats.byKind.process).toBe(2);
      expect(stats.byKind.composition).toBe(0);
      expect(stats.byLineage.core).toBe(2);
      expect(stats.byLineage.mcp).toBe(1);
      expect(stats.omegaMotifCount).toBe(2);
    });
  });

  describe('load', () => {
    beforeEach(async () => {
      await registry.init();
    });

    it('loads services from NDJSON file', async () => {
      // Register some services
      registry.registerService({ name: 'svc1', kind: 'process', entryPoint: 'a.py' });
      registry.registerService({ name: 'svc2', kind: 'port', entryPoint: 'b.py' });

      // Create new registry and load
      const newRegistry = createRegistry({ root: tmpDir });
      await newRegistry.init();
      await newRegistry.load();

      const services = newRegistry.listServices();
      expect(services.length).toBe(2);
      expect(services.map((s) => s.name)).toContain('svc1');
      expect(services.map((s) => s.name)).toContain('svc2');
    });

    it('loads instances from NDJSON file', async () => {
      const scriptsDir = path.join(tmpDir, 'scripts');
      fs.mkdirSync(scriptsDir, { recursive: true });
      fs.writeFileSync(path.join(scriptsDir, 'test.py'), '# test', 'utf-8');

      const serviceId = registry.registerService({
        name: 'load-test',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });
      await registry.startService(serviceId);

      // Create new registry and load
      const newRegistry = createRegistry({ root: tmpDir });
      await newRegistry.init();
      await newRegistry.load();

      const instances = newRegistry.listInstances();
      expect(instances.length).toBe(1);
    });
  });

  describe('createRegistry', () => {
    it('creates registry with string root', () => {
      const reg = createRegistry(tmpDir);
      expect(reg.getRoot()).toBe(tmpDir);
    });

    it('creates registry with config object', () => {
      const reg = createRegistry({
        root: tmpDir,
        actor: 'test-actor',
        healthTtlMs: 60000,
      });
      expect(reg.getRoot()).toBe(tmpDir);
    });
  });

  describe('VERSION', () => {
    it('exports version string', () => {
      expect(VERSION).toBe('0.1.0');
    });
  });

  describe('checkHealth with HTTP', () => {
    beforeEach(async () => {
      await registry.init();

      const scriptsDir = path.join(tmpDir, 'scripts');
      fs.mkdirSync(scriptsDir, { recursive: true });
      fs.writeFileSync(path.join(scriptsDir, 'test.py'), '# test', 'utf-8');
    });

    it('returns unhealthy when HTTP health check fails', async () => {
      const serviceId = registry.registerService({
        name: 'http-health',
        kind: 'port',
        entryPoint: 'scripts/test.py',
        healthCheck: 'http://localhost:99999/nonexistent',
      });

      const instance = await registry.startService(serviceId);
      const health = await registry.checkHealth(instance!.instanceId);

      expect(health).toBe('unhealthy');
    });

    it('returns unknown for non-HTTP health check', async () => {
      const serviceId = registry.registerService({
        name: 'cmd-health',
        kind: 'process',
        entryPoint: 'scripts/test.py',
        healthCheck: 'some-command --check',
      });

      const instance = await registry.startService(serviceId);
      const health = await registry.checkHealth(instance!.instanceId);

      expect(health).toBe('unknown');
    });

    it('returns stopped for non-running instance without health check', async () => {
      const serviceId = registry.registerService({
        name: 'stopped-health',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await registry.startService(serviceId);
      await registry.stopService(instance!.instanceId);

      const health = await registry.checkHealth(instance!.instanceId);
      expect(health).toBe('stopped');
    });
  });

  describe('refreshInstances', () => {
    beforeEach(async () => {
      await registry.init();

      const scriptsDir = path.join(tmpDir, 'scripts');
      fs.mkdirSync(scriptsDir, { recursive: true });
      fs.writeFileSync(path.join(scriptsDir, 'test.py'), '# test', 'utf-8');
    });

    it('checks health of all running instances', async () => {
      const svc1Id = registry.registerService({
        name: 'refresh-svc1',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });
      const svc2Id = registry.registerService({
        name: 'refresh-svc2',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      await registry.startService(svc1Id);
      await registry.startService(svc2Id);

      // refreshInstances calls checkHealth for running instances
      await registry.refreshInstances();

      // For services without health checks, checkHealth returns 'healthy' for running
      // but does not update the instance's health field (it returns early)
      // The test just verifies refreshInstances runs without error
      const instances = registry.listInstances();
      expect(instances.filter((i) => i.status === 'running').length).toBe(2);
    });

    it('skips stopped instances', async () => {
      const serviceId = registry.registerService({
        name: 'refresh-stopped',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await registry.startService(serviceId);
      await registry.stopService(instance!.instanceId);

      await registry.refreshInstances();

      const updated = registry.getInstance(instance!.instanceId);
      expect(updated!.health).toBe('stopped');
    });
  });

  describe('getStaleInstances', () => {
    beforeEach(async () => {
      await registry.init();

      const scriptsDir = path.join(tmpDir, 'scripts');
      fs.mkdirSync(scriptsDir, { recursive: true });
      fs.writeFileSync(path.join(scriptsDir, 'test.py'), '# test', 'utf-8');
    });

    it('returns instances with no lastHealthIso', async () => {
      const serviceId = registry.registerService({
        name: 'stale-test',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      await registry.startService(serviceId);

      const stale = registry.getStaleInstances();
      expect(stale.length).toBe(1);
    });

    it('returns instances past TTL', async () => {
      // Create registry with very short TTL
      const shortTtlRegistry = createRegistry({
        root: tmpDir,
        healthTtlMs: 1,
      });
      await shortTtlRegistry.load();

      const serviceId = shortTtlRegistry.registerService({
        name: 'stale-ttl',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await shortTtlRegistry.startService(serviceId);
      shortTtlRegistry.updateHealth(instance!.instanceId, 'healthy');

      // Wait for TTL to expire
      await new Promise((r) => setTimeout(r, 10));

      const stale = shortTtlRegistry.getStaleInstances();
      expect(stale.length).toBe(1);
    });

    it('excludes stopped instances', async () => {
      const serviceId = registry.registerService({
        name: 'stale-stopped',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await registry.startService(serviceId);
      await registry.stopService(instance!.instanceId);

      const stale = registry.getStaleInstances();
      expect(stale.length).toBe(0);
    });

    it('excludes instances within TTL', async () => {
      // Create registry with very long TTL
      const longTtlRegistry = createRegistry({
        root: tmpDir,
        healthTtlMs: 1000000,
      });
      await longTtlRegistry.load();

      const serviceId = longTtlRegistry.registerService({
        name: 'not-stale',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await longTtlRegistry.startService(serviceId);
      longTtlRegistry.updateHealth(instance!.instanceId, 'healthy');

      const stale = longTtlRegistry.getStaleInstances();
      expect(stale.length).toBe(0);
    });
  });

  describe('edge cases', () => {
    beforeEach(async () => {
      await registry.init();
    });

    it('handles service with dependencies', async () => {
      const scriptsDir = path.join(tmpDir, 'scripts');
      fs.mkdirSync(scriptsDir, { recursive: true });
      fs.writeFileSync(path.join(scriptsDir, 'dep.py'), '# dep', 'utf-8');
      fs.writeFileSync(path.join(scriptsDir, 'main.py'), '# main', 'utf-8');

      const depId = registry.registerService({
        id: 'dependency',
        name: 'Dependency',
        kind: 'process',
        entryPoint: 'scripts/dep.py',
      });

      const mainId = registry.registerService({
        name: 'Main',
        kind: 'process',
        entryPoint: 'scripts/main.py',
        dependsOn: [depId],
      });

      // Starting main should auto-start dependency
      const instance = await registry.startService(mainId);

      expect(instance).not.toBeNull();
      expect(instance!.status).toBe('running');

      // Dependency should also be running
      const depInstances = registry.getRunningInstances(depId);
      expect(depInstances.length).toBe(1);
    });

    it('fails to start when dependency cannot start', async () => {
      registry.registerService({
        id: 'missing-dep',
        name: 'Missing Dependency',
        kind: 'process',
        entryPoint: 'scripts/nonexistent.py',
      });

      const mainId = registry.registerService({
        name: 'Main',
        kind: 'process',
        entryPoint: 'scripts/main.py',
        dependsOn: ['missing-dep'],
      });

      const instance = await registry.startService(mainId);
      expect(instance).toBeNull();
    });

    it('checkHealth returns unknown when service not found', async () => {
      const scriptsDir = path.join(tmpDir, 'scripts');
      fs.mkdirSync(scriptsDir, { recursive: true });
      fs.writeFileSync(path.join(scriptsDir, 'test.py'), '# test', 'utf-8');

      const serviceId = registry.registerService({
        name: 'orphan',
        kind: 'process',
        entryPoint: 'scripts/test.py',
      });

      const instance = await registry.startService(serviceId);

      // Remove the service (simulating external modification)
      registry.unregisterService(serviceId);

      const health = await registry.checkHealth(instance!.instanceId);
      expect(health).toBe('unknown');
    });
  });
});
