/**
 * Logic Core Verification Tests
 * Author: gemini_qa_1
 * Context: Verification & Excellence
 */

import { describe, it, expect } from 'vitest';
import { IntentRouter } from './IntentRouter';
import { CommandParser } from './CommandParser';

describe('Dialogos Intent Router (The Brain)', () => {
  
  it('detects slash commands', () => {
    expect(IntentRouter.route('/task fix bugs')).toBe('task');
    expect(IntentRouter.route('/fix main.ts')).toBe('mutation');
    expect(IntentRouter.route('/sota transformer')).toBe('reflection');
  });

  it('infers mutation from context clues', () => {
    expect(IntentRouter.route('refactor the user class')).toBe('mutation');
    expect(IntentRouter.route('change line 50')).toBe('mutation');
    expect(IntentRouter.route('add export to the module')).toBe('mutation');
  });

  it('infers reflection from deep questions', () => {
    expect(IntentRouter.route('why is the sky blue?')).toBe('reflection');
    expect(IntentRouter.route('analyze the performance')).toBe('reflection');
  });

  it('defaults to query for ambiguity', () => {
    expect(IntentRouter.route('hello world')).toBe('query');
  });
});

describe('Dialogos Command Parser (The Translator)', () => {

  it('parses basic text tasks', () => {
    const res = CommandParser.parse('buy milk', 'task');
    expect(res.intent).toBe('task');
    expect(res.structuredContent).toEqual({
      type: 'task',
      title: 'buy milk',
      status: 'todo',
      laneId: 'inbox'
    });
  });

  it('extracts flags', () => {
    const res = CommandParser.parse('/task buy milk --lane=urgent', 'task');
    expect(res.flags['lane']).toBe('urgent');
    expect(res.structuredContent).toMatchObject({
      laneId: 'urgent'
    });
  });

  it('extracts code blocks for mutation', () => {
    const input = 'fix this ```ts\nconsole.log("hi")\n```';
    const res = CommandParser.parse(input, 'mutation');
    expect(res.structuredContent).toEqual({
      type: 'code',
      language: 'ts',
      value: 'console.log("hi")'
    });
  });
});
