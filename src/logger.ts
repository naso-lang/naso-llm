import { LogEntry } from './types.js';

class Logger {
  private logs: Map<string, LogEntry[]> = new Map();
  private maxLogs = 500;
  private listeners: Map<string, (entry: LogEntry) => void> = new Map();

  constructor() {
    this.logs.set('main', []);
    this.logs.set('webgpu', []);
    this.logs.set('naso', []);
  }

  log(source: 'main' | 'webgpu' | 'naso', level: LogEntry['level'], message: string) {
    const entry: LogEntry = {
      timestamp: Date.now(),
      level,
      source,
      message,
    };

    const arr = this.logs.get(source) || [];
    arr.push(entry);
    if (arr.length > this.maxLogs) arr.shift();
    this.logs.set(source, arr);

    const listener = this.listeners.get(source);
    if (listener) listener(entry);

    // Also log to console
    const prefix = `[${source.toUpperCase()}]`;
    switch (level) {
      case 'error': console.error(prefix, message); break;
      case 'warn': console.warn(prefix, message); break;
      case 'debug': console.debug(prefix, message); break;
      default: console.log(prefix, message);
    }
  }

  info(source: 'main' | 'webgpu' | 'naso', message: string) { this.log(source, 'info', message); }
  success(source: 'main' | 'webgpu' | 'naso', message: string) { this.log(source, 'success', message); }
  warn(source: 'main' | 'webgpu' | 'naso', message: string) { this.log(source, 'warn', message); }
  error(source: 'main' | 'webgpu' | 'naso', message: string) { this.log(source, 'error', message); }
  debug(source: 'main' | 'webgpu' | 'naso', message: string) { this.log(source, 'debug', message); }

  getLogs(source: 'main' | 'webgpu' | 'naso'): LogEntry[] {
    return this.logs.get(source) || [];
  }

  subscribe(source: 'main' | 'webgpu' | 'naso', callback: (entry: LogEntry) => void) {
    this.listeners.set(source, callback);
  }

  unsubscribe(source: 'main' | 'webgpu' | 'naso') {
    this.listeners.delete(source);
  }

  clear(source: 'main' | 'webgpu' | 'naso') {
    this.logs.set(source, []);
  }
}

export const logger = new Logger();