#!/usr/bin/env python3
import time
import os
import sys
import json
import collections
import argparse

# Configuration
BUS_DIR = os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus")
BUS_FILE = os.path.join(BUS_DIR, "events.ndjson")
MAX_LOG_LINES = 1000

class OmegaHeartMonitor:
    def __init__(self):
        self.running = True
        self.log_buffer = collections.deque(maxlen=MAX_LOG_LINES)
        self.seen_lines = set()
        self.metrics = {
            "agents": set(),
            "tasks_started": 0,
            "tasks_completed": 0,
            "dispatches": [],
            "responses": []
        }

    def tail_bus(self):
        """Reads new lines from the bus file."""
        if not os.path.exists(BUS_FILE):
            return

        try:
            with open(BUS_FILE, 'r') as f:
                f.seek(0, os.SEEK_END)
                f_size = f.tell()
                seek_point = max(0, f_size - 20000)
                f.seek(seek_point)
                
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line or line in self.seen_lines:
                        continue
                    
                    self.seen_lines.add(line)
                    
                    try:
                        entry = json.loads(line)
                        sender = entry.get('sender', entry.get('data', {}).get('sender', 'Unknown'))
                        topic = entry.get('topic', 'unknown')
                        
                        self.metrics["agents"].add(sender)
                        
                        if topic == 'task.dispatch':
                            self.metrics["tasks_started"] += 1
                            task_id = entry.get('data', {}).get('task_id', 'N/A')
                            target = entry.get('data', {}).get('target_agent', 'N/A')
                            self.metrics["dispatches"].append({'id': task_id, 'target': target})
                        elif topic == 'rd.tasks.response':
                            self.metrics["tasks_completed"] += 1
                            self.metrics["responses"].append(entry)
                    except:
                        pass
                    
                    self.log_buffer.append(line)

        except Exception as e:
            self.log_buffer.append(f"[SYSTEM ERROR]: {e}")

    def print_cli_status(self):
        """Prints a compact CLI status (non-curses mode)."""
        self.tail_bus()
        
        # ANSI colors for neon effect
        CYAN = '\033[96m'
        MAGENTA = '\033[95m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        # Clear screen and move cursor to top
        print('\033[2J\033[H', end='')
        
        # Header
        print(f"{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════════╗{RESET}")
        print(f"{CYAN}{BOLD}║              OMEGA HEART MONITOR (OHM) v1.1 - CLI Mode                   ║{RESET}")
        print(f"{CYAN}{BOLD}╠══════════════════════════════════════════════════════════════════════════╣{RESET}")
        
        # Metrics Row 1
        agents = list(self.metrics["agents"])
        print(f"{CYAN}║{RESET} {MAGENTA}AGENTS:{RESET} {len(agents)} active  {CYAN}│{RESET} {GREEN}DISPATCHES:{RESET} {self.metrics['tasks_started']}  {CYAN}│{RESET} {GREEN}RESPONSES:{RESET} {self.metrics['tasks_completed']}")
        
        # Metrics Row 2 - Agent List
        agent_str = ', '.join(agents[:5]) + ('...' if len(agents) > 5 else '')
        print(f"{CYAN}║{RESET} {YELLOW}Agents:{RESET} [{agent_str}]")
        
        print(f"{CYAN}╠══════════════════════════════════════════════════════════════════════════╣{RESET}")
        
        # Recent Dispatches
        print(f"{CYAN}║{RESET} {YELLOW}{BOLD}PENDING DISPATCHES:{RESET}")
        for d in self.metrics["dispatches"][-3:]:
            print(f"{CYAN}║{RESET}   → [{d['id']}] -> {d['target']}")
        
        # Recent Responses
        if self.metrics["responses"]:
            print(f"{CYAN}╠──────────────────────────────────────────────────────────────────────────╣{RESET}")
            print(f"{CYAN}║{RESET} {GREEN}{BOLD}RESPONSES:{RESET}")
            for r in self.metrics["responses"][-3:]:
                sender = r.get('sender', 'Unknown')
                rtype = r.get('data', {}).get('type', 'task_response')
                print(f"{CYAN}║{RESET}   ← [{GREEN}{sender}{RESET}] {rtype[:40]}")
        
        print(f"{CYAN}╠══════════════════════════════════════════════════════════════════════════╣{RESET}")
        
        # Event Log Tail
        print(f"{CYAN}║{RESET} {BOLD}EVENT LOG (last 3):{RESET}")
        for line in list(self.log_buffer)[-3:]:
            # Truncate long lines
            line = line[:75] + '...' if len(line) > 75 else line
            color = RESET
            if "ERROR" in line: color = RED
            elif "AUDIT" in line: color = MAGENTA
            elif "task.dispatch" in line: color = YELLOW
            elif "response" in line: color = GREEN
            print(f"{CYAN}║{RESET} {color}{line}{RESET}")
        
        print(f"{CYAN}╚══════════════════════════════════════════════════════════════════════════╝{RESET}")
        print(f"{CYAN}Press Ctrl+C to exit. Refreshing every 2s...{RESET}")

    def run_cli_mode(self, iterations=None):
        """Runs in CLI mode (non-curses)."""
        count = 0
        try:
            while self.running:
                self.print_cli_status()
                time.sleep(2)
                count += 1
                if iterations and count >= iterations:
                    break
        except KeyboardInterrupt:
            print("\nOHM Stopped.")

    def run_curses_mode(self):
        """Runs in full curses TUI mode."""
        import curses
        
        def main(stdscr):
            self.stdscr = stdscr
            self.maximized = False
            
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_MAGENTA, -1)
            curses.init_pair(3, curses.COLOR_GREEN, -1)
            curses.init_pair(4, curses.COLOR_RED, -1)
            curses.init_pair(5, curses.COLOR_YELLOW, -1)
            
            stdscr.nodelay(True)
            curses.curs_set(0)
            
            while self.running:
                stdscr.clear()
                
                try:
                    c = stdscr.getch()
                    if c == ord('q'):
                        self.running = False
                    elif c == ord('M'):
                        self.maximized = not self.maximized
                    elif c == 27:
                        self.maximized = False
                except:
                    pass

                self.tail_bus()
                
                rows, cols = stdscr.getmaxyx()
                info_height = 8 if not self.maximized else rows - 2
                
                # Draw info box
                try:
                    stdscr.addstr(0, 2, " OMEGA HEART MONITOR (v1.1) ", curses.color_pair(1) | curses.A_BOLD)
                    stdscr.addstr(1, 2, f"AGENTS: {len(self.metrics['agents'])}", curses.color_pair(2))
                    stdscr.addstr(2, 2, f"DISPATCHES: {self.metrics['tasks_started']}", curses.color_pair(3))
                    stdscr.addstr(2, 30, f"RESPONSES: {self.metrics['tasks_completed']}", curses.color_pair(3))
                    
                    # Log lines
                    y = 4
                    for line in list(self.log_buffer)[-5:]:
                        line = line[:cols-4] + '..' if len(line) > cols-4 else line
                        color = curses.color_pair(0)
                        if "ERROR" in line: color = curses.color_pair(4)
                        elif "dispatch" in line: color = curses.color_pair(5)
                        elif "response" in line: color = curses.color_pair(3)
                        stdscr.addstr(y, 2, line, color)
                        y += 1
                except:
                    pass
                
                stdscr.refresh()
                time.sleep(0.5)
        
        curses.wrapper(main)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omega Heart Monitor - Pluribus Observability")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (no curses)")
    parser.add_argument("--once", action="store_true", help="Print status once and exit")
    parser.add_argument("--iterations", type=int, help="Run for N iterations then exit")
    args = parser.parse_args()
    
    ohm = OmegaHeartMonitor()
    
    # Auto-detect: use CLI mode if not in a TTY
    if args.cli or args.once or not sys.stdout.isatty():
        if args.once:
            ohm.tail_bus()
            ohm.print_cli_status()
        else:
            ohm.run_cli_mode(iterations=args.iterations)
    else:
        try:
            ohm.run_curses_mode()
        except Exception as e:
            print(f"Curses mode failed ({e}), falling back to CLI mode...")
            ohm.run_cli_mode()
