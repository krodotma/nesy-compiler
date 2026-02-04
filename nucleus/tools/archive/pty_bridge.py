#!/usr/bin/env python3
import os
import pty
import select
import subprocess
import sys
import termios
import tty
import time

def main():
    # Ensure tmux session exists
    subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "ensure_tmux.sh")])

    # Command to run
    cmd = ["tmux", "attach", "-t", "pluribus_dash"]

    # Create PTY
    pid, fd = pty.fork()

    if pid == 0:
        # Child: run tmux
        os.execvp(cmd[0], cmd)
    else:
        # Parent: bridge fd to stdin/stdout
        try:
            # Set stdin to raw mode? No, we are piping from Node.js, not a real terminal.
            # Node will send raw bytes.
            
            while True:
                r, w, e = select.select([sys.stdin.buffer, fd], [], [])
                
                if sys.stdin.buffer in r:
                    d = sys.stdin.buffer.read1(1024)
                    if not d:
                        break
                    os.write(fd, d)
                
                if fd in r:
                    d = os.read(fd, 1024)
                    if not d:
                        break
                    sys.stdout.buffer.write(d)
                    sys.stdout.buffer.flush()
        except OSError:
            pass
        finally:
            os.kill(pid, 15)

if __name__ == "__main__":
    main()
