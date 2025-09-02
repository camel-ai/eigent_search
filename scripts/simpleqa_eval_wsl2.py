#!/usr/bin/env python3
"""
Wrapper script to run the evaluation in a headless environment.
This script sets environment variables to prevent X11/XCB issues in WSL2.
"""

import os
import sys
import subprocess


def main():
    # Set environment variables to prevent X11 issues
    env = os.environ.copy()

    # Disable X11 display
    env["DISPLAY"] = ""
    env["WAYLAND_DISPLAY"] = ""
    env["XDG_RUNTIME_DIR"] = ""

    # Set Playwright-specific environment variables
    env["PLAYWRIGHT_BROWSERS_PATH"] = os.path.expanduser("~/.cache/ms-playwright")
    env["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = "1"

    # Additional Chromium flags to prevent X11 issues
    env["CHROME_HEADLESS"] = "1"
    env["CHROME_NO_SANDBOX"] = "1"

    # Set environment variables for the subprocess
    for key, value in env.items():
        if key in [
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "XDG_RUNTIME_DIR",
            "PLAYWRIGHT_BROWSERS_PATH",
            "PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD",
            "CHROME_HEADLESS",
            "CHROME_NO_SANDBOX",
        ]:
            print(f"Setting {key}={value}")

    # Get the original command line arguments
    script_args = sys.argv[1:]

    if not script_args:
        print("Usage: python simpleqa_eval_wsl2.py <original_script_args>")
        print("Example: python simpleqa_eval_wsl2.py -a deep_search -n 5 -s 1")
        sys.exit(1)

    # Construct the command
    cmd = [sys.executable, "simpleqa_eval.py"] + script_args

    print(f"Running: {' '.join(cmd)}")
    print("Environment configured for headless operation...")

    try:
        # Run the original script with modified environment
        # Change to scripts directory to find simpleqa_eval.py
        scripts_dir = os.path.join(os.getcwd(), "scripts")
        result = subprocess.run(cmd, env=env, cwd=scripts_dir)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running script: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
