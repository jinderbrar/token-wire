#!/usr/bin/env python3
"""
TokenWire Setup Script

This script sets up the TokenWire project for first-time use:
1. Installs backend dependencies
2. Installs frontend dependencies
3. Detects available Ollama models
4. Generates dictionaries for each model
"""

import subprocess
import sys
import os
import json
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_success(msg):
    print(f"{Colors.GREEN}[OK] {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARNING] {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}[ERROR] {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}[INFO] {msg}{Colors.END}")

def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        return result.stdout.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stderr.strip(), e.returncode

def check_command_exists(cmd):
    """Check if a command exists in PATH."""
    try:
        subprocess.run(
            f"where {cmd}" if sys.platform == "win32" else f"which {cmd}",
            shell=True,
            capture_output=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False

def get_ollama_models():
    """Get list of installed Ollama models."""
    output, code = run_command("ollama list", check=False)
    if code != 0:
        return []

    models = []
    lines = output.strip().split('\n')
    for line in lines[1:]:  # Skip header
        if line.strip():
            parts = line.split()
            if parts:
                models.append(parts[0])  # Model name is first column
    return models

def setup_backend(project_root):
    """Install backend dependencies."""
    print_header("Setting up Backend")

    backend_dir = project_root / "backend"

    if not check_command_exists("uv"):
        print_warning("'uv' not found. Trying pip instead...")
        output, code = run_command("pip install -r requirements.txt", cwd=backend_dir)
        if code != 0:
            print_error(f"Failed to install backend dependencies: {output}")
            return False
    else:
        print_info("Installing backend dependencies with uv...")
        output, code = run_command("uv sync", cwd=backend_dir)
        if code != 0:
            print_error(f"Failed to install backend dependencies: {output}")
            return False

    print_success("Backend dependencies installed")
    return True

def setup_frontend(project_root):
    """Install frontend dependencies."""
    print_header("Setting up Frontend")

    frontend_dir = project_root / "frontend"

    if not check_command_exists("npm"):
        print_error("'npm' not found. Please install Node.js first.")
        return False

    print_info("Installing frontend dependencies with npm...")
    output, code = run_command("npm install", cwd=frontend_dir)
    if code != 0:
        print_error(f"Failed to install frontend dependencies: {output}")
        return False

    print_success("Frontend dependencies installed")
    return True

def generate_dictionaries(project_root, models):
    """Generate dictionaries for the given models."""
    print_header("Generating Dictionaries")

    scripts_dir = project_root / "scripts"
    extract_script = scripts_dir / "extract_vocab.py"

    if not extract_script.exists():
        print_error(f"Extract script not found: {extract_script}")
        return False

    # Check if transformers is available
    try:
        import transformers
        print_success("transformers library found")
    except ImportError:
        print_warning("transformers library not found. Installing...")
        output, code = run_command("pip install transformers", check=False)
        if code != 0:
            print_error("Failed to install transformers. Dictionary generation will fail.")
            print_info("Run: pip install transformers")
            return False

    success_count = 0
    for model in models:
        print_info(f"Generating dictionary for: {model}")
        output, code = run_command(
            f'python "{extract_script}" --model "{model}"',
            cwd=project_root,
            check=False
        )
        if code == 0:
            print_success(f"Dictionary generated for {model}")
            success_count += 1
        else:
            print_warning(f"Failed to generate dictionary for {model}")
            print(f"  {output[:200]}..." if len(output) > 200 else f"  {output}")

    if success_count > 0:
        print_success(f"Generated {success_count}/{len(models)} dictionaries")
        return True
    else:
        print_warning("No dictionaries were generated")
        return False

def main():
    print_header("TokenWire Setup")

    # Determine project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent

    print_info(f"Project root: {project_root}")

    # Check prerequisites
    print_header("Checking Prerequisites")

    prereqs_ok = True

    if check_command_exists("ollama"):
        print_success("Ollama found")
    else:
        print_error("Ollama not found. Please install Ollama first: https://ollama.ai")
        prereqs_ok = False

    if check_command_exists("python"):
        print_success("Python found")
    else:
        print_error("Python not found")
        prereqs_ok = False

    if check_command_exists("npm"):
        print_success("npm found")
    else:
        print_warning("npm not found. Frontend setup will be skipped.")

    if check_command_exists("uv"):
        print_success("uv found")
    else:
        print_warning("uv not found. Will use pip for backend.")

    # Get Ollama models
    print_header("Detecting Ollama Models")

    models = get_ollama_models()

    if models:
        print_success(f"Found {len(models)} model(s):")
        for model in models:
            print(f"  - {model}")
    else:
        print_warning("No Ollama models found!")
        print_info("Install a model with: ollama pull qwen2.5-coder:7b")
        print_info("Or: ollama pull gemma3:4b")

    # Setup backend
    if not setup_backend(project_root):
        print_error("Backend setup failed")
        return 1

    # Setup frontend
    if check_command_exists("npm"):
        if not setup_frontend(project_root):
            print_warning("Frontend setup failed, but continuing...")

    # Generate dictionaries
    if models:
        generate_dictionaries(project_root, models)
    else:
        print_warning("Skipping dictionary generation - no models found")
        print_info("After installing models, run:")
        print_info(f"  python scripts/extract_vocab.py --model <model_name>")

    # Final summary
    print_header("Setup Complete")

    print_info("To start the backend:")
    print(f"  cd {project_root / 'backend'}")
    print("  uv run uvicorn app.main:app --reload")
    print()
    print_info("To start the frontend:")
    print(f"  cd {project_root / 'frontend'}")
    print("  npm run dev")
    print()

    if not models:
        print_warning("Remember to install Ollama models and generate dictionaries!")
        print_info("  ollama pull qwen2.5-coder:7b")
        print_info("  python scripts/extract_vocab.py --model qwen2.5-coder:7b")

    return 0

if __name__ == "__main__":
    sys.exit(main())
