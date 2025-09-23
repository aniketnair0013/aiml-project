import subprocess
import sys
import os

def run_pytest():
    # Ensure we're in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Set PYTHONPATH to include src/ for imports
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([env.get('PYTHONPATH', ''), os.path.join(project_root, 'src')])
    
    # Check if pytest is installed
    try:
        subprocess.run([sys.executable, '-m', 'pytest', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: pytest not installed. Run 'pip install pytest' from project root.")
        return
    
    # Check if maps/ exists (for tests)
    if not os.path.exists('maps/small.map'):
        print("Error: Missing maps/small.map. Ensure maps/ directory and files are in place.")
        return
    
    # Run pytest with verbose output
    cmd = [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short']  # -v for verbose, --tb=short for clean errors
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    print("Pytest Output:")
    print(result.stdout)
    if result.returncode != 0:
        print("\nTests failed! Errors:")
        print(result.stderr)
    else:
        print("\nAll tests passed!")

if __name__ == "__main__":
    run_pytest()