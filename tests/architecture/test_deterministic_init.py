# tests/architecture/test_deterministic_init.py

import os
import subprocess


def test_no_lazy_defaults_in_src():
    """
    Rule 5 Audit: Prohibit '.get(..., default)' in orchestration and common logic.
    Reason: Fallbacks hide configuration errors and break determinism.
    """
    # Regex to find .get() calls with a second argument (the default value)
    forbidden_pattern = r"\.get\(.*,.*\)"
    
    # We scan the root src/ and all subdirectories, but skip io/ and debug/ 
    # as they often deal with external environment fallbacks.
    base_src = "src"
    exclude_dirs = {"io", "debug", "__pycache__"}
    
    violations = []
    
    for root, dirs, files in os.walk(base_src):
        # Filter out excluded directories in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                result = subprocess.run(
                    ["grep", "-nE", forbidden_pattern, file_path],
                    capture_output=True, check=False,
                    text=True
                )
                if result.returncode == 0:
                    violations.append(f"{file_path}:\n{result.stdout}")

    assert not violations, (
        "🚨 DETERMINISTIC BREACH: Lazy defaults detected via .get()!\n"
        "Direct access is mandated to ensure explicit error reporting.\n"
        + "".join(violations)
    )