# tests/architecture/test_anti_density.py

import subprocess


def test_anti_density_axiom():
    """
    Axiom: No dense matrix conversions (.todense or .toarray) permitted.
    Reason: Memory scales at O(N^3); density is a critical contract violation.
    """
    forbidden_terms = [".todense()", ".toarray()"]
    target_dir = "src/step3" # Focus on the solver engine
    
    for term in forbidden_terms:
        # Use grep to search for violations
        result = subprocess.run(
            ["grep", "-r", term, target_dir], 
            capture_output=True, check=False, 
            text=True
        )
        
        assert result.returncode != 0, (
            f"🚨 SCALE GUARD BREACH: Found '{term}' in {target_dir}. "
            f"All operators must remain sparse to ensure O(N) scaling.\n"
            f"Details:\n{result.stdout}"
        )