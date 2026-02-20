#!/usr/bin/env python
"""
Check that dependencies are synchronized across packaging files.
Run this before releases to catch dependency mismatches.
"""

import re
import sys
from pathlib import Path


def extract_setup_py_deps(setup_path):
    """Extract install_requires from setup.py"""
    content = setup_path.read_text()
    match = re.search(r'install_requires=\[(.*?)\]', content, re.DOTALL)
    if not match:
        return set()
    deps_str = match.group(1)
    # Extract quoted strings
    deps = re.findall(r"['\"]([^'\"]+)['\"]", deps_str)
    # Remove version specifiers
    deps = [d.split('>=')[0].split('==')[0].split('<')[0].strip() for d in deps]
    return {d for d in deps if d and not d.startswith('#')}


def extract_requirements_txt(req_path):
    """Extract dependencies from requirements.txt"""
    if not req_path.exists():
        return set()
    deps = []
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            deps.append(line.split('>=')[0].split('==')[0].split('<')[0].strip())
    return set(deps)


def extract_meta_yaml_deps(meta_path):
    """Extract run dependencies from meta.yaml"""
    if not meta_path.exists():
        return set()
    content = meta_path.read_text()
    # Extract run section
    run_match = re.search(r'run:\s*\n(.*?)(?=\n\w+:|$)', content, re.DOTALL)
    if not run_match:
        return set()
    run_section = run_match.group(1)
    deps = []
    for line in run_section.splitlines():
        line = line.strip()
        if line.startswith('- ') and 'python' not in line:
            dep = line[2:].split('>=')[0].split('==')[0].split('<')[0].strip()
            # Normalize conda package names
            dep = dep.replace('matplotlib-base', 'matplotlib')
            dep = dep.replace('netcdf4', 'netCDF4')
            deps.append(dep)
    return set(deps)


def main():
    """Check dependency synchronization"""
    repo_root = Path(__file__).parent.parent
    
    setup_py = repo_root / "setup.py"
    requirements_txt = repo_root / "requirements.txt"
    meta_yaml = repo_root / "recipe" / "meta.yaml"
    
    print("Checking dependency synchronization...\n")
    
    setup_deps = extract_setup_py_deps(setup_py)
    requirements_deps = extract_requirements_txt(requirements_txt)
    meta_deps = extract_meta_yaml_deps(meta_yaml)
    
    print(f"setup.py install_requires ({len(setup_deps)} packages):")
    print(f"  {sorted(setup_deps)}\n")
    
    print(f"requirements.txt ({len(requirements_deps)} packages):")
    print(f"  {sorted(requirements_deps)}\n")
    
    print(f"recipe/meta.yaml run deps ({len(meta_deps)} packages):")
    print(f"  {sorted(meta_deps)}\n")
    
    # Check mismatches
    errors = []
    
    # Check setup.py vs requirements.txt
    missing_in_req = setup_deps - requirements_deps
    if missing_in_req:
        errors.append(f"⚠️  Missing in requirements.txt: {missing_in_req}")
    
    extra_in_req = requirements_deps - setup_deps
    if extra_in_req:
        errors.append(f"⚠️  Extra in requirements.txt: {extra_in_req}")
    
    # Check setup.py vs meta.yaml (excluding netCDF4 casing difference)
    missing_in_meta = setup_deps - meta_deps - {'netCDF4'}
    if missing_in_meta:
        errors.append(f"⚠️  Missing in recipe/meta.yaml: {missing_in_meta}")
    
    extra_in_meta = meta_deps - setup_deps
    if extra_in_meta:
        errors.append(f"⚠️  Extra in recipe/meta.yaml: {extra_in_meta}")
    
    print("-" * 60)
    if errors:
        print("\n❌ DEPENDENCY MISMATCH DETECTED:\n")
        for error in errors:
            print(error)
        print("\nPlease synchronize dependencies across all files before release.")
        return 1
    else:
        print("✅ All dependencies are synchronized!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
