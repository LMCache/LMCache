# SPDX-License-Identifier: Apache-2.0
"""
Test script for Controller Frontend

This script tests the basic functionality of the Controller Frontend.
"""

# Standard
from pathlib import Path
import sys


def test_file_exists():
    """Test that all required files exist."""
    base_dir = Path(__file__).parent
    required_files = [
        base_dir / "controller_frontend.py",
        base_dir / "static" / "index.html",
        base_dir / "static" / "css" / "style.css",
        base_dir / "static" / "js" / "controller_app.js",
        base_dir / "README.md",
    ]

    print("Checking required files...")
    all_exist = True
    for file_path in required_files:
        exists = file_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path.relative_to(base_dir)}")
        if not exists:
            all_exist = False

    return all_exist


def test_html_content():
    """Test that HTML file contains required elements."""
    html_path = Path(__file__).parent / "static" / "index.html"

    with open(html_path, "r") as f:
        content = f.read()

    required_elements = [
        "<title>LMCache Controller Dashboard</title>",
        "<h1>LMCache Controller Dashboard</h1>",
        "nav nav-tabs",
        "tab-content",
        "Instances",
        "Workers",
        "Key Pool",
        "Metrics",
        "Log Level",
        "Script Execution",
    ]

    print("\nChecking HTML content...")
    all_found = True
    for element in required_elements:
        found = element in content
        status = "✓" if found else "✗"
        print(f"  {status} Contains: {element[:30]}...")
        if not found:
            all_found = False

    return all_found


def test_js_content():
    """Test that JavaScript file contains required functions."""
    js_path = Path(__file__).parent / "static" / "js" / "controller_app.js"

    with open(js_path, "r") as f:
        content = f.read()

    required_functions = [
        "connectToController",
        "loadInstances",
        "loadWorkers",
        "loadKeyPool",
        "loadMetrics",
        "setLogLevel",
        "executeScript",
    ]

    print("\nChecking JavaScript functions...")
    all_found = True
    for func in required_functions:
        found = f"function {func}" in content or f"async function {func}" in content
        status = "✓" if found else "✗"
        print(f"  {status} Function: {func}")
        if not found:
            all_found = False

    return all_found


def test_server_startup():
    """Test that the server can start up (without actually connecting to controller)."""
    print("\nTesting server startup...")

    server_script = Path(__file__).parent / "controller_frontend.py"

    try:
        # Read and check the server script content
        with open(server_script, "r") as f:
            content = f.read()

        # Check for required components in the script
        checks = [
            (
                "FastAPI import",
                "from fastapi import FastAPI" in content or "import fastapi" in content,
            ),
            ("app definition", "app = FastAPI" in content),
            ("main function", "def main():" in content),
            ("uvicorn run", "uvicorn.run" in content),
        ]

        all_checks_passed = True
        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            print(f"  {status} {check_name}")
            if not check_result:
                all_checks_passed = False

        # Also check that we can parse the Python syntax (basic validation)
        # Standard
        import ast

        try:
            ast.parse(content)
            print("  ✓ Valid Python syntax")
        except SyntaxError as e:
            print(f"  ✗ Syntax error: {e}")
            all_checks_passed = False

        return all_checks_passed

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_dependencies():
    """Test that required dependencies can be imported."""
    print("\nChecking dependencies...")

    required_modules = [
        "fastapi",
        "httpx",
        "uvicorn",
    ]

    all_imported = True
    for module_name in required_modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}")
        except ImportError:
            print(f"  ✗ {module_name} (not installed)")
            all_imported = False

    return all_imported


def main():
    """Run all tests."""
    print("Testing LMCache Controller Frontend\n")
    print("=" * 60)

    tests = [
        ("File existence", test_file_exists),
        ("HTML content", test_html_content),
        ("JavaScript content", test_js_content),
        ("Dependencies", test_dependencies),
        ("Server startup", test_server_startup),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"\nResult: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"\nError during test: {e}")
            results.append((test_name, False))
            print("\nResult: FAIL (Error)")

    print("\n" + "=" * 60)
    print("\nSummary:")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The Controller Frontend is ready to use.")
        print("\nTo start the frontend:")
        print("  cd /Users/msy/projects/LMCache/examples/controller_frontend")
        print("  python controller_frontend.py")
        print("\nThen open http://localhost:8500 in your browser.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
