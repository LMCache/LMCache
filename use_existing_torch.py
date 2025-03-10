# SPDX-License-Identifier: Apache-2.0

import glob
import re

requires_files = glob.glob('requirements.txt')
requires_files += ["pyproject.toml"]
for file in requires_files:
    print(f">>> cleaning {file}")
    with open(file) as f:
        lines = f.readlines()
    if "torch" in "".join(lines).lower():
        print("removed:")
        with open(file, 'w') as f:
            for line in lines:
                if 'torch' not in line.lower():
                    f.write(line)
                else:
                    print(line.strip())
    print(f"<<< done cleaning {file}")
    print()

setup_file_path = "setup.py"
with open(setup_file_path, "r") as file:
    content = file.read()
modified_content = re.sub(
    r'"torch\s*==\s*[\d\.]+"',
    '"torch"',
    content
)

with open(setup_file_path, "w") as file:
    file.write(modified_content)

print("setup.py has been updated successfully!")

