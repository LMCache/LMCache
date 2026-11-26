// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use tempfile::TempDir;

fn fixture(source: &str) -> TempDir {
    let directory = tempfile::tempdir().unwrap();
    let module = directory.path().join("lmcache/cli");
    fs::create_dir_all(&module).unwrap();
    fs::write(directory.path().join("lmcache/__init__.py"), "").unwrap();
    fs::write(module.join("__init__.py"), "").unwrap();
    fs::write(module.join("main.py"), source).unwrap();
    directory
}

fn invoke(directory: &Path, arguments: &[&str]) -> (u32, Output) {
    let child = Command::new(env!("CARGO_BIN_EXE_lmcache-server"))
        .args(arguments)
        .current_dir(directory)
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id();
    (pid, child.wait_with_output().unwrap())
}

#[test]
fn embeds_selected_environment_and_preserves_arguments() {
    let directory = fixture(
        "import os, sys\nprint(os.getpid())\nprint(sys.executable)\n\
         print(sys.prefix)\nprint(repr(sys.argv[1:]))\n",
    );
    let (pid, output) = invoke(
        directory.path(),
        &["--host", "127.0.0.1", "--shm-name", "cache 空间; $HOME"],
    );
    assert!(output.status.success(), "{:?}", output);
    let stdout = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<_> = stdout.lines().collect();
    assert_eq!(lines[0], pid.to_string());
    assert_eq!(lines[1], env!("LMCACHE_PYTHON_EXECUTABLE"));
    let interpreter = Path::new(env!("LMCACHE_PYTHON_EXECUTABLE"));
    // A virtualenv's pyvenv.cfg must still govern site-package discovery.
    let prefix = interpreter.parent().unwrap().parent().unwrap();
    if prefix.join("pyvenv.cfg").exists() {
        assert_eq!(Path::new(lines[2]), prefix);
    }
    assert_eq!(
        lines[3],
        "['server', '--host', '127.0.0.1', '--shm-name', 'cache 空间; $HOME']"
    );
}

#[test]
fn preserves_python_exit_status() {
    for status in [0, 2, 37] {
        let directory = fixture(&format!("raise SystemExit({status})\n"));
        let (_, output) = invoke(directory.path(), &[]);
        assert_eq!(output.status.code(), Some(status));
    }
}

#[test]
fn reports_uncaught_python_exception() {
    let directory = fixture("raise RuntimeError('server bootstrap failed')\n");
    let (_, output) = invoke(directory.path(), &[]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("RuntimeError: server bootstrap failed"));
}
