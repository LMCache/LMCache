# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import atexit
import os
import subprocess
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class PluginLauncher:
    def __init__(self, config, role, worker_count, worker_id):
        self.config = config
        self.role = role
        self.worker_count = worker_count
        self.worker_id = worker_id
        self.plugin_processes = []
        # Register cleanup handler
        atexit.register(self.stop_plugins)

    def launch_plugins(self):
        """Launch all configured plugins"""
        if not self.config.plugin_locations:
            return

        for loc in self.config.plugin_locations:
            self._launch_plugins(loc)

    def _launch_plugins(self, loc: str):
        """Launch plugins from specified location"""
        path = Path(loc)
        if not path.exists():
            logger.warning(f"Plugin location {loc} does not exist")
            return

        files = []
        if path.is_file():
            files = [path]
        elif path.is_dir():
            # Recursively find all .py and .sh files
            for ext in ["*.py", "*.sh"]:
                files.extend(path.rglob(ext))

        for file in files:
            self._launch_plugin(file)

    def _should_skip_plugin(self, file: Path, parts: list[str]) -> bool:
        """Determine if plugin should be skipped based on role/worker ID"""
        if len(parts) < 2:
            return False

        # Check role match
        plugin_role = parts[0].upper()
        if plugin_role != "ALL" and plugin_role != self.role.name:
            logger.info(f"Skipping {file}: requires role {plugin_role}")
            return True

        # Check worker ID match
        if len(parts) > 2 and parts[1].isdigit():
            plugin_worker_id = int(parts[1])
            if plugin_worker_id != self.worker_id:
                logger.info(
                    f"worker {self.worker_id} is skipping plugin {file}, "
                    f"which is only for worker ID {plugin_worker_id}"
                )
                return True

        return False

    def _launch_plugin(self, file: Path):
        """Launch a plugin"""
        try:
            filename = file.stem.lower()
            parts = filename.split("_")

            if self._should_skip_plugin(file, parts):
                return

            # Get interpreter from first line (shebang)
            interpreter = self._get_interpreter(file)

            # Pass role and config as environment variables
            env = os.environ.copy()
            env["LMCACHE_PLUGIN_ROLE"] = str(self.role)
            env["LMCACHE_PLUGIN_CONFIG"] = self.config.to_json()
            env["LMCACHE_PLUGIN_WORKER_COUNT"] = str(self.worker_count)
            env["LMCACHE_PLUGIN_WORKER_ID"] = str(self.worker_id)

            proc = subprocess.Popen(
                [interpreter, str(file)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.plugin_processes.append(proc)
            logger.info(f"Launched plugin: {file} with {interpreter}")

            # Start thread to capture output continuously
            threading.Thread(
                target=self._capture_plugin_output, args=(proc, str(file)), daemon=True
            ).start()
        except Exception as e:
            logger.error(f"Failed to launch plugin {file}: {e}")

    def _get_interpreter(self, file: Path) -> str:
        """Get interpreter from first line comment"""
        try:
            with open(file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line.startswith("#!"):
                    # Extract interpreter path from shebang
                    return first_line[2:].strip()
        except Exception:
            pass

        # Fallback to default interpreters
        if file.suffix == ".py":
            return "python"
        elif file.suffix == ".sh":
            return "bash"
        else:
            raise ValueError(f"Unsupported plugin type: {file.suffix}")

    def _capture_plugin_output(self, proc: subprocess.Popen, plugin_name: str):
        """Continuously capture and log plugin output"""
        try:
            assert proc.stdout is not None, "The plugin subprocess does not have stdout"
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                logger.info(f"[{plugin_name}] {line.strip()}")

            proc.wait()
            logger.info(f"Plugin {plugin_name} exited with code {proc.returncode}")
        except Exception as e:
            logger.error(f"Error capturing output for {plugin_name}: {e}")

    def stop_plugins(self):
        """Terminate all plugin processes"""
        for proc in self.plugin_processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    logger.info(f"Terminated plugin process: {proc.pid}")
            except Exception as e:
                logger.error(f"Error terminating plugin process: {e}")
