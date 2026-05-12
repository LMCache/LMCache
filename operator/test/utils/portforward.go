/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package utils

import (
	"fmt"
	"net"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// PortForwardSpec identifies what kubectl port-forward should target.
// Target follows the kubectl convention (e.g. "svc/my-cache",
// "pod/my-cache-abc", "deployment/my-cache").
type PortForwardSpec struct {
	Namespace string
	Target    string
}

// PortForward starts a `kubectl port-forward` subprocess and waits until
// the first local port begins accepting TCP connections. Each port arg
// follows kubectl syntax: "LOCAL:REMOTE" or just "PORT" (which uses the
// same port on both sides).
//
// Returns:
//   - closer: must be called to terminate the subprocess and free ports.
//     Safe to call multiple times.
//   - localBase: "http://127.0.0.1:<localport>" using the first port mapping.
//
// Deviation from the TMOP-18 sketch: namespace is passed via the spec
// struct rather than encoded into target, because kubectl requires
// namespace as a separate -n flag and silently ignores prefixes embedded
// in the target string.
func PortForward(spec PortForwardSpec, ports ...string) (func(), string, error) {
	if len(ports) == 0 {
		return nil, "", fmt.Errorf("PortForward: at least one port mapping is required")
	}
	localPort, err := localPortFromMapping(ports[0])
	if err != nil {
		return nil, "", err
	}

	args := []string{"port-forward"}
	if spec.Namespace != "" {
		args = append(args, "-n", spec.Namespace)
	}
	args = append(args, spec.Target)
	args = append(args, ports...)

	cmd := exec.Command("kubectl", args...)
	if err := cmd.Start(); err != nil {
		return nil, "", fmt.Errorf("start kubectl port-forward: %w", err)
	}

	closer := func() {
		// Killing the process is sufficient — kubectl port-forward
		// closes the listener on SIGKILL, freeing the local port.
		// We drop the Wait error because once we kill, the typical
		// exit status is "signal: killed" which is expected.
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	}

	if err := waitForLocalPort(localPort, 30*time.Second); err != nil {
		closer()
		return nil, "", fmt.Errorf("port-forward to %s/%s did not become ready: %w",
			spec.Namespace, spec.Target, err)
	}
	return closer, fmt.Sprintf("http://127.0.0.1:%d", localPort), nil
}

// localPortFromMapping extracts the LOCAL port from a kubectl mapping
// of the form "LOCAL:REMOTE" or just "PORT" (where the local and remote
// ports are equal).
func localPortFromMapping(mapping string) (int, error) {
	parts := strings.SplitN(mapping, ":", 2)
	p, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, fmt.Errorf("invalid port mapping %q: %w", mapping, err)
	}
	return p, nil
}

// waitForLocalPort polls 127.0.0.1:port until a TCP connection succeeds
// or timeout elapses. kubectl port-forward briefly accepts connections
// and immediately closes them once before the upstream is wired, so we
// also require the connection to stay open long enough to write to.
func waitForLocalPort(port int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return nil
		}
		time.Sleep(200 * time.Millisecond)
	}
	return fmt.Errorf("local port %d not reachable after %s", port, timeout)
}
