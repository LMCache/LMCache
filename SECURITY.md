# Security Policy

The LMCache maintainers take security seriously. We appreciate the work of researchers who find and responsibly report vulnerabilities, and we commit to working with you openly and quickly.

**Please do not report security vulnerabilities through public GitHub issues, discussions, pull requests, or social media.**

## Reporting a Vulnerability

Report vulnerabilities privately using one of these channels:

1. **GitHub private vulnerability reporting (preferred).** Go to the [Security tab](https://github.com/LMCache/LMCache/security) of this repository and click **Report a vulnerability**. Your report is visible only to the maintainers.
2. **Email:** security@lmcache.ai. 

Please include as much of the following as you can:

- The affected component and version(s), commit, or branch
- The deployment mode (in-process, multiprocess, management services, etc.) and relevant configuration
- A description of the issue and its impact
- Steps to reproduce, or a proof of concept
- Any suggested fix or mitigation
- Whether, and how, you would like to be credited

If you aren't sure whether something is a security issue, report it privately anyway. We would rather triage a false alarm than have a real issue disclosed publicly.

## What to Expect

| Stage | Our target |
| --- | --- |
| Acknowledge your report | Within 2 business days |
| Initial triage and severity assessment | Within 7 days |
| Status updates | At least every 14 days until resolved |
| Fix for critical/high severity issues | Within 30 days where feasible |
| Fix for medium/low severity issues | Within 90 days |

If you don't hear back within 2 business days, please follow up by email. Reports should never get lost, and we want to know if one has.

## Coordinated Disclosure

We follow coordinated disclosure:

- We ask that you keep the vulnerability confidential until a fix is released, or for **90 days** from your report, whichever comes first. We may ask for a short extension for complex issues and will explain why.
- We will keep you informed of our progress and agree on a disclosure date with you.
- We will publish a GitHub Security Advisory and request a CVE when the fix is released. If you already have a CVE ID, let us know so we can reference it.
- Before public disclosure, we may notify major downstream users and distributors under embargo so they can prepare.
- If an issue is being actively exploited, we may disclose earlier and will coordinate with you on timing.

## Credit

We credit reporters in the security advisory and release notes unless you ask to remain anonymous.

## Supported Versions

Security fixes are applied to the latest release and the `dev` branch.

| Version | Supported |
| --- | --- |
| Latest release ([TBD: e.g. 0.5.x]) | Yes |
| `dev` branch | Yes |
| Older releases | No — please upgrade |

## Scope

In scope:

- Code in this repository, including the multiprocess transport, RPC, and management services
- Official container images and example deployment manifests published by this project

Out of scope:

- Vulnerabilities in third-party dependencies that do not affect LMCache (please report these upstream)
- Issues that require an attacker who already has local root access or full control of the host
- Deployments that deliberately disable documented security defaults, unless the documentation or examples recommend doing so

## Secure Deployment Guidance

Some LMCache components open network services. As a baseline:

- Keep transport and management services bound to `localhost` or a trusted cluster network.
- Restrict access to service ports with firewalls or network policies.
- Run LMCache as a non-root user.

See [TBD: link to security/deployment docs] for details.

## Past Advisories

Published advisories are listed on our [GitHub Security Advisories page](https://github.com/LMCache/LMCache/security/advisories).
