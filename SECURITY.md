# Security Policy

## Supported versions

vlmscope is pre-1.0; only the latest released version receives fixes.

| Version | Supported |
| --- | --- |
| 0.x (latest) | yes |
| older | no |

## Reporting a vulnerability

vlmscope is a pure-Python library with NumPy as its only runtime dependency, so
the attack surface is small. Still, if you find a security issue — for example a
loader mishandling untrusted input — please report it privately rather than
opening a public issue.

Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository, or send a direct message to the maintainer.

Please include:

- a description of the issue and its impact,
- a minimal reproduction, and
- the version you tested.

You can expect an acknowledgement within a few days.
