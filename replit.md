modules = ["python-3.11", "python-3.10"]
run = "sh run.sh"

[nix]
channel = "stable-23_11"

[deployment]
run = ["sh", "run.sh"]

[languages.python]
version = "python3"

[languages.python.package-install]
package-manager = "pip"

[[ports]]
localPort = 8000
externalPort = 80