# Minimal image for the two hosted MeshCompute services (control-plane, gateway).
#
# NOT for the node daemon: `mesh node start` is meant to run directly on a
# contributor's machine (often with a GPU it self-detects), not inside this
# container — see the comment at the top of ../compose/docker-compose.yml.
#
# Build (context must be the repo root, since `pip install -e .` needs the
# whole tree pyproject.toml points `[tool.setuptools.packages.find]` at):
#   docker build -f infra/docker/Dockerfile -t meshcompute:latest .
#
# Run either service — CMD below is just the default, override at `docker run`
# or in docker-compose.yml's `command:`:
#   docker run --rm -p 8080:8080 meshcompute:latest
#   docker run --rm -p 8081:8081 meshcompute:latest \
#       uvicorn meshcompute_gateway.app:app --host 0.0.0.0 --port 8081

FROM python:3.12-slim

WORKDIR /app

# Only the source needed to install the package — never the gitignored local
# deploy config (deploy/nodes.local.yaml, deploy/identities/*.key.json) or any
# runtime state (var/, models weight caches). Those are mounted in at run time.
COPY pyproject.toml README.md ./
COPY packages ./packages
COPY apps ./apps
COPY node ./node
COPY models/manifests ./models/manifests
COPY deploy/nodes.example.yaml ./deploy/nodes.example.yaml

RUN pip install --no-cache-dir -e .

# var/ (sqlite state) and deploy/ (local config, node identity keys) are
# expected to be mounted volumes in production — see docker-compose.yml.
RUN mkdir -p var

EXPOSE 8080 8081

CMD ["uvicorn", "meshcompute_control.app:app", "--host", "0.0.0.0", "--port", "8080"]
