"""A disposable HTTP tracking service with a process startup handshake."""

import json
import os
import subprocess
import sys
from urllib.request import urlopen

import pytest

_HTTP_SERVER = r"""
import json
import socket
import traceback

try:
    import uvicorn
    from mlflow.server.fastapi_app import app
    from mlflow.server.handlers import initialize_backend_stores

    initialize_backend_stores()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if not self.started:
                raise RuntimeError("HTTP server did not start")
            print(json.dumps({"uri": "http://127.0.0.1:" + str(listener.getsockname()[1])}), flush=True)

    ReadyServer(uvicorn.Config(app, log_level="warning", access_log=False)).run(sockets=[listener])
except BaseException:
    print(json.dumps({"error": traceback.format_exc()}), flush=True)
    raise
"""


@pytest.fixture(scope="session")
def mlflow_http(tmp_path_factory):
    root = tmp_path_factory.mktemp("http-artifact-service")
    backend = "sqlite:///" + str(root / "tracking.sqlite")
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("MLFLOW_", "_MLFLOW_"))}
    environment.update(
        {
            "_MLFLOW_SERVER_FILE_STORE": backend,
            "_MLFLOW_SERVER_REGISTRY_STORE": backend,
            "_MLFLOW_SERVER_ARTIFACT_ROOT": "mlflow-artifacts:/",
            "_MLFLOW_SERVER_SERVE_ARTIFACTS": "true",
            "_MLFLOW_SERVER_ARTIFACT_DESTINATION": str(root / "artifacts"),
            "MLFLOW_SERVER_ALLOWED_HOSTS": "127.0.0.1:*",
        }
    )
    with (root / "server.log").open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", _HTTP_SERVER],
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        try:
            # EOF is the failure signal if the child exits before sending readiness.
            line = process.stdout.readline()
            if not line:
                process.wait()
                log.seek(0)
                pytest.fail("MLflow exited before readiness: " + log.read())
            handshake = json.loads(line)
            assert "error" not in handshake, handshake.get("error")
            uri = handshake["uri"]
            with urlopen(uri + "/health") as response:
                assert response.status == 200
            yield uri
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()
            process.stdout.close()
