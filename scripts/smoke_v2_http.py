"""Exercise real local Streamable HTTP and REST transports, then stop both servers."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time

import httpx
from fastmcp import Client

from v2.demo_token import make_demo_token
from v2.settings import Settings


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def main() -> None:
    hospital_port, mcp_port = free_port(), free_port()
    env = os.environ.copy()
    env["V2_DEMO_MODE"] = "true"
    env["V2_HOSPITAL_PORT"] = str(hospital_port)
    env["V2_MCP_PORT"] = str(mcp_port)
    env["V2_HOSPITAL_URL"] = f"http://127.0.0.1:{hospital_port}"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    processes = [
        subprocess.Popen([sys.executable, "-m", "v2.synthetic_api"], env=env, creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([sys.executable, "-m", "v2.mcp_server"], env=env, creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
    ]
    try:
        hospital = f"http://127.0.0.1:{hospital_port}"
        for _ in range(100):
            try:
                if httpx.get(f"{hospital}/health", timeout=0.2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("Synthetic hospital did not start")
        token = make_demo_token(Settings(hospital_url=hospital, jwt_secret=env.get("V2_JWT_SECRET", "synthetic-demo-only-secret-change-for-real-deployment")))
        for _ in range(100):
            try:
                async with Client(f"http://127.0.0.1:{mcp_port}/mcp", auth=token) as client:
                    tools = await client.list_tools()
                    assert [tool.name for tool in tools] == ["execute_query"]
                    result = await client.call_tool("execute_query", {"query": "Come sta il paziente 101?"})
                    assert "101" in repr(result) and "unauthorized" not in repr(result)
                    print("real HTTP MCP to hospital API: ok")
                    return
            except (httpx.HTTPError, OSError, RuntimeError):
                if processes[1].poll() is not None:
                    raise RuntimeError("MCP process exited before accepting requests")
                await asyncio.sleep(0.1)
        raise RuntimeError("MCP HTTP request did not complete")
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    start = time.monotonic()
    asyncio.run(main())
    print(f"finished in {time.monotonic() - start:.1f}s")

