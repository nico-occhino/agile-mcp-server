"""Only public v2 MCP entrypoint: one secured execute_query tool."""

from __future__ import annotations

import httpx
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

from v2.llm import OpenAICompatibleGenerator
from v2.repository import HttpPatientRepository
from v2.service import QueryService
from v2.settings import Settings


def build_mcp(settings: Settings | None = None) -> FastMCP:
    config = settings or Settings.from_env()
    mcp = FastMCP(name="agile-v2-synthetic", instructions="Synthetic hospital data only. Use execute_query for all reads.")

    @mcp.tool()
    async def execute_query(query: str) -> dict:
        """Authorize, assess input, parse, validate, and execute a synthetic hospital read."""
        headers = get_http_headers(include={"authorization"})
        authorization = headers.get("authorization")
        async with httpx.AsyncClient(timeout=10.0) as client:
            repository = HttpPatientRepository(client, config.hospital_url)
            generator = None
            if config.llm_base_url and config.llm_api_key and config.llm_model:
                generator = OpenAICompatibleGenerator(client, config.llm_base_url, config.llm_api_key, config.llm_model)
            return await QueryService(repository, config, generator).execute(query, authorization)

    return mcp


if __name__ == "__main__":
    import os

    build_mcp().run(transport="http", host="0.0.0.0", port=int(os.getenv("V2_MCP_PORT", "8000")))

