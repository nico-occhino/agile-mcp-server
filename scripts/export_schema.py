"""
scripts/export_schema.py
------------------------
Extract the generated tool schemas from the MCP server and save them
to a JSON file for documentation and debugging.
"""
import asyncio
import json
import os
from pathlib import Path

from server import mcp

async def export():
    tools = await mcp.list_tools()
    
    schema_dump = {
        "server_name": mcp.name,
        "tools": []
    }
    
    for fastmcp_tool in tools:
        t = fastmcp_tool.to_mcp_tool()
        schema_dump["tools"].append({
            "name": t.name,
            "description": t.description,
            "inputSchema": t.inputSchema,
            "annotations": t.annotations.model_dump(exclude_none=True) if t.annotations else None,
        })
        
    out_path = Path(__file__).parent.parent / "mcp_schema.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(schema_dump, f, indent=2)
        
    print(f"Successfully exported {len(tools)} tools to {out_path}")

if __name__ == "__main__":
    asyncio.run(export())
