import pytest


@pytest.mark.asyncio
async def test_guardrail_tool_annotations_are_safe():
    from server import mcp

    tools = await mcp.list_tools()
    tool_by_name = {tool.name: tool.to_mcp_tool() for tool in tools}
    guardrail_tool = tool_by_name["evaluate_clinical_output_guardrail"]

    assert guardrail_tool.annotations is not None
    assert guardrail_tool.annotations.readOnlyHint is True
    assert guardrail_tool.annotations.destructiveHint is False
    assert guardrail_tool.annotations.idempotentHint is True
    assert guardrail_tool.annotations.openWorldHint is False
