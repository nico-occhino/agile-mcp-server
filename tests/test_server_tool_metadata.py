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

    input_guardrail_tool = tool_by_name["evaluate_input_prompt_guardrail"]

    assert input_guardrail_tool.annotations is not None
    assert input_guardrail_tool.annotations.readOnlyHint is True
    assert input_guardrail_tool.annotations.destructiveHint is False
    assert input_guardrail_tool.annotations.idempotentHint is True
    assert input_guardrail_tool.annotations.openWorldHint is False

    decode_auth_tool = tool_by_name["decode_jwt_auth_context"]
    authorize_auth_tool = tool_by_name["authorize_tool_access"]
    current_auth_tool = tool_by_name["current_jwt_auth_context"]

    for tool in [decode_auth_tool, authorize_auth_tool, current_auth_tool]:
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.openWorldHint is False
