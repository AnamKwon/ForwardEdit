"""MCP SDK1/2 names for the same public project tool implementations."""
from importlib.metadata import version
SDK_MAJOR=int(version('mcp').split('.')[0])
if SDK_MAJOR>=2:
 from mcp.server.mcpserver import MCPServer as FastMCP
else:
 from mcp.server.fastmcp import FastMCP

def schema(tool):
 return tool.input_schema if SDK_MAJOR>=2 else tool.inputSchema
