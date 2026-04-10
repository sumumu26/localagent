# Importing each module triggers the @register decorators.
# To add a new tool: create agent/tools/my_tool.py with @register, then add it here.
from agent.tools import web_search, file_glob, file_read
