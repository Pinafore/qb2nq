import language_tool_python
tool = language_tool_python.LanguageTool('en-US')

"""text = "Your the best but their are allso  good !"
matches = tool.check(text)
print(len(matches))
print(matches)"""

text = "You are the best but they are also good !"
matches = tool.check(text)
print(len(matches))
print(matches)
