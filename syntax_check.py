import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
def syntax_checker(text):
    """text = "Your the best but their are allso  good !"
    matches = tool.check(text)
    print(len(matches))
    print(matches)"""
    matches = tool.check(text)
    #print(len(matches))
    #print("suggestion:",matches)
    return (tool.correct(text))
def is_quote_ok(s):
    stack = []
    for c in s:
        if c in [ '"', "`"]:
            if stack and stack[-1] == c:
                # this single-quote is close character
                stack.pop()
            else:
                # a new quote started
                stack.append(c)
        else:
            # ignore it
            pass

    return len(stack) == 0
