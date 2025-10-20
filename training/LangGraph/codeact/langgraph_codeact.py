
from langchain.tools import tool
import sys
from io import StringIO


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return the output.
    
    Args:
        code: Python code to execute
    """
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Execute code
        exec(code)
        output = sys.stdout.getvalue()
        return output or "Code executed successfully"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout