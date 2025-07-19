class PushDownAutomataStack:
    """
    A stack for storing the pda conversion process.
    """
    def __init__(self):
        self.stack = []
    
    def push(self,cf_string:str, conversion: str):
        self.stack.append({
            "cf_string":cf_string,
            "conversion":conversion
        })

    def pop(self):
        if self.stack:
            return self.stack.pop()
        return None
    
    def peek(self):
        if self.stack:
            return self.stack[-1]
        return None

    def size(self):
        return len(self.stack)
    
    def is_empty(self):
        return len(self.stack) == 0
    
    def all_items(self):
        return self.stack
    
    