class enfaConversionStack:
    """
    A stack for storing the e-nfa conversion process.
    """
    def __init__(self):
        self.stack = []
    
    def push(self,enfa:str, conversion: str):
        self.stack.append({
            "e-nfa":enfa,
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
    
    