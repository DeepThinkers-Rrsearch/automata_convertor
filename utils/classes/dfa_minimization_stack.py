class DfaMinimizationStack:
    """
    A stack for storing the dfa minimization process.
    """
    def __init__(self):
        self.stack = []
    
    def push(self,dfa:str, minimization: str):
        self.stack.append({
            "dfa":dfa,
            "minimization":minimization
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
    
    