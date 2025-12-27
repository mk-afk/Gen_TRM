class ResponseBuffer:
    def __init__(self, tokens, cursor=0):
        self.tokens = list(tokens)
        self.cursor = cursor

    def copy(self):
        return ResponseBuffer(self.tokens.copy(), self.cursor)

    def state(self):
        return {
            "tokens": self.tokens,
            "cursor": self.cursor
        }
