class SenseKeyError(Exception):
    def __init__(self, message="Cannot find pos and lex for sense key"):
        self.message = message
        super().__init__(self.message)
