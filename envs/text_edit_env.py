from actions import EditAction
from response_buffer import ResponseBuffer

def step(buffer: ResponseBuffer, action, token=None):
    buf = buffer.copy()

    if action == EditAction.MOVE_LEFT:
        buf.cursor = max(0, buf.cursor - 1)

    elif action == EditAction.MOVE_RIGHT:
        buf.cursor = min(len(buf.tokens), buf.cursor + 1)

    elif action == EditAction.DELETE and buf.tokens:
        if buf.cursor < len(buf.tokens):
            del buf.tokens[buf.cursor]

    elif action == EditAction.INSERT and token is not None:
        buf.tokens.insert(buf.cursor, token)
        buf.cursor += 1

    elif action == EditAction.REPLACE and token is not None:
        if buf.cursor < len(buf.tokens):
            buf.tokens[buf.cursor] = token

    return buf
