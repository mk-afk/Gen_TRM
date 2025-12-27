from enum import IntEnum

class EditAction(IntEnum):
    MOVE_LEFT   = 0
    MOVE_RIGHT  = 1
    DELETE      = 2
    INSERT      = 3
    REPLACE     = 4
    STOP        = 5
