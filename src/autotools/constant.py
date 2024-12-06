from dataclasses import dataclass
from typing import List, Optional, Union


# BEGIN = "<python>"
# END = "</python>"

BEGIN = "```python"
END = "```"


@dataclass
class ExecResponse:
    code: Union[str, None] = None
    response: Union[str, None] = None
    state: Union[bool, None] = False
    urls: Optional[Union[List[str], None]] = None
