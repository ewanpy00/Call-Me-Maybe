from typing import Dict, Any
from pydantic import BaseModel


class Funtion(BaseModel):
    name: str
    description: str
    params: Dict[str, Any]
