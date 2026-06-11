from pydantic import BaseModel


class MicrosoftLoginIn(BaseModel):
    id_token: str
