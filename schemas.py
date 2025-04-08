from pydantic import BaseModel, Field

class Transaction(BaseModel):
    id: int
    montant: float = Field(gt=0)
    devise: str = Field(min_length=3, max_length=3)
    pays: str = Field(min_length=2, max_length=2)
    utilisateur_id: int