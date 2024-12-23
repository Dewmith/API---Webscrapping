from fastapi_utils.camelcase import snake2camel
from pydantic import BaseConfig, BaseModel


def snake_2_camel(m: str) -> str:
    return snake2camel(m, True)


class CamelCase(BaseModel):
    class Config(BaseConfig):
        populate_by_name = True
        alias_generator = snake_2_camel


class GenericCamelCase(BaseModel):
    class Config(BaseConfig):
        populate_by_name = True
        alias_generator = snake_2_camel
