import sqlmodel

from ..config import get_cache_path

_cache_engine = sqlmodel.create_engine(f"sqlite:///{get_cache_path()}")


class Erc20TokenData(sqlmodel.SQLModel, table=True):
    id: int | None = sqlmodel.Field(default=None, primary_key=True)
    chain_id: int
    address: str
    name: str
    symbol: str
    decimals: int


class UniswapV2LiquidityPoolData(sqlmodel.SQLModel, table=True):
    id: int | None = sqlmodel.Field(default=None, primary_key=True)
    chain_id: int
    address: str
    factory: str
    token0: str
    token1: str
    fee_token0_numerator: int
    fee_token0_denominator: int
    fee_token1_numerator: int
    fee_token1_denominator: int


class UniswapV3LiquidityPoolData(sqlmodel.SQLModel, table=True):
    id: int | None = sqlmodel.Field(default=None, primary_key=True)
    chain_id: int
    address: str
    factory: str
    deployer: str
    token0: str
    token1: str
    fee: int


def create_database_and_tables() -> None:
    sqlmodel.SQLModel.metadata.create_all(_cache_engine)


def get_db_session() -> sqlmodel.Session:
    return sqlmodel.Session(_cache_engine)
