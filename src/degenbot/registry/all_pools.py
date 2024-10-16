import contextlib

from eth_typing import ChecksumAddress
from eth_utils.address import to_checksum_address
from typing_extensions import Self

from ..exceptions import DegenbotValueError, RegistryAlreadyInitialized
from ..types import AbstractLiquidityPool, AbstractRegistry


class PoolRegistry(AbstractRegistry):
    instance: Self | None = None

    @classmethod
    def get_instance(cls) -> Self | None:
        return cls.instance

    def __init__(self) -> None:
        if self.instance is not None:
            raise RegistryAlreadyInitialized(
                "A registry has already been initialized. Access it using the get_instance() class method"  # noqa:E501
            )

        self._all_pools: dict[
            tuple[
                int,  # chain ID
                ChecksumAddress,  # pool address
            ],
            AbstractLiquidityPool,
        ] = dict()

    def get(self, pool_address: str, chain_id: int) -> AbstractLiquidityPool | None:
        return self._all_pools.get(
            (chain_id, to_checksum_address(pool_address)),
        )

    def add(self, pool_address: str, chain_id: int, pool: AbstractLiquidityPool) -> None:
        pool_address = to_checksum_address(pool_address)
        if self.get(pool_address=pool_address, chain_id=chain_id):
            raise DegenbotValueError("Pool is already registered")
        self._all_pools[(chain_id, pool_address)] = pool

    def remove(self, pool_address: str, chain_id: int) -> None:
        pool_address = to_checksum_address(pool_address)

        with contextlib.suppress(KeyError):
            del self._all_pools[(chain_id, pool_address)]


pool_registry = PoolRegistry()
