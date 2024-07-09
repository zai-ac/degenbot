# TODO: support unwinding updates for re-org


from io import TextIOWrapper
from typing import Any, Dict, List, TextIO, Tuple

import polars
import ujson
from eth_typing import ChecksumAddress
from eth_utils.address import to_checksum_address
from web3 import Web3
from web3._utils.events import get_event_data
from web3._utils.filters import construct_event_filter_params

from .. import config
from ..logging import logger
from .abi import UNISWAP_V3_POOL_ABI
from .v3_dataclasses import UniswapV3LiquidityEvent, UniswapV3PoolExternalUpdate
from .v3_liquidity_pool import (
    TICK_BITMAP_SCHEMA,
    TICK_DATA_SCHEMA,
    EMPTY_TICK_DATA,
    EMPTY_TICK_BITMAP,
)


class UniswapV3LiquiditySnapshot:
    """
    Retrieve and maintain liquidity positions for Uniswap V3 pools.
    """

    def __init__(
        self,
        file: TextIO | str,
        chain_id: int | None = None,
    ):
        file_handle: TextIOWrapper
        json_liquidity_snapshot: Dict[str, Any]

        match file:
            case TextIOWrapper():
                file_handle = file
                json_liquidity_snapshot = ujson.load(file)
            case str():
                with open(file) as file_handle:
                    json_liquidity_snapshot = ujson.load(file_handle)
            case _:  # pragma: no cover
                raise ValueError(f"Unrecognized file type {type(file)}")

        self._chain_id = chain_id if chain_id is not None else config.get_web3().eth.chain_id

        self.newest_block = json_liquidity_snapshot.pop("snapshot_block")

        self._liquidity_snapshot: Dict[ChecksumAddress, Dict[str, polars.DataFrame]] = {
            to_checksum_address(pool_address): {
                "tick_bitmap": polars.DataFrame(
                    data={
                        "word": [
                            int(word)
                            for word, pool_bitmaps in pool_liquidity_snapshot["tick_bitmap"].items()
                        ],
                        "bitmap": [
                            str(pool_bitmaps["bitmap"])
                            for word, pool_bitmaps in pool_liquidity_snapshot["tick_bitmap"].items()
                        ],
                    },
                    schema=TICK_BITMAP_SCHEMA,
                ),
                "tick_data": polars.DataFrame(
                    data={
                        "tick": [
                            int(tick)
                            for tick, pool_tick_data in pool_liquidity_snapshot["tick_data"].items()
                        ],
                        "liquidityNet": [
                            str(pool_tick_data["liquidityNet"])
                            for tick, pool_tick_data in pool_liquidity_snapshot["tick_data"].items()
                        ],
                        "liquidityGross": [
                            str(pool_tick_data["liquidityGross"])
                            for tick, pool_tick_data in pool_liquidity_snapshot["tick_data"].items()
                        ],
                    },
                    schema=TICK_DATA_SCHEMA,
                ),
            }
            for pool_address, pool_liquidity_snapshot in json_liquidity_snapshot.items()
        }

        logger.info(
            f"Loaded LP snapshot: {len(json_liquidity_snapshot)} pools @ block {self.newest_block}"
        )

        self._liquidity_events: Dict[ChecksumAddress, List[UniswapV3LiquidityEvent]] = dict()

    def _add_pool_if_missing(self, pool_address: ChecksumAddress) -> None:
        try:
            self._liquidity_events[pool_address]
        except KeyError:
            self._liquidity_events[pool_address] = []

        try:
            self._liquidity_snapshot[pool_address]
        except KeyError:
            self._liquidity_snapshot[pool_address] = {
                "tick_bitmap": EMPTY_TICK_BITMAP,
                "tick_data": EMPTY_TICK_DATA,
            }

    def fetch_new_liquidity_events(
        self,
        to_block: int,
        span: int = 1000,
    ) -> None:
        def _process_log() -> Tuple[ChecksumAddress, UniswapV3LiquidityEvent]:
            decoded_event = get_event_data(config.get_web3().codec, event_abi, log)

            pool_address = to_checksum_address(decoded_event["address"])
            tx_index = decoded_event["transactionIndex"]
            liquidity_block = decoded_event["blockNumber"]
            liquidity = decoded_event["args"]["amount"] * (
                -1 if decoded_event["event"] == "Burn" else 1
            )
            tick_lower = decoded_event["args"]["tickLower"]
            tick_upper = decoded_event["args"]["tickUpper"]

            return pool_address, UniswapV3LiquidityEvent(
                block_number=liquidity_block,
                liquidity=liquidity,
                tick_lower=tick_lower,
                tick_upper=tick_upper,
                tx_index=tx_index,
            )

        logger.info(f"Updating snapshot from block {self.newest_block} to {to_block}")

        v3pool = Web3().eth.contract(abi=UNISWAP_V3_POOL_ABI)

        for event in [v3pool.events.Mint, v3pool.events.Burn]:
            logger.info(f"Processing {event.event_name} events")
            event_abi = event._get_event_abi()
            start_block = self.newest_block + 1

            while True:
                end_block = min(to_block, start_block + span - 1)

                _, event_filter_params = construct_event_filter_params(
                    event_abi=event_abi,
                    abi_codec=config.get_web3().codec,
                    fromBlock=start_block,
                    toBlock=end_block,
                )

                event_logs = config.get_web3().eth.get_logs(event_filter_params)

                for log in event_logs:
                    pool_address, liquidity_event = _process_log()

                    if liquidity_event.liquidity == 0:  # pragma: no cover
                        continue

                    self._add_pool_if_missing(pool_address)
                    self._liquidity_events[pool_address].append(liquidity_event)

                if end_block == to_block:
                    break
                else:
                    start_block = end_block + 1

        logger.info(f"Updated snapshot to block {to_block}")
        self.newest_block = to_block

    def get_new_liquidity_updates(self, pool_address: str) -> List[UniswapV3PoolExternalUpdate]:
        pool_address = to_checksum_address(pool_address)
        pool_updates = self._liquidity_events.get(pool_address, list())
        self._liquidity_events[pool_address] = list()

        # The V3LiquidityPool helper will reject liquidity events associated with a past block, so
        # they must be applied in chronological order
        sorted_events = sorted(
            pool_updates,
            key=lambda event: (event.block_number, event.tx_index),
        )

        return [
            UniswapV3PoolExternalUpdate(
                block_number=event.block_number,
                liquidity_change=(
                    event.liquidity,
                    event.tick_lower,
                    event.tick_upper,
                ),
            )
            for event in sorted_events
        ]

    def get_tick_bitmap(self, pool: ChecksumAddress | str) -> polars.DataFrame | None:
        pool_address = to_checksum_address(pool)

        try:
            return self._liquidity_snapshot[pool_address]["tick_bitmap"]
        except KeyError:
            return None

    def get_tick_data(self, pool: ChecksumAddress | str) -> polars.DataFrame | None:
        pool_address = to_checksum_address(pool)

        try:
            return self._liquidity_snapshot[pool_address]["tick_data"]
        except KeyError:
            return None

    def update_snapshot(
        self,
        pool: ChecksumAddress | str,
        tick_data: polars.DataFrame,
        tick_bitmap: polars.DataFrame,
    ) -> None:
        pool_address = to_checksum_address(pool)

        self._add_pool_if_missing(pool_address)

        self._liquidity_snapshot[pool_address]["tick_bitmap"] = self._liquidity_snapshot[
            pool_address
        ]["tick_bitmap"].update(
            other=tick_bitmap,
            left_on=["word"],
            right_on=["word"],
            how="full",
        )
        self._liquidity_snapshot[pool_address]["tick_data"] = self._liquidity_snapshot[
            pool_address
        ]["tick_data"].update(
            other=tick_data,
            left_on=["tick"],
            right_on=["tick"],
            how="full",
        )
