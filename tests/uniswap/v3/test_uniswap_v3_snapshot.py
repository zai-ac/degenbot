from typing import Dict

import polars
import pytest
from degenbot.config import set_web3
from degenbot.fork.anvil_fork import AnvilFork
from degenbot.uniswap.managers import UniswapV3LiquidityPoolManager
from degenbot.uniswap.v3_dataclasses import UniswapV3LiquidityEvent
from degenbot.uniswap.v3_liquidity_pool import (
    EMPTY_TICK_BITMAP,
    EMPTY_TICK_DATA,
    TICK_BITMAP_SCHEMA,
    TICK_DATA_SCHEMA,
)
from degenbot.uniswap.v3_snapshot import UniswapV3LiquiditySnapshot
from eth_typing import ChecksumAddress
from eth_utils.address import to_checksum_address
from web3 import Web3

EMPTY_SNAPSHOT_FILENAME = "tests/uniswap/v3/empty_v3_liquidity_snapshot.json"
EMPTY_SNAPSHOT_BLOCK = 12_369_620  # Uniswap V3 factory was deployed on the next block, so use this as the initial zero state


@pytest.fixture
def empty_snapshot(ethereum_full_node_web3) -> UniswapV3LiquiditySnapshot:
    set_web3(ethereum_full_node_web3)
    return UniswapV3LiquiditySnapshot(file=EMPTY_SNAPSHOT_FILENAME)


@pytest.fixture
def first_250_blocks_snapshot(
    fork_mainnet_archive: AnvilFork,
) -> UniswapV3LiquiditySnapshot:
    set_web3(fork_mainnet_archive.w3)
    snapshot = UniswapV3LiquiditySnapshot(file=EMPTY_SNAPSHOT_FILENAME)
    snapshot.fetch_new_liquidity_events(to_block=EMPTY_SNAPSHOT_BLOCK + 250, span=50)
    return snapshot


def test_create_snapshot_from_file_path(ethereum_full_node_web3: Web3):
    set_web3(ethereum_full_node_web3)
    UniswapV3LiquiditySnapshot(file=EMPTY_SNAPSHOT_FILENAME)


def test_create_snapshot_from_file_handle():
    with open(EMPTY_SNAPSHOT_FILENAME) as file:
        UniswapV3LiquiditySnapshot(file)


def test_fetch_liquidity_events_first_250_blocks(
    first_250_blocks_snapshot: UniswapV3LiquiditySnapshot,
    fork_mainnet_archive: AnvilFork,
):
    set_web3(fork_mainnet_archive.w3)

    # Liquidity snapshots for each pool will be empty, since they only reflect the starting
    # liquidity at the initial snapshot block
    for pool_address in [
        "0x6c6Bc977E13Df9b0de53b251522280BB72383700",
        "0x7BeA39867e4169DBe237d55C8242a8f2fcDcc387",
        "0xCBCdF9626bC03E24f779434178A73a0B4bad62eD",
        "0xC2e9F25Be6257c210d7Adf0D4Cd6E3E881ba25f8",
        "0x7858E59e0C01EA06Df3aF3D20aC7B0003275D4Bf",
    ]:
        pool_address = to_checksum_address(pool_address)
        assert first_250_blocks_snapshot._liquidity_snapshot[pool_address]["tick_bitmap"].equals(
            EMPTY_TICK_BITMAP
        )
        assert first_250_blocks_snapshot._liquidity_snapshot[pool_address]["tick_data"].equals(
            EMPTY_TICK_DATA
        )

    # Unprocessed events should be found for these pools
    assert first_250_blocks_snapshot._liquidity_events == {
        "0x1d42064Fc4Beb5F8aAF85F4617AE8b3b5B8Bd801": [
            UniswapV3LiquidityEvent(
                block_number=12369739,
                liquidity=383995753785830744,
                tick_lower=-50580,
                tick_upper=-36720,
                tx_index=33,
            )
        ],
        "0x6c6Bc977E13Df9b0de53b251522280BB72383700": [
            UniswapV3LiquidityEvent(
                block_number=12369760,
                liquidity=3964498619038659,
                tick_lower=-276330,
                tick_upper=-276320,
                tx_index=82,
            ),
            UniswapV3LiquidityEvent(
                block_number=12369823,
                liquidity=2698389804940873511,
                tick_lower=-276400,
                tick_upper=-276250,
                tx_index=19,
            ),
        ],
        "0x7BeA39867e4169DBe237d55C8242a8f2fcDcc387": [
            UniswapV3LiquidityEvent(
                block_number=12369811,
                liquidity=123809464957093,
                tick_lower=192200,
                tick_upper=198000,
                tx_index=255,
            )
        ],
        "0xCBCdF9626bC03E24f779434178A73a0B4bad62eD": [
            UniswapV3LiquidityEvent(
                block_number=12369821,
                liquidity=34399999543676,
                tick_lower=253320,
                tick_upper=264600,
                tx_index=17,
            ),
            UniswapV3LiquidityEvent(
                block_number=12369846,
                liquidity=2154941425,
                tick_lower=255540,
                tick_upper=262440,
                tx_index=119,
            ),
        ],
        "0xC2e9F25Be6257c210d7Adf0D4Cd6E3E881ba25f8": [
            UniswapV3LiquidityEvent(
                block_number=12369854,
                liquidity=80059851033970806503,
                tick_lower=-84120,
                tick_upper=-78240,
                tx_index=85,
            )
        ],
        "0x7858E59e0C01EA06Df3aF3D20aC7B0003275D4Bf": [
            UniswapV3LiquidityEvent(
                block_number=12369863,
                liquidity=21206360421978,
                tick_lower=-10,
                tick_upper=10,
                tx_index=43,
            )
        ],
    }


def test_get_new_liquidity_updates(
    first_250_blocks_snapshot: UniswapV3LiquiditySnapshot,
    fork_mainnet_archive: AnvilFork,
):
    set_web3(fork_mainnet_archive.w3)

    for pool_address in [
        "0x1d42064Fc4Beb5F8aAF85F4617AE8b3b5B8Bd801",
        "0x6c6Bc977E13Df9b0de53b251522280BB72383700",
        "0x7BeA39867e4169DBe237d55C8242a8f2fcDcc387",
        "0xCBCdF9626bC03E24f779434178A73a0B4bad62eD",
        "0xC2e9F25Be6257c210d7Adf0D4Cd6E3E881ba25f8",
        "0x7858E59e0C01EA06Df3aF3D20aC7B0003275D4Bf",
    ]:
        first_250_blocks_snapshot.get_new_liquidity_updates(pool_address)
        assert first_250_blocks_snapshot._liquidity_events[to_checksum_address(pool_address)] == []


def test_apply_update_to_snapshot(
    empty_snapshot: UniswapV3LiquiditySnapshot,
    fork_mainnet_archive: AnvilFork,
):
    POOL_ADDRESS = "0xCBCdF9626bC03E24f779434178A73a0B4bad62eD"

    working_snapshot = empty_snapshot

    set_web3(fork_mainnet_archive.w3)

    tick_data = polars.DataFrame(
        data={
            "tick": [253320, 264600, 255540, 262440],
            "liquidityNet": [
                str(liq) for liq in [34399999543676, -34399999543676, 2154941425, -2154941425]
            ],
            "liquidityGross": [
                str(liq) for liq in [34399999543676, 34399999543676, 2154941425, 2154941425]
            ],
        },
        schema=TICK_DATA_SCHEMA,
    )
    tick_bitmap = polars.DataFrame(
        data={
            "word": [16, 17],
            "bitmap": [
                str(bitmap)
                for bitmap in [
                    11692013098732293937359713277596107809105402396672,
                    288230376155906048,
                ]
            ],
        },
        schema=TICK_BITMAP_SCHEMA,
    )

    # Test that the snapshot stores and returns the specified tick data and bitmap
    working_snapshot.update_snapshot(
        pool=POOL_ADDRESS,
        tick_data=tick_data,
        tick_bitmap=tick_bitmap,
    )
    _tick_data = working_snapshot.get_tick_data(POOL_ADDRESS)
    assert _tick_data is not None
    assert _tick_data.equals(tick_data)

    _tick_bitmap = working_snapshot.get_tick_bitmap(POOL_ADDRESS)
    assert _tick_bitmap is not None
    assert _tick_bitmap.equals(tick_bitmap)

    # Test that the pool manager injects the tick data and bitmap from the snapshot when building
    # a pool
    pool_manager = UniswapV3LiquidityPoolManager(
        factory_address="0x1F98431c8aD98523631AE4a59f267346ea31F984",
        chain_id=1,
        snapshot=working_snapshot,
    )
    pool = pool_manager.get_pool(POOL_ADDRESS)
    assert pool.tick_bitmap.equals(tick_bitmap)
    assert pool.tick_data.equals(tick_data)


def test_pool_manager_applies_snapshots(
    first_250_blocks_snapshot: UniswapV3LiquiditySnapshot,
    fork_mainnet_archive: AnvilFork,
):
    set_web3(fork_mainnet_archive.w3)

    # Build a pool manager to inject the liquidity events into the new pools as they are created
    pool_manager = UniswapV3LiquidityPoolManager(
        factory_address="0x1F98431c8aD98523631AE4a59f267346ea31F984",
        chain_id=1,
        snapshot=first_250_blocks_snapshot,
    )

    pool_creation_blocks: Dict[str, int] = {
        "0x1d42064Fc4Beb5F8aAF85F4617AE8b3b5B8Bd801": 12369739,
        "0x6c6Bc977E13Df9b0de53b251522280BB72383700": 12369760,
        "0x7BeA39867e4169DBe237d55C8242a8f2fcDcc387": 12369811,
        "0xCBCdF9626bC03E24f779434178A73a0B4bad62eD": 12369821,
        "0xC2e9F25Be6257c210d7Adf0D4Cd6E3E881ba25f8": 12369854,
        "0x7858E59e0C01EA06Df3aF3D20aC7B0003275D4Bf": 12369863,
    }

    # Check that the pending events were applied
    for pool_address in first_250_blocks_snapshot._liquidity_snapshot:
        pool = pool_manager.get_pool(
            pool_address,
            state_block=pool_creation_blocks[pool_address],
            v3liquiditypool_kwargs={"tick_bitmap": EMPTY_TICK_BITMAP, "tick_data": EMPTY_TICK_DATA},
        )

        match pool.address:
            case "0x1d42064Fc4Beb5F8aAF85F4617AE8b3b5B8Bd801":
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -50580)
                        .select("liquidityNet")
                        .item()
                    )
                    == 383995753785830744
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -50580)
                        .select("liquidityGross")
                        .item()
                    )
                    == 383995753785830744
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -36720)
                        .select("liquidityNet")
                        .item()
                    )
                    == -383995753785830744
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -36720)
                        .select("liquidityGross")
                        .item()
                    )
                    == 383995753785830744
                )

                assert (
                    int(pool.tick_bitmap.filter(polars.col("word") == -4).select("bitmap").item())
                    == 3064991081731777716716694054300618367237478244367204352
                )
                assert (
                    int(pool.tick_bitmap.filter(polars.col("word") == -3).select("bitmap").item())
                    == 91343852333181432387730302044767688728495783936
                )

            case "0x6c6Bc977E13Df9b0de53b251522280BB72383700":
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -276330)
                        .select("liquidityNet")
                        .item()
                    )
                    == 3964498619038659
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -276330)
                        .select("liquidityGross")
                        .item()
                    )
                    == 3964498619038659
                )

                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -276320)
                        .select("liquidityNet")
                        .item()
                    )
                    == -3964498619038659
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -276320)
                        .select("liquidityGross")
                        .item()
                    )
                    == 3964498619038659
                )

                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -276400)
                        .select("liquidityNet")
                        .item()
                    )
                    == 2698389804940873511
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -276400)
                        .select("liquidityGross")
                        .item()
                    )
                    == 2698389804940873511
                )

                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -276250)
                        .select("liquidityNet")
                        .item()
                    )
                    == -2698389804940873511
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -276250)
                        .select("liquidityGross")
                        .item()
                    )
                    == 2698389804940873511
                )

                assert (
                    int(pool.tick_bitmap.filter(polars.col("word") == -108).select("bitmap").item())
                    == 8487168
                )

            case "0x7BeA39867e4169DBe237d55C8242a8f2fcDcc387":
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 192200)
                        .select("liquidityNet")
                        .item()
                    )
                    == 123809464957093
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 192200)
                        .select("liquidityGross")
                        .item()
                    )
                    == 123809464957093
                )

                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 198000)
                        .select("liquidityNet")
                        .item()
                    )
                    == -123809464957093
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 198000)
                        .select("liquidityGross")
                        .item()
                    )
                    == 123809464957093
                )

                assert (
                    int(pool.tick_bitmap.filter(polars.col("word") == 3).select("bitmap").item())
                    == 6739986679341863419440115299426486514824618937839854009203971588096
                )

            case "0xCBCdF9626bC03E24f779434178A73a0B4bad62eD":
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 253320)
                        .select("liquidityNet")
                        .item()
                    )
                    == 34399999543676
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 253320)
                        .select("liquidityGross")
                        .item()
                    )
                    == 34399999543676
                )

                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 264600)
                        .select("liquidityNet")
                        .item()
                    )
                    == -34399999543676
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 264600)
                        .select("liquidityGross")
                        .item()
                    )
                    == 34399999543676
                )

                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 255540)
                        .select("liquidityNet")
                        .item()
                    )
                    == 2154941425
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 255540)
                        .select("liquidityGross")
                        .item()
                    )
                    == 2154941425
                )

                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 262440)
                        .select("liquidityNet")
                        .item()
                    )
                    == -2154941425
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 262440)
                        .select("liquidityGross")
                        .item()
                    )
                    == 2154941425
                )

                assert (
                    int(pool.tick_bitmap.filter(polars.col("word") == 16).select("bitmap").item())
                    == 11692013098732293937359713277596107809105402396672
                )
                assert (
                    int(pool.tick_bitmap.filter(polars.col("word") == 17).select("bitmap").item())
                    == 288230376155906048
                )

            case "0xC2e9F25Be6257c210d7Adf0D4Cd6E3E881ba25f8":
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -84120)
                        .select("liquidityNet")
                        .item()
                    )
                    == 80059851033970806503
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -84120)
                        .select("liquidityGross")
                        .item()
                    )
                    == 80059851033970806503
                )

                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -78240)
                        .select("liquidityNet")
                        .item()
                    )
                    == -80059851033970806503
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -78240)
                        .select("liquidityGross")
                        .item()
                    )
                    == 80059851033970806503
                )

                assert (
                    int(pool.tick_bitmap.filter(polars.col("word") == -6).select("bitmap").item())
                    == 6901746346790563787434755862298803523934049033832042530038157389332480
                )

            case "0x7858E59e0C01EA06Df3aF3D20aC7B0003275D4Bf":
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -10)
                        .select("liquidityNet")
                        .item()
                    )
                    == 21206360421978
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == -10)
                        .select("liquidityGross")
                        .item()
                    )
                    == 21206360421978
                )

                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 10)
                        .select("liquidityNet")
                        .item()
                    )
                    == -21206360421978
                )
                assert (
                    int(
                        pool.tick_data.filter(polars.col("tick") == 10)
                        .select("liquidityGross")
                        .item()
                    )
                    == 21206360421978
                )

                assert (
                    int(pool.tick_bitmap.filter(polars.col("word") == -1).select("bitmap").item())
                    == 57896044618658097711785492504343953926634992332820282019728792003956564819968
                )
                assert (
                    int(pool.tick_bitmap.filter(polars.col("word") == 0).select("bitmap").item())
                    == 2
                )

    # Check that the injected events were removed from the queue
    for pool_address in first_250_blocks_snapshot._liquidity_events:
        assert first_250_blocks_snapshot._liquidity_events[pool_address] == []
