# TODO: add event prototype exporter method and handler for callbacks

import dataclasses
import warnings
from bisect import bisect_left
from decimal import Decimal
from fractions import Fraction
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Mapping, cast

import polars
import sqlmodel
from eth_typing import ChecksumAddress
from eth_utils.address import to_checksum_address
from web3.contract.contract import Contract

from .. import config
from ..baseclasses import BaseLiquidityPool
from ..cache.database import UniswapV3LiquidityPoolData, get_db_session
from ..dex.uniswap import TICKLENS_ADDRESSES
from ..erc20_token import Erc20Token
from ..exceptions import (
    BitmapWordUnavailableError,
    EVMRevertError,
    ExternalUpdateError,
    InsufficientAmountOutError,
    LiquidityPoolError,
    NoPoolStateAvailable,
)
from ..logging import logger
from ..manager.token_manager import Erc20TokenHelperManager
from ..registry.all_pools import AllPools
from .abi import UNISWAP_V3_POOL_ABI
from .v3_dataclasses import (
    UniswapV3PoolExternalUpdate,
    UniswapV3PoolSimulationResult,
    UniswapV3PoolState,
    UniswapV3PoolStateUpdated,
)
from .v3_functions import exchange_rate_from_sqrt_price_x96, generate_v3_pool_address
from .v3_libraries import LiquidityMath, SwapMath, TickBitmap, TickMath
from .v3_libraries.functions import to_int256
from .v3_tick_lens import TickLens

TICK_DATA_SCHEMA = {
    "tick": polars.Int32,
    "liquidityNet": polars.String,  # Polars only supports up to 64-bit integers
    "liquidityGross": polars.String,  # Polars only supports up to 64-bit integers
}
TICK_BITMAP_SCHEMA = {
    "word": polars.Int16,
    "bitmap": polars.String,  # Polars only supports up to 64-bit integers
}
EMPTY_TICK_DATA = polars.DataFrame(
    data={
        "tick": [],
        "liquidityNet": [],
        "liquidityGross": [],
    },
    schema=TICK_DATA_SCHEMA,
)
EMPTY_TICK_BITMAP = polars.DataFrame(
    data={
        "word": [],
        "bitmap": [],
    },
    schema=TICK_BITMAP_SCHEMA,
)


class V3LiquidityPool(BaseLiquidityPool):
    # Holds a reference to a TickLens contract object. This is a singleton
    # contract so there is no need to create separate references for each pool.
    # Dict is keyed by a tuple of chain ID and factory address
    _lens_contracts: Dict[Tuple[int, ChecksumAddress], TickLens] = dict()

    uniswap_version = 3

    _TICKSPACING_BY_FEE = {
        100: 1,
        500: 10,
        2500: 50,  # Pancake V3
        3000: 60,
        10000: 200,
    }

    def __init__(
        self,
        address: str,
        fee: int | None = None,  # TODO: add deprecation warning
        lens: TickLens | None = None,
        tokens: List[Erc20Token] | None = None,  # TODO: add deprecation warning
        name: str = "",
        update_method: str | None = None,
        abi: List[Any] | None = None,
        factory_address: str | None = None,  # TODO: add deprecation warning
        factory_init_hash: str | None = None,
        deployer_address: str | None = None,
        init_hash: str | None = None,
        extra_words: int = 10,
        silent: bool = False,
        tick_data: Mapping[Any, Mapping[Any, Any]]
        | Dict[int, UniswapV3LiquidityAtTick]
        | polars.DataFrame
        | None = None,
        tick_bitmap: Mapping[Any, Mapping[Any, Any]]
        | Dict[int, UniswapV3BitmapAtWord]
        | polars.DataFrame
        | None = None,
        state_block: int | None = None,
        archive_states: bool = True,
    ):
        self.address = to_checksum_address(address)
        self.abi = abi if abi is not None else UNISWAP_V3_POOL_ABI

        w3 = config.get_web3()
        w3_contract = self._w3_contract

        if state_block is None:
            state_block = w3.eth.block_number
        self._update_block = state_block

        self.state: UniswapV3PoolState = UniswapV3PoolState(
            pool=self.address,
            liquidity=0,
            sqrt_price_x96=0,
            tick=0,
        )
        self.liquidity = w3_contract.functions.liquidity().call(block_identifier=state_block)
        self.sqrt_price_x96, self.tick, *_ = w3_contract.functions.slot0().call(
            block_identifier=state_block
        )

        self._fee: int
        self.factory: ChecksumAddress
        self.deployer: ChecksumAddress | None

        with get_db_session() as session:
            selection = sqlmodel.select(UniswapV3LiquidityPoolData).where(
                UniswapV3LiquidityPoolData.address == self.address,
                UniswapV3LiquidityPoolData.chain_id == w3.eth.chain_id,
            )
            cached_pool_data = session.exec(selection).first()

        if cached_pool_data:
            self.factory = cached_pool_data.factory
            token0_address = cached_pool_data.token0
            token1_address = cached_pool_data.token1
            self._fee = cached_pool_data.fee
            deployer_address = cached_pool_data.deployer
        else:
            self.factory = to_checksum_address(w3_contract.functions.factory().call())
            token0_address = to_checksum_address(w3_contract.functions.token0().call())
            token1_address = to_checksum_address(w3_contract.functions.token1().call())
            self._fee = w3_contract.functions.fee().call()
            self.factory = to_checksum_address(w3_contract.functions.factory().call())
            token0_address = to_checksum_address(w3_contract.functions.token0().call())
            token1_address = to_checksum_address(w3_contract.functions.token1().call())
            self._fee = w3_contract.functions.fee().call()

        if deployer_address:
            self.deployer = to_checksum_address(deployer_address)
        else:
            self.deployer = self.factory

        token_manager = Erc20TokenHelperManager(w3.eth.chain_id)
        self.token0 = token_manager.get_erc20token(
            address=token0_address,
            silent=silent,
        )
        self.token1 = token_manager.get_erc20token(
            address=token1_address,
            silent=silent,
        )
        self.tokens = (self.token0, self.token1)

        self._tick_spacing = self._TICKSPACING_BY_FEE[self._fee]  # immutable

        # held for operations that manipulate state data
        self._state_lock = Lock()

        self.init_hash: str | None = None
        if factory_init_hash is not None:  # pragma: no cover
            logger.warning(
                "factory_init_hash has been renamed to init_hash. It has been automatically converted, but will be removed in a future release. Provide the init_hash argument to remove this warning."
            )
            self.init_hash = factory_init_hash

        if lens:
            self.lens = lens
        else:
            # Use the singleton TickLens helper if available
            try:
                self.lens = self._lens_contracts[(w3.eth.chain_id, self.factory)]
            except KeyError:
                self.lens = TickLens(address=TICKLENS_ADDRESSES[w3.eth.chain_id][self.factory])
                self._lens_contracts[(w3.eth.chain_id, self.factory)] = self.lens

        if self.deployer is not None and init_hash is not None:
            computed_pool_address = generate_v3_pool_address(
                token_addresses=[self.token0.address, self.token1.address],
                fee=self._fee,
                factory_or_deployer_address=self.deployer,
                init_hash=init_hash,
            )
            if computed_pool_address != self.address:
                raise ValueError(
                    f"Pool address {self.address} does not match deterministic address {computed_pool_address} from factory"
                )

        if name:  # pragma: no cover
            self.name = name
        else:
            self.name = f"{self.token0}-{self.token1} (V3, {self._fee/10000:.2f}%)"

        if update_method is not None:  # pragma: no cover
            warnings.warn(
                "The `update_method` argument to `V3LiquidityPool()` is unused and otherwise ignored. Remove it to stop seeing this message."
            )

        # Default to an empty, sparse bitmap with no tick data
        self.tick_data = EMPTY_TICK_DATA
        self.tick_bitmap = EMPTY_TICK_BITMAP
        self._sparse_bitmap = True

        if (tick_bitmap is not None) != (tick_data is not None):
            raise ValueError(
                f"Must provide both tick_bitmap and tick_data! Got {tick_bitmap=}, {tick_data=}"
            )

        self._extra_words = extra_words
        if tick_bitmap is None and tick_data is None:
            word_position, _ = self._get_tick_bitmap_word_and_bit_position(self.tick)
            self.tick_bitmap, self.tick_data = self._fetch_tick_data_at_word(
                tick_bitmap=self.tick_bitmap,
                tick_data=self.tick_data,
                word_position=word_position,
                block_number=self._update_block,
            )
        else:
            self._sparse_bitmap = False

            match tick_bitmap:
                case polars.DataFrame():
                    self.tick_bitmap = tick_bitmap
                case dict() if tick_bitmap == {}:
                    self.tick_bitmap = EMPTY_TICK_BITMAP
                case dict():
                    _, first_value = list(tick_bitmap.items())[0]

                    match first_value:
                        case UniswapV3BitmapAtWord():
                            self.tick_bitmap = polars.DataFrame(
                                data={
                                    "word": tick_bitmap.keys(),
                                    "bitmap": [str(value.bitmap) for value in tick_bitmap.values()],
                                },
                                schema=TICK_BITMAP_SCHEMA,
                            )
                        case dict():
                            self.tick_bitmap = polars.DataFrame(
                                data={
                                    # cast() is used here to satisfy mypy, since pattern matching
                                    # can't narrow the type of all values used in a dict with
                                    # multiple keys
                                    "word": [int(word) for word in tick_bitmap.keys()],
                                    "bitmap": [
                                        str(value["bitmap"])
                                        for value in cast(
                                            Dict[str, Any],
                                            tick_bitmap,
                                        ).values()
                                    ],
                                },
                                schema=TICK_BITMAP_SCHEMA,
                            )
                        case _:
                            raise ValueError("Did not recognize bitmap format.")
                case _:
                    raise ValueError("Unknown tick_bitmap type.")

            match tick_data:
                case polars.DataFrame():
                    self.tick_data = tick_data
                case dict() if tick_data == {}:
                    self.tick_data = EMPTY_TICK_DATA
                case dict():
                    _, first_value = list(tick_data.items())[0]

                    match first_value:
                        case UniswapV3LiquidityAtTick():
                            self.tick_data = polars.DataFrame(
                                data={
                                    "tick": tick_data.keys(),
                                    "liquidityNet": [
                                        str(liquidity_at_tick.liquidityNet)
                                        for liquidity_at_tick in tick_data.values()
                                    ],
                                    "liquidityGross": [
                                        str(liquidity_at_tick.liquidityGross)
                                        for liquidity_at_tick in tick_data.values()
                                    ],
                                },
                                schema=TICK_DATA_SCHEMA,
                            )
                        case dict():
                            self.tick_data = polars.DataFrame(
                                data={
                                    # cast() here used to satisfy mypy, since pattern matching
                                    # can't narrow the type of all values used in a dict with
                                    # multiple keys
                                    "tick": [int(tick) for tick in tick_data.keys()],
                                    "liquidityNet": [
                                        str(liquidity_at_tick["liquidityNet"])
                                        for liquidity_at_tick in cast(
                                            Dict[Any, Any],
                                            tick_data,
                                        ).values()
                                    ],
                                    "liquidityGross": [
                                        str(liquidity_at_tick["liquidityGross"])
                                        for liquidity_at_tick in cast(
                                            Dict[Any, Any],
                                            tick_data,
                                        ).values()
                                    ],
                                },
                                schema=TICK_DATA_SCHEMA,
                            )
                        case _:
                            raise ValueError("Did not recognize liquidity data format.")
                case _:
                    raise ValueError("Unknown tick_data type")

        self._pool_state_archive: Dict[int, UniswapV3PoolState] = {}
        if archive_states:
            self._pool_state_archive[self._update_block] = self.state

        AllPools(w3.eth.chain_id)[self.address] = self

        self._subscribers = set()

        if not cached_pool_data:
            with get_db_session() as session:
                session.add(
                    UniswapV3LiquidityPoolData(
                        chain_id=w3.eth.chain_id,
                        address=self.address,
                        factory=self.factory,
                        deployer=self.deployer,
                        token0=self.token0.address,
                        token1=self.token1.address,
                        fee=self._fee,
                    )
                )
                session.commit()

        if not silent:  # pragma: no cover
            logger.info(self.name)
            logger.info(f"• Token 0: {self.token0}")
            logger.info(f"• Token 1: {self.token1}")
            logger.info(f"• Fee: {self._fee}")
            logger.info(f"• Liquidity: {self.liquidity}")
            logger.info(f"• SqrtPrice: {self.sqrt_price_x96}")
            logger.info(f"• Tick: {self.tick}")
            logger.info(f"• Tick Bitmap: {self.tick_bitmap}")
            logger.info(f"• Tick Data: {self.tick_data}")

    def __getstate__(self) -> Dict[str, Any]:
        # Remove objects that cannot be pickled and are unnecessary to perform
        # the calculation
        copied_attributes = ()
        # dropped_attributes = (
        #     "_state_lock",
        #     "_subscribers",
        #     "address",
        #     "lens",
        # )
        dropped_attributes = [
            "_extra_words",
            "_pool_state_archive",
            "_state_lock",
            "_subscribers",
            "_update_block",
            "abi",
            "deployer",
            "factory",
            "init_hash",
            "lens",
            "name",
        ]

        with self._state_lock:
            return {
                k: (v.copy() if k in copied_attributes else v)
                for k, v in self.__dict__.items()
                if k not in dropped_attributes
            }

    def __repr__(self) -> str:  # pragma: no cover
        return f"V3LiquidityPool(address={self.address}, token0={self.token0}, token1={self.token1}, fee={self._fee})"

    def __str__(self) -> str:
        return self.name

    def _calculate_swap(
        self,
        zero_for_one: bool,
        amount_specified: int,
        sqrt_price_limit_x96: int,
        override_start_liquidity: int | None = None,
        override_start_sqrt_price_x96: int | None = None,
        override_start_tick: int | None = None,
        override_tick_data: polars.DataFrame | None = None,
        override_tick_bitmap: polars.DataFrame | None = None,
    ) -> Tuple[int, int, int, int, int]:
        """
        This function is ported and adapted from the UniswapV3Pool.sol contract at
        https://github.com/Uniswap/v3-core/blob/main/contracts/UniswapV3Pool.sol

        Returns a tuple with amounts and final pool state values for a successful swap:
        (amount0, amount1, sqrt_price_x96, liquidity, tick)

        A negative amount indicates the token quantity sent to the swapper, and a positive amount
        indicates the token quantity deposited.
        """

        @dataclasses.dataclass(slots=True, eq=False)
        class SwapCache:
            liquidity_start: int
            tick_cumulative: int

        @dataclasses.dataclass(slots=True, eq=False)
        class SwapState:
            amount_specified_remaining: int
            amount_calculated: int
            sqrt_price_x96: int
            tick: int
            liquidity: int

        @dataclasses.dataclass(slots=True, eq=False)
        class StepComputations:
            sqrt_price_start_x96: int = 0
            tick_next: int = 0
            initialized: bool = False
            sqrt_price_next_x96: int = 0
            amount_in: int = 0
            amount_out: int = 0
            fee_amount: int = 0

        if amount_specified == 0:  # pragma: no cover
            raise EVMRevertError("AS")

        _liquidity = (
            override_start_liquidity if override_start_liquidity is not None else self.liquidity
        )
        _sqrt_price_x96 = (
            override_start_sqrt_price_x96
            if override_start_sqrt_price_x96 is not None
            else self.sqrt_price_x96
        )
        _tick = override_start_tick if override_start_tick is not None else self.tick
        _tick_bitmap = (
            override_tick_bitmap if override_tick_bitmap is not None else self.tick_bitmap
        )
        _tick_data = override_tick_data if override_tick_data is not None else self.tick_data

        if zero_for_one is True and not (
            sqrt_price_limit_x96 < _sqrt_price_x96
            and sqrt_price_limit_x96 > TickMath.MIN_SQRT_RATIO
        ):  # pragma: no cover
            raise EVMRevertError("SPL")

        if zero_for_one is False and not (
            sqrt_price_limit_x96 > _sqrt_price_x96
            and sqrt_price_limit_x96 < TickMath.MAX_SQRT_RATIO
        ):  # pragma: no cover
            raise EVMRevertError("SPL")

        exact_input = amount_specified > 0
        cache = SwapCache(liquidity_start=_liquidity, tick_cumulative=0)
        state = SwapState(
            amount_specified_remaining=amount_specified,
            amount_calculated=0,
            sqrt_price_x96=_sqrt_price_x96,
            tick=_tick,
            liquidity=cache.liquidity_start,
        )

        while (
            state.amount_specified_remaining != 0 and state.sqrt_price_x96 != sqrt_price_limit_x96
        ):
            step = StepComputations()
            step.sqrt_price_start_x96 = state.sqrt_price_x96

            while True:
                try:
                    (
                        step.tick_next,
                        step.initialized,
                    ) = TickBitmap.nextInitializedTickWithinOneWord(
                        _tick_bitmap,
                        state.tick,
                        self._tick_spacing,
                        zero_for_one,
                    )
                except BitmapWordUnavailableError as e:
                    missing_word = e.args[1]

                    if self._sparse_bitmap:
                        logger.debug(f"Fetching missing word {missing_word}")
                        _tick_bitmap, _tick_data = self._fetch_tick_data_at_word(
                            tick_bitmap=_tick_bitmap,
                            tick_data=_tick_data,
                            word_position=missing_word,
                        )
                    else:
                        # bitmap is complete, so mark word as empty
                        _tick_bitmap = _tick_bitmap.update(
                            other=polars.DataFrame(
                                data={
                                    "word": missing_word,
                                    "bitmap": "0",
                                },
                                schema=_tick_bitmap.schema,
                            ),
                            left_on=["word"],
                            right_on=["word"],
                            how="full",
                        )

                else:
                    # nextInitializedTickWithinOneWord will search up to 256 ticks away, which may
                    # return a tick in an adjacent word if there are no initialized ticks in the current word.
                    # This word may not be known to the helper, so check and fetch the containing word for this tick
                    tick_next_word, _ = self._get_tick_bitmap_word_and_bit_position(step.tick_next)

                    if self._sparse_bitmap and tick_next_word not in _tick_bitmap["word"]:
                        logger.debug(
                            f"tickNext={step.tick_next} out of range! Fetching word={tick_next_word}"
                            f"\n{self.name}"
                        )
                        _tick_bitmap, _tick_data = self._fetch_tick_data_at_word(
                            tick_bitmap=_tick_bitmap,
                            tick_data=_tick_data,
                            word_position=tick_next_word,
                        )

                    break

            # ensure that we do not overshoot the min/max tick, as the tick bitmap is not aware of these bounds
            if step.tick_next < TickMath.MIN_TICK:
                step.tick_next = TickMath.MIN_TICK
            elif step.tick_next > TickMath.MAX_TICK:
                step.tick_next = TickMath.MAX_TICK

            # get the price for the next tick
            step.sqrt_price_next_x96 = TickMath.getSqrtRatioAtTick(step.tick_next)

            # compute values to swap to the target tick, price limit, or point where input/output amount is exhausted
            (
                state.sqrt_price_x96,
                step.amount_in,
                step.amount_out,
                step.fee_amount,
            ) = SwapMath.computeSwapStep(
                state.sqrt_price_x96,
                sqrt_price_limit_x96
                if (
                    step.sqrt_price_next_x96 < sqrt_price_limit_x96
                    if zero_for_one
                    else step.sqrt_price_next_x96 > sqrt_price_limit_x96
                )
                else step.sqrt_price_next_x96,
                state.liquidity,
                state.amount_specified_remaining,
                self._fee,
            )

            if exact_input:
                state.amount_specified_remaining -= to_int256(step.amount_in + step.fee_amount)
                state.amount_calculated = to_int256(state.amount_calculated - step.amount_out)
            else:
                state.amount_specified_remaining += to_int256(step.amount_out)
                state.amount_calculated = to_int256(
                    state.amount_calculated + step.amount_in + step.fee_amount
                )

            # shift tick if we reached the next price
            if state.sqrt_price_x96 == step.sqrt_price_next_x96:  # pragma: no branch
                # if the tick is initialized, run the tick transition
                if step.initialized:
                    tick_next = step.tick_next
                    liquidityNet = int(
                        _tick_data.filter(polars.col("tick") == tick_next)
                        .select("liquidityNet")
                        .item()
                    )

                    if zero_for_one:
                        liquidityNet = -liquidityNet

                    state.liquidity = LiquidityMath.addDelta(state.liquidity, liquidityNet)

                state.tick = step.tick_next - 1 if zero_for_one else step.tick_next

            elif state.sqrt_price_x96 != step.sqrt_price_start_x96:  # pragma: no branch
                # recompute unless we're on a lower tick boundary (i.e. already transitioned ticks), and haven't moved
                state.tick = TickMath.getTickAtSqrtRatio(state.sqrt_price_x96)

        amount0, amount1 = (
            (
                amount_specified - state.amount_specified_remaining,
                state.amount_calculated,
            )
            if zero_for_one == exact_input
            else (
                state.amount_calculated,
                amount_specified - state.amount_specified_remaining,
            )
        )

        # Polars DataFrames are immutable, so store any changes to the attribute if overrides
        # were not provided
        if override_tick_bitmap is None:
            self.tick_bitmap = _tick_bitmap
        if override_tick_data is None:
            self.tick_data = _tick_data

        return amount0, amount1, state.sqrt_price_x96, state.liquidity, state.tick

    def _fetch_tick_data_at_word(
        self,
        tick_bitmap: polars.DataFrame,
        tick_data: polars.DataFrame,
        word_position: int,
        block_number: int | None = None,
    ) -> Tuple[polars.DataFrame, polars.DataFrame]:
        """
        Update the initialized tick values within a specified word position. A word is divided into
        256 ticks, spaced per the tickSpacing interval.

        Polars DataFrames are immutable, so the inputs cannot be mutated in place. Calling code
        must store the returned values.
        """

        _w3_contract = self._w3_contract

        if block_number is None:
            block_number = config.get_web3().eth.get_block_number()

        # bugfix: handle case where the pool is not yet deployed at this block, so proceed with an
        # empty bitmap and no tick data
        if not config.get_web3().eth.get_code(self.address, block_identifier=block_number):
            _tick_bitmap = 0
            _tick_data = []
        else:
            _tick_bitmap = _w3_contract.functions.tickBitmap(word_position).call(
                block_identifier=block_number,
            )
            _tick_data = self.lens._w3_contract.functions.getPopulatedTicksInWord(
                self.address, word_position
            ).call(block_identifier=block_number)

        updated_word = polars.DataFrame(
            data={
                "word": word_position,
                "bitmap": str(_tick_bitmap),
            },
            schema=tick_bitmap.schema,
        )
        tick_bitmap = tick_bitmap.update(
            other=updated_word,
            left_on=["word"],
            right_on=["word"],
            how="full",
        )

        for tick, liquidity_net, liquidity_gross in _tick_data:
            updated_tick_data = polars.DataFrame(
                data={
                    "tick": tick,
                    "liquidityNet": str(liquidity_net),
                    "liquidityGross": str(liquidity_gross),
                },
                schema=tick_data.schema,
            )
            tick_data = tick_data.update(
                other=updated_tick_data,
                left_on=["tick"],
                right_on=["tick"],
                how="full",
            )

        return tick_bitmap, tick_data

    def _get_tick_bitmap_word_and_bit_position(self, tick: int) -> Tuple[int, int]:
        """
        Retrieves the word and bit position (both zero indexed) for the tick. Accounts for the pool spacing.
        """
        return TickBitmap.position(int(Decimal(tick) // self._tick_spacing))

    @property
    def update_block(self) -> int:
        return self._update_block

    @property
    def liquidity(self) -> int:
        return self.state.liquidity

    @liquidity.setter
    def liquidity(self, new_liquidity: int) -> None:
        self.state = UniswapV3PoolState(
            pool=self.address,
            liquidity=new_liquidity,
            sqrt_price_x96=self.sqrt_price_x96,
            tick=self.tick,
            tick_bitmap=self.tick_bitmap,
            tick_data=self.tick_data,
        )

    @property
    def sqrt_price_x96(self) -> int:
        return self.state.sqrt_price_x96

    @sqrt_price_x96.setter
    def sqrt_price_x96(self, new_sqrt_price_x96: int) -> None:
        self.state = UniswapV3PoolState(
            pool=self.address,
            liquidity=self.liquidity,
            sqrt_price_x96=new_sqrt_price_x96,
            tick=self.tick,
            tick_bitmap=self.tick_bitmap,
            tick_data=self.tick_data,
        )

    @property
    def tick(self) -> int:
        return self.state.tick

    @tick.setter
    def tick(self, new_tick: int) -> None:
        self.state = UniswapV3PoolState(
            pool=self.address,
            liquidity=self.liquidity,
            sqrt_price_x96=self.sqrt_price_x96,
            tick=new_tick,
            tick_bitmap=self.tick_bitmap,
            tick_data=self.tick_data,
        )

    @property
    def tick_bitmap(self) -> polars.DataFrame:
        if TYPE_CHECKING:
            assert self.state.tick_bitmap is not None
        return self.state.tick_bitmap

    @tick_bitmap.setter
    def tick_bitmap(self, new_tick_bitmap: polars.DataFrame) -> None:
        self.state = UniswapV3PoolState(
            pool=self.address,
            liquidity=self.liquidity,
            sqrt_price_x96=self.sqrt_price_x96,
            tick=self.tick,
            tick_bitmap=new_tick_bitmap,
            tick_data=self.tick_data,
        )

    @property
    def tick_data(self) -> polars.DataFrame:
        if TYPE_CHECKING:
            assert self.state.tick_data is not None
        return self.state.tick_data

    @tick_data.setter
    def tick_data(self, new_tick_data: polars.DataFrame) -> None:
        self.state = UniswapV3PoolState(
            pool=self.address,
            liquidity=self.liquidity,
            sqrt_price_x96=self.sqrt_price_x96,
            tick=self.tick,
            tick_bitmap=self.tick_bitmap,
            tick_data=new_tick_data,
        )

    @property
    def _w3_contract(self) -> Contract:
        return config.get_web3().eth.contract(
            address=self.address,
            abi=self.abi,
        )

    def auto_update(
        self,
        block_number: int | None = None,
        silent: bool = True,
    ) -> Tuple[bool, UniswapV3PoolState]:
        """
        Retrieves the current slot0 and liquidity values from the LP, stores any that have changed,
        and returns a tuple with a status boolean indicating whether any update was found,
        and a dictionary holding current state values:
            - liquidity
            - sqrt_price_x96
            - tick

        Uses a lock to guard state-modifying methods that might cause race conditions
        when used with threads.
        """

        _w3_contract = self._w3_contract

        with self._state_lock:
            updated = False

            block_number = (
                config.get_web3().eth.get_block_number() if block_number is None else block_number
            )

            _sqrt_price_x96, _tick, *_ = _w3_contract.functions.slot0().call(
                block_identifier=block_number
            )
            _liquidity = _w3_contract.functions.liquidity().call(block_identifier=block_number)

            if self.sqrt_price_x96 != _sqrt_price_x96:
                updated = True
                self.sqrt_price_x96 = _sqrt_price_x96

            if self.tick != _tick:
                updated = True
                self.tick = _tick

            if self.liquidity != _liquidity:
                updated = True
                self.liquidity = _liquidity

            self._update_block = block_number

            if updated:
                self._notify_subscribers(
                    message=UniswapV3PoolStateUpdated(self.state),
                )
                if self._pool_state_archive:
                    self._pool_state_archive[block_number] = self.state

            if not silent:  # pragma: no cover
                logger.info(f"Liquidity: {self.liquidity}")
                logger.info(f"SqrtPriceX96: {self.sqrt_price_x96}")
                logger.info(f"Tick: {self.tick}")

        return updated, self.state

    def calculate_tokens_out_from_tokens_in(
        self,
        token_in: Erc20Token,
        token_in_quantity: int,
        override_state: UniswapV3PoolState | None = None,
    ) -> int:
        """
        This function implements the common degenbot interface `calculate_tokens_out_from_tokens_in`
        to calculate the number of tokens withdrawn (out) for a given number of tokens deposited (in).

        It is similar to calling quoteExactInputSingle using the quoter contract with arguments:
        `quoteExactInputSingle(
            tokenIn=token_in,
            tokenOut=[automatically determined by helper],
            fee=[automatically determined by helper],
            amountIn=token_in_quantity,
            sqrt_price_limitX96 = 0
        )` which returns the value `amountOut`

        Note that this wrapper function always assumes that the sqrt_price_limitx96 argument is unset,
        thus the swap calculation will continue until the target amount is satisfied, regardless of
        price impact.

        Accepts a dictionary of state values (`override_state`) to allow calculations beginning from an
        arbitrary starting point. This dictionary must have one or more of the following keys:
            - 'liquidity'
            - 'sqrt_price_x96'
            - 'tick'
            - 'tick_data'  (not yet implemented)
            - 'tick_bitmap'  (not yet implemented)
        """

        if token_in not in self.tokens:  # pragma: no cover
            raise ValueError("token_in not found!")

        # if override_state:
        #     logger.debug(f"V3 calc with overridden state: {override_state}")

        _is_zero_for_one = token_in == self.token0

        try:
            amount0_delta, amount1_delta, *_ = self._calculate_swap(
                zero_for_one=_is_zero_for_one,
                amount_specified=token_in_quantity,
                sqrt_price_limit_x96=(
                    TickMath.MIN_SQRT_RATIO + 1 if _is_zero_for_one else TickMath.MAX_SQRT_RATIO - 1
                ),
                override_start_liquidity=(
                    override_state.liquidity if override_state is not None else None
                ),
                override_start_sqrt_price_x96=(
                    override_state.sqrt_price_x96 if override_state is not None else None
                ),
                override_start_tick=(override_state.tick if override_state is not None else None),
                override_tick_bitmap=(
                    override_state.tick_bitmap if override_state is not None else self.tick_bitmap
                ),
                override_tick_data=(
                    override_state.tick_data if override_state is not None else self.tick_data
                ),
            )
        except EVMRevertError as e:  # pragma: no cover
            raise LiquidityPoolError(f"Simulated execution reverted: {e}") from e
        else:
            return -amount1_delta if _is_zero_for_one else -amount0_delta

    def calculate_tokens_in_from_tokens_out(
        self,
        token_out: Erc20Token,
        token_out_quantity: int,
        override_state: UniswapV3PoolState | None = None,
    ) -> int:
        """
        This function implements the common degenbot interface `calculate_tokens_in_from_tokens_out`
        to calculate the number of tokens deposited (in) for a given number of tokens withdrawn (out).

        It is similar to calling quoteExactOutputSingle using the quoter contract with arguments:
        `quoteExactOutputSingle(
            tokenIn=[automatically determined by helper],
            tokenOut=token_out,
            fee=[automatically determined by helper],
            amountOut=token_out_quantity,
            sqrtPriceLimitX96 = 0
        )` which returns the value `amountIn`

        Note that this wrapper function always assumes that the sqrtPriceLimitX96 argument is unset, thus the
        swap calculation will continue until the target amount is satisfied, regardless of price impact

        Accepts a dictionary of state values (`override_state`) to allow calculations beginning from an
        arbitrary starting point. This dictionary must have one or more of the following keys:
            - 'liquidity'
            - 'sqrt_price_x96'
            - 'tick'
            - 'tick_data'
            - 'tick_bitmap'
        """

        if token_out not in self.tokens:  # pragma: no cover
            raise ValueError("token_out not found!")

        _is_zero_for_one = token_out == self.token1

        try:
            amount0_delta, amount1_delta, *_ = self._calculate_swap(
                zero_for_one=_is_zero_for_one,
                amount_specified=-token_out_quantity,
                sqrt_price_limit_x96=(
                    TickMath.MIN_SQRT_RATIO + 1 if _is_zero_for_one else TickMath.MAX_SQRT_RATIO - 1
                ),
                override_start_liquidity=(
                    override_state.liquidity if override_state is not None else None
                ),
                override_start_sqrt_price_x96=(
                    override_state.sqrt_price_x96 if override_state is not None else None
                ),
                override_start_tick=(override_state.tick if override_state is not None else None),
                override_tick_bitmap=(
                    override_state.tick_bitmap if override_state is not None else self.tick_bitmap
                ),
                override_tick_data=(
                    override_state.tick_data if override_state is not None else self.tick_data
                ),
            )
        except EVMRevertError as e:  # pragma: no cover
            raise LiquidityPoolError(f"Simulated execution reverted: {e}") from e
        else:
            if _is_zero_for_one is True and -amount1_delta < token_out_quantity:
                raise InsufficientAmountOutError(
                    "Insufficient liquidity to swap for the requested amount."
                )
            if _is_zero_for_one is False and -amount0_delta < token_out_quantity:
                raise InsufficientAmountOutError(
                    "Insufficient liquidity to swap for the requested amount."
                )

            return amount0_delta if _is_zero_for_one else amount1_delta

    def external_update(
        self,
        update: UniswapV3PoolExternalUpdate,
        silent: bool = True,
    ) -> bool:
        """
        Process a `UniswapV3PoolExternalUpdate` with one or more of the following update types:
            - `block_number`: int
            - `tick`: int
            - `liquidity`: int
            - `sqrt_price_x96`: int
            - `liquidity_change`: tuple of (liquidity_delta, lower_tick, upper_tick). The delta can
                be positive or negative to indicate added or removed liquidity.

        `block_number` is validated against the most recently recorded block prior to recording any changes.

        If any update is processed, `self.state` and `self.update_block` are updated.

        Returns a bool indicating whether any updated state value was recorded.

        @dev This method uses a lock to guard state-modifying methods that might cause race conditions when used with threads.
        """

        if TYPE_CHECKING:
            assert isinstance(update, UniswapV3PoolExternalUpdate)

        if update.block_number < self._update_block:
            raise ExternalUpdateError(
                f"Rejected update for block {update.block_number} in the past, current update block is {self._update_block}"
            )

        with self._state_lock:
            updated_state = False

            if update.tick is not None and update.tick != self.tick:
                updated_state = True
                self.tick = update.tick

            if update.liquidity is not None and update.liquidity != self.liquidity:
                updated_state = True
                self.liquidity = update.liquidity

            if update.sqrt_price_x96 is not None and update.sqrt_price_x96 != self.sqrt_price_x96:
                updated_state = True
                self.sqrt_price_x96 = update.sqrt_price_x96

            if update.liquidity_change is not None and update.liquidity_change[0] != 0:
                updated_state = True
                liquidity_delta, lower_tick, upper_tick = update.liquidity_change

                # Adjust in-range liquidity if current tick is within the modified range
                if lower_tick <= self.tick < upper_tick:
                    liquidity_before = self.liquidity
                    self.liquidity += liquidity_delta
                    logger.debug(
                        f"Adjusting in-range liquidity, TX {update.tx}, tick range = [{lower_tick},{upper_tick}], current tick = {self.tick}, {self.address=}, previous liquidity = {liquidity_before}, liquidity change = {liquidity_delta}, current liquidity = {self.liquidity}"
                    )

                for tick in (lower_tick, upper_tick):
                    tick_word, _ = self._get_tick_bitmap_word_and_bit_position(tick)

                    # The tick bitmap must be known for the word prior to changing the initialized
                    # status of any tick
                    if tick_word not in self.tick_bitmap["word"]:
                        if self._sparse_bitmap:
                            self.tick_bitmap, self.tick_data = self._fetch_tick_data_at_word(
                                tick_bitmap=self.tick_bitmap,
                                tick_data=self.tick_data,
                                word_position=tick_word,
                                # Fetch the word using the previous block as a known "good" state snapshot
                                block_number=update.block_number - 1,
                            )

                        else:
                            # The bitmap is complete (sparse=False), so mark this word as empty
                            self.tick_bitmap = self.tick_bitmap.update(
                                other=polars.DataFrame(
                                    data={
                                        "word": tick_word,
                                        "bitmap": "0",
                                    },
                                    schema=self.tick_bitmap.schema,
                                ),
                                left_on=["word"],
                                right_on=["word"],
                                how="full",
                            )

                    if tick in self.tick_data["tick"]:
                        tick_liquidity_net = int(
                            self.tick_data.filter(polars.col("tick") == tick)
                            .select("liquidityNet")
                            .item()
                        )
                        tick_liquidity_gross = int(
                            self.tick_data.filter(polars.col("tick") == tick)
                            .select("liquidityGross")
                            .item()
                        )

                    else:
                        # if it doesn't exist, initialize the tick and set the current values to zero
                        tick_liquidity_net = 0
                        tick_liquidity_gross = 0
                        self.tick_bitmap = TickBitmap.flipTick(
                            self.tick_bitmap,
                            tick,
                            self._tick_spacing,
                            update_block=update.block_number,
                        )

                    # MINT: add liquidity at lower tick (i==0), subtract at upper tick (i==1)
                    # BURN: subtract liquidity at lower tick (i==0), add at upper tick (i==1)
                    # Same equation, but for BURN events the liquidity_delta value is negative
                    new_liquidity_net = (
                        tick_liquidity_net + liquidity_delta
                        if tick == lower_tick
                        else tick_liquidity_net - liquidity_delta
                    )
                    new_liquidity_gross = tick_liquidity_gross + liquidity_delta

                    if new_liquidity_gross == 0:
                        # Delete tick there is no remaining liquidity referencing it, and
                        # flip it in the bitmap
                        self.tick_data = self.tick_data.filter(polars.col("tick") != tick)
                        self.tick_bitmap = TickBitmap.flipTick(
                            self.tick_bitmap,
                            tick,
                            self._tick_spacing,
                            update_block=update.block_number,
                        )
                    else:
                        self.tick_data = self.tick_data.update(
                            other=polars.DataFrame(
                                data={
                                    "tick": tick,
                                    "liquidityNet": str(new_liquidity_net),
                                    "liquidityGross": str(new_liquidity_gross),
                                },
                                schema=self.tick_data.schema,
                            ),
                            left_on=["tick"],
                            right_on=["tick"],
                            how="full",
                        )

            if not silent:
                logger.debug(f"Liquidity: {self.liquidity}")
                logger.debug(f"SqrtPriceX96: {self.sqrt_price_x96}")
                logger.debug(f"Tick: {self.tick}")
                logger.debug(
                    f"liquidity event: {liquidity_delta} in tick range [{lower_tick},{upper_tick}], pool: {self.name}"
                    "\n"
                    f"old liquidity: {tick_liquidity_net} net, {tick_liquidity_gross} gross"
                    "\n"
                    f"new liquidity: {new_liquidity_net} net, {new_liquidity_gross} gross"
                )
                logger.debug(f"update block: {update.block_number} (last={self._update_block})")

            if updated_state:
                if self._pool_state_archive:
                    self._pool_state_archive[update.block_number] = self.state
                self._notify_subscribers(
                    message=UniswapV3PoolStateUpdated(self.state),
                )
                self._update_block = update.block_number

            return updated_state

    def get_absolute_price(
        self,
        token: Erc20Token,
        override_state: UniswapV3PoolState | None = None,
    ) -> Fraction:
        """
        Get the absolute price for the given token, expressed in units of the other.
        """

        return 1 / self.get_absolute_rate(token, override_state=override_state)

    def get_absolute_rate(
        self,
        token: Erc20Token,
        override_state: UniswapV3PoolState | None = None,
    ) -> Fraction:
        """
        Get the absolute rate of exchange for the given token, expressed in units of the other.
        """

        state = self.state if override_state is None else override_state

        if token == self.token0:
            return 1 / exchange_rate_from_sqrt_price_x96(state.sqrt_price_x96)
        elif token == self.token1:
            return exchange_rate_from_sqrt_price_x96(state.sqrt_price_x96)
        else:  # pragma: no cover
            raise ValueError(f"Unknown token {token}")

    def get_nominal_price(
        self,
        token: Erc20Token,
        override_state: UniswapV3PoolState | None = None,
    ) -> Fraction:
        """
        Get the nominal price for the given token, expressed in units of the other, corrected for
        decimal place values.
        """
        return 1 / self.get_nominal_rate(token, override_state=override_state)

    def get_nominal_rate(
        self,
        token: Erc20Token,
        override_state: UniswapV3PoolState | None = None,
    ) -> Fraction:
        """
        Get the nominal rate for the given token, expressed in units of the other, corrected for
        decimal place values.
        """

        state = self.state if override_state is None else override_state

        if token == self.token0:
            return (
                1
                / exchange_rate_from_sqrt_price_x96(state.sqrt_price_x96)
                * Fraction(10**self.token1.decimals, 10**self.token0.decimals)
            )
        elif token == self.token1:
            return exchange_rate_from_sqrt_price_x96(state.sqrt_price_x96) * Fraction(
                10**self.token0.decimals, 10**self.token1.decimals
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown token {token}")

    def restore_state_before_block(
        self,
        block: int,
    ) -> None:
        """
        Restore the last pool state recorded prior to a target block.

        Use this method to maintain consistent state data following a chain re-organization.
        """

        # Find the index for the most recent pool state PRIOR to the requested
        # block number.
        #
        # e.g. Calling restore_state_before_block(block=104) for a pool with
        # states at blocks 100, 101, 102, 103, 104. `bisect_left()` returns
        # block_index=3, since block 104 is at index=4. The state held at
        # index=3 is for block 103.
        # block_index = self._pool_state_archive.bisect_left(block)

        if not self._pool_state_archive:
            raise NoPoolStateAvailable("No archived states are available")

        with self._state_lock:
            known_blocks = list(self._pool_state_archive.keys())
            block_index = bisect_left(known_blocks, block)

            if block_index == 0:
                raise NoPoolStateAvailable(f"No pool state known prior to block {block}")

            # The last known state already meets the criterion, so return early
            if block_index == len(known_blocks):
                return

            # Remove states at and after the specified block
            for block in known_blocks[block_index:]:
                del self._pool_state_archive[block]

            restored_block, restored_state = list(self._pool_state_archive.items())[-1]

            self.state = restored_state
            self._update_block = restored_block

        self._notify_subscribers(
            message=UniswapV3PoolStateUpdated(self.state),
        )

    def simulate_exact_input_swap(
        self,
        token_in: Erc20Token,
        token_in_quantity: int,
        sqrt_price_limit_x96: int | None = None,
        override_state: UniswapV3PoolState | None = None,
    ) -> UniswapV3PoolSimulationResult:
        """
        Simulate an exact input swap.
        """

        if token_in not in self.tokens:  # pragma: no cover
            raise ValueError("token_in is unknown!")
        if token_in_quantity == 0:  # pragma: no cover
            raise ValueError("Zero input swap requested.")

        zero_for_one = token_in == self.token0

        try:
            (
                amount0_delta,
                amount1_delta,
                end_sqrt_price_x96,
                end_liquidity,
                end_tick,
            ) = self._calculate_swap(
                zero_for_one=zero_for_one,
                amount_specified=token_in_quantity,
                sqrt_price_limit_x96=(
                    sqrt_price_limit_x96
                    if sqrt_price_limit_x96 is not None
                    else (
                        TickMath.MIN_SQRT_RATIO + 1 if zero_for_one else TickMath.MAX_SQRT_RATIO - 1
                    )
                ),
                override_start_liquidity=override_state.liquidity if override_state else None,
                override_start_sqrt_price_x96=override_state.sqrt_price_x96
                if override_state
                else None,
                override_start_tick=override_state.tick if override_state else None,
                override_tick_bitmap=override_state.tick_bitmap if override_state else None,
                override_tick_data=override_state.tick_data if override_state else None,
            )
        except EVMRevertError as e:  # pragma: no cover
            raise LiquidityPoolError(f"Simulated execution reverted: {e}") from e
        else:
            return UniswapV3PoolSimulationResult(
                amount0_delta=amount0_delta,
                amount1_delta=amount1_delta,
                initial_state=self.state.copy(),
                final_state=UniswapV3PoolState(
                    pool=self.address,
                    liquidity=end_liquidity,
                    sqrt_price_x96=end_sqrt_price_x96,
                    tick=end_tick,
                ),
            )

    def simulate_exact_output_swap(
        self,
        token_out: Erc20Token,
        token_out_quantity: int,
        sqrt_price_limit_x96: int | None = None,
        override_state: UniswapV3PoolState | None = None,
    ) -> UniswapV3PoolSimulationResult:
        """
        Simulate an exact output swap.
        """

        if token_out not in self.tokens:  # pragma: no cover
            raise ValueError("token_out is unknown!")

        if token_out_quantity == 0:  # pragma: no cover
            raise ValueError("Zero output swap requested.")

        zero_for_one = token_out == self.token1

        try:
            (
                amount0_delta,
                amount1_delta,
                end_sqrtprice,
                end_liquidity,
                end_tick,
            ) = self._calculate_swap(
                zero_for_one=zero_for_one,
                amount_specified=-token_out_quantity,
                sqrt_price_limit_x96=(
                    sqrt_price_limit_x96
                    if sqrt_price_limit_x96 is not None
                    else (
                        TickMath.MIN_SQRT_RATIO + 1 if zero_for_one else TickMath.MAX_SQRT_RATIO - 1
                    )
                ),
                override_start_liquidity=override_state.liquidity if override_state else None,
                override_start_sqrt_price_x96=override_state.sqrt_price_x96
                if override_state
                else None,
                override_start_tick=override_state.tick if override_state else None,
                override_tick_bitmap=override_state.tick_bitmap if override_state else None,
                override_tick_data=override_state.tick_data if override_state else None,
            )
        except EVMRevertError as e:  # pragma: no cover
            raise LiquidityPoolError(f"Simulated execution reverted: {e}") from e
        else:
            return UniswapV3PoolSimulationResult(
                amount0_delta=amount0_delta,
                amount1_delta=amount1_delta,
                initial_state=self.state.copy(),
                final_state=UniswapV3PoolState(
                    pool=self.address,
                    liquidity=end_liquidity,
                    sqrt_price_x96=end_sqrtprice,
                    tick=end_tick,
                ),
            )
