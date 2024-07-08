from decimal import Decimal
from typing import Tuple

import polars

from ...constants import MAX_UINT8
from ...exceptions import BitmapWordUnavailableError, EVMRevertError, MissingTickWordError
from ...logging import logger
from . import bit_math as BitMath


def flipTick(
    tick_bitmap: polars.DataFrame,
    tick: int,
    tick_spacing: int,
    update_block: int | None = None,  # TODO: add deprecation warning
) -> polars.DataFrame:
    if not (tick % tick_spacing == 0):
        raise EVMRevertError("Tick not correctly spaced!")

    word_pos, bit_pos = position(int(Decimal(tick) // tick_spacing))
    if word_pos not in tick_bitmap["word"]:
        raise MissingTickWordError(f"Called flipTick on missing word={word_pos}")

    logger.debug(f"Flipping {tick=} @ {word_pos=}, {bit_pos=}")

    mask = 1 << bit_pos
    bitmap_at_word = int(tick_bitmap.filter(polars.col("word") == word_pos).select("bitmap").item())

    updated_word = polars.DataFrame(
        data={
            "word": word_pos,
            "bitmap": str(bitmap_at_word ^ mask),
        },
        schema=tick_bitmap.schema,
    )
    logger.debug(f"Flipped {tick=} @ {word_pos=}, {bit_pos=}")

    return tick_bitmap.update(
        other=updated_word,
        left_on=["word"],
        right_on=["word"],
        how="full",
    )


def position(tick: int) -> Tuple[int, int]:
    word_pos: int = tick >> 8
    bit_pos: int = tick % 256
    return (word_pos, bit_pos)


def nextInitializedTickWithinOneWord(
    tick_bitmap: polars.DataFrame,
    tick: int,
    tick_spacing: int,
    less_than_or_equal: bool,
) -> Tuple[int, bool]:
    compressed = int(
        Decimal(tick) // tick_spacing
    )  # tick can be negative, use Decimal so floor division rounds to zero instead of negative infinity
    if tick < 0 and tick % tick_spacing != 0:
        compressed -= 1  # round towards negative infinity

    if less_than_or_equal:
        word_pos, bit_pos = position(compressed)
        if word_pos not in tick_bitmap["word"]:
            raise BitmapWordUnavailableError(f"Bitmap at word {word_pos} unavailable.", word_pos)

        bitmap_at_word = int(
            tick_bitmap.filter(polars.col("word") == word_pos).select("bitmap").item()
        )

        # all the 1s at or to the right of the current bitPos
        mask = 2 * (1 << bit_pos) - 1
        masked = bitmap_at_word & mask

        # if there are no initialized ticks to the right of or at the current tick, return rightmost in the word
        initialized_status = masked != 0
        # overflow/underflow is possible, but prevented externally by limiting both tickSpacing and tick
        next_tick = (
            (compressed - (bit_pos - BitMath.mostSignificantBit(masked))) * tick_spacing
            if initialized_status
            else (compressed - (bit_pos)) * tick_spacing
        )
    else:
        # start from the word of the next tick, since the current tick state doesn't matter
        word_pos, bit_pos = position(compressed + 1)
        if word_pos not in tick_bitmap["word"]:
            raise BitmapWordUnavailableError(f"Bitmap at word {word_pos} unavailable.", word_pos)

        bitmap_at_word = int(
            tick_bitmap.filter(polars.col("word") == word_pos).select("bitmap").item()
        )

        # all the 1s at or to the left of the bitPos
        mask = ~((1 << bit_pos) - 1)
        masked = bitmap_at_word & mask

        # if there are no initialized ticks to the left of the current tick, return leftmost in the word
        initialized_status = masked != 0
        # overflow/underflow is possible, but prevented externally by limiting both tickSpacing and tick
        next_tick = (
            (compressed + 1 + (BitMath.leastSignificantBit(masked) - bit_pos)) * tick_spacing
            if initialized_status
            else (compressed + 1 + (MAX_UINT8 - bit_pos)) * tick_spacing
        )

    return next_tick, initialized_status
