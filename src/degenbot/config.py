import web3
import web3.providers
import pathlib

from .exceptions import DegenbotError
from .logging import logger

_web3: web3.Web3
_provider_address: str
_ipc_path: str
_cache_dir = pathlib.Path.home() / ".degenbot"


def set_cache_dir(path) -> None:
    global _cache_dir
    _cache_dir = path


def get_cache_dir() -> pathlib.Path:
    return _cache_dir


def get_cache_path() -> pathlib.Path:
    return _cache_dir / "cache.sqlite"


def get_web3() -> web3.Web3:
    global _web3

    try:
        return _web3
    except NameError:
        try:
            _provider_address
        except NameError:
            _web3 = web3.Web3(web3.IPCProvider(_ipc_path))
            return _web3
        else:
            if "http://" in _provider_address or "https://" in _provider_address:
                _web3 = web3.Web3(web3.HTTPProvider(_provider_address))
                return _web3
            elif "ws://" in _provider_address or "wss://" in _provider_address:
                _web3 = web3.Web3(web3.WebsocketProvider(_provider_address))
                return _web3
            raise DegenbotError("A Web3 instance has not been provided.") from None


def set_web3(w3: web3.Web3) -> None:
    if w3.is_connected() is False:
        raise DegenbotError("Web3 object is not connected.")

    logger.info(f"Connected to Web3 provider {w3.provider}")

    global _web3
    global _provider_address
    global _ipc_path

    _web3 = w3
    if isinstance(w3.provider, (web3.HTTPProvider, web3.WebsocketProvider)):
        _provider_address = w3.provider.endpoint_uri
    if isinstance(w3.provider, web3.IPCProvider):
        _ipc_path = w3.provider.ipc_path


if not pathlib.Path.exists(get_cache_path()):
    from .cache.database import create_database_and_tables

    pathlib.Path.mkdir(_cache_dir, parents=True)
    print(f"Created cache in {_cache_dir}")
    create_database_and_tables()
    print(f"Created DB at {get_cache_path()}")
