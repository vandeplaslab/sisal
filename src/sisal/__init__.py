"""Init."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sisal")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Paul-Louis Delacour"
__email__ = "delacourpaullouis@gmail.com"
