"""
SCFG (Stochastic Context-Free Grammar) package.

This package provides functionality for working with stochastic context-free grammars,
including grammar generation, parsing, and prompt management.
"""

from .scfg import SCFG, SCFGParams, CFGParams
from .prompt import ChatCompletionResponse, basic_prompt
from .utils import *

__version__ = "0.1.0"
__all__ = ["SCFG", "SCFGParams", "CFGParams", "ChatCompletionResponse", "basic_prompt"]
