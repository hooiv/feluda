"""
Fuzzing Module

This module provides tools for fuzzing in Feluda.
Fuzzing involves generating random inputs to test system robustness.
"""

import json
import logging
import random
import string
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

from feluda.observability import get_logger

log = get_logger(__name__)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


class FuzzingStrategy(str, Enum):
    """Enum for fuzzing strategies."""
    
    RANDOM = "random"
    GRAMMAR_BASED = "grammar_based"
    MUTATION_BASED = "mutation_based"
    EVOLUTIONARY = "evolutionary"
    COVERAGE_GUIDED = "coverage_guided"


class FuzzingConfig:
    """
    Configuration for fuzzing.
    
    This class holds the configuration for fuzzing, including the strategy
    and parameters for generating inputs.
    """
    
    def __init__(
        self,
        strategy: FuzzingStrategy = FuzzingStrategy.RANDOM,
        min_length: int = 1,
        max_length: int = 100,
        seed: Optional[int] = None,
        grammar: Optional[Dict[str, List[str]]] = None,
        corpus: Optional[List[str]] = None,
    ):
        """
        Initialize a FuzzingConfig.
        
        Args:
            strategy: The fuzzing strategy to use.
            min_length: The minimum length of generated inputs.
            max_length: The maximum length of generated inputs.
            seed: The random seed to use. If None, a random seed is used.
            grammar: The grammar to use for grammar-based fuzzing.
                    This should be a dictionary mapping non-terminal symbols to lists of productions.
            corpus: The corpus to use for mutation-based fuzzing.
                  This should be a list of valid inputs.
        """
        self.strategy = strategy
        self.min_length = min_length
        self.max_length = max_length
        self.random = random.Random(seed)
        self.grammar = grammar or {}
        self.corpus = corpus or []


class Fuzzer:
    """
    Base class for fuzzers.
    
    This class defines the interface for fuzzers.
    Concrete implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, config: FuzzingConfig):
        """
        Initialize a Fuzzer.
        
        Args:
            config: The fuzzing configuration.
        """
        self.config = config
    
    def generate(self) -> str:
        """
        Generate a fuzzed input.
        
        Returns:
            A fuzzed input.
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate")
    
    def generate_batch(self, count: int) -> List[str]:
        """
        Generate a batch of fuzzed inputs.
        
        Args:
            count: The number of inputs to generate.
            
        Returns:
            A list of fuzzed inputs.
        """
        return [self.generate() for _ in range(count)]


class RandomFuzzer(Fuzzer):
    """
    Fuzzer that generates random inputs.
    
    This fuzzer generates random strings of ASCII characters.
    """
    
    def generate(self) -> str:
        """
        Generate a random input.
        
        Returns:
            A random input.
        """
        length = self.config.random.randint(self.config.min_length, self.config.max_length)
        return ''.join(self.config.random.choice(string.printable) for _ in range(length))


class GrammarFuzzer(Fuzzer):
    """
    Fuzzer that generates inputs based on a grammar.
    
    This fuzzer generates inputs by expanding a grammar.
    """
    
    def __init__(self, config: FuzzingConfig):
        """
        Initialize a GrammarFuzzer.
        
        Args:
            config: The fuzzing configuration.
            
        Raises:
            ValueError: If the grammar is not specified.
        """
        super().__init__(config)
        
        if not self.config.grammar:
            raise ValueError("Grammar must be specified for grammar-based fuzzing")
        
        # Ensure that the grammar has a start symbol
        if "start" not in self.config.grammar:
            raise ValueError("Grammar must have a 'start' symbol")
    
    def generate(self) -> str:
        """
        Generate an input based on the grammar.
        
        Returns:
            An input generated from the grammar.
        """
        return self._expand_symbol("start", 0)
    
    def _expand_symbol(self, symbol: str, depth: int) -> str:
        """
        Expand a symbol in the grammar.
        
        Args:
            symbol: The symbol to expand.
            depth: The current recursion depth.
            
        Returns:
            The expanded symbol.
        """
        # Check if the symbol is a terminal (not in the grammar)
        if symbol not in self.config.grammar:
            return symbol
        
        # Check if we've reached the maximum recursion depth
        if depth > 10:  # Arbitrary limit to prevent infinite recursion
            return ""
        
        # Choose a random production for the symbol
        production = self.config.random.choice(self.config.grammar[symbol])
        
        # Expand the production
        result = ""
        for token in production.split():
            result += self._expand_symbol(token, depth + 1)
        
        return result


class MutationFuzzer(Fuzzer):
    """
    Fuzzer that generates inputs by mutating existing inputs.
    
    This fuzzer generates inputs by mutating inputs from a corpus.
    """
    
    def __init__(self, config: FuzzingConfig):
        """
        Initialize a MutationFuzzer.
        
        Args:
            config: The fuzzing configuration.
            
        Raises:
            ValueError: If the corpus is not specified.
        """
        super().__init__(config)
        
        if not self.config.corpus:
            raise ValueError("Corpus must be specified for mutation-based fuzzing")
    
    def generate(self) -> str:
        """
        Generate an input by mutating an existing input.
        
        Returns:
            A mutated input.
        """
        # Choose a random input from the corpus
        input_str = self.config.random.choice(self.config.corpus)
        
        # Apply random mutations
        num_mutations = self.config.random.randint(1, 10)
        for _ in range(num_mutations):
            mutation_type = self.config.random.choice(["insert", "delete", "replace", "swap"])
            
            if mutation_type == "insert" and len(input_str) < self.config.max_length:
                # Insert a random character
                pos = self.config.random.randint(0, len(input_str))
                char = self.config.random.choice(string.printable)
                input_str = input_str[:pos] + char + input_str[pos:]
                
            elif mutation_type == "delete" and len(input_str) > self.config.min_length:
                # Delete a random character
                pos = self.config.random.randint(0, len(input_str) - 1)
                input_str = input_str[:pos] + input_str[pos + 1:]
                
            elif mutation_type == "replace" and len(input_str) > 0:
                # Replace a random character
                pos = self.config.random.randint(0, len(input_str) - 1)
                char = self.config.random.choice(string.printable)
                input_str = input_str[:pos] + char + input_str[pos + 1:]
                
            elif mutation_type == "swap" and len(input_str) > 1:
                # Swap two adjacent characters
                pos = self.config.random.randint(0, len(input_str) - 2)
                input_str = input_str[:pos] + input_str[pos + 1] + input_str[pos] + input_str[pos + 2:]
        
        return input_str


def create_fuzzer(config: FuzzingConfig) -> Fuzzer:
    """
    Create a fuzzer based on the configuration.
    
    Args:
        config: The fuzzing configuration.
        
    Returns:
        A fuzzer instance.
        
    Raises:
        ValueError: If the strategy is not supported.
    """
    if config.strategy == FuzzingStrategy.RANDOM:
        return RandomFuzzer(config)
    elif config.strategy == FuzzingStrategy.GRAMMAR_BASED:
        return GrammarFuzzer(config)
    elif config.strategy == FuzzingStrategy.MUTATION_BASED:
        return MutationFuzzer(config)
    else:
        raise ValueError(f"Unsupported fuzzing strategy: {config.strategy}")


def fuzz_function(
    func: Callable[[str], Any],
    config: FuzzingConfig,
    iterations: int = 100,
) -> Dict[str, Any]:
    """
    Fuzz a function with generated inputs.
    
    Args:
        func: The function to fuzz.
        config: The fuzzing configuration.
        iterations: The number of iterations to run.
        
    Returns:
        A dictionary with the fuzzing results.
    """
    fuzzer = create_fuzzer(config)
    
    results = {
        "iterations": iterations,
        "successes": 0,
        "failures": 0,
        "exceptions": {},
    }
    
    for _ in range(iterations):
        input_str = fuzzer.generate()
        
        try:
            func(input_str)
            results["successes"] += 1
        except Exception as e:
            results["failures"] += 1
            exception_type = type(e).__name__
            results["exceptions"][exception_type] = results["exceptions"].get(exception_type, 0) + 1
    
    return results


# Example grammars for common input formats

JSON_GRAMMAR = {
    "start": ["object", "array"],
    "object": ["{ }", "{ members }"],
    "members": ["pair", "pair , members"],
    "pair": ["string : value"],
    "array": ["[ ]", "[ elements ]"],
    "elements": ["value", "value , elements"],
    "value": ["string", "number", "object", "array", "true", "false", "null"],
    "string": ["\" characters \""],
    "characters": ["", "character characters"],
    "character": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                 "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
                 "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                 "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                 "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "_", "-", "+", ".",
                 " ", "\\\"", "\\\\", "\\/", "\\b", "\\f", "\\n", "\\r", "\\t"],
    "number": ["int", "int frac", "int exp", "int frac exp"],
    "int": ["digit", "digit1-9 digits", "- digit", "- digit1-9 digits"],
    "digit": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "digit1-9": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "digits": ["digit", "digit digits"],
    "frac": [". digits"],
    "exp": ["e digits", "E digits", "e+ digits", "E+ digits", "e- digits", "E- digits"],
}

XML_GRAMMAR = {
    "start": ["element"],
    "element": ["< tag attributes > content </ tag >", "< tag attributes />"],
    "tag": ["name"],
    "attributes": ["", "attribute attributes"],
    "attribute": ["name = \" value \""],
    "content": ["", "element content", "text content"],
    "name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
             "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
             "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
             "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
    "value": ["a", "b", "c", "1", "2", "3"],
    "text": ["a", "b", "c", "1", "2", "3", " "],
}

SQL_GRAMMAR = {
    "start": ["query"],
    "query": ["select_query", "insert_query", "update_query", "delete_query"],
    "select_query": ["SELECT columns FROM table where_clause"],
    "insert_query": ["INSERT INTO table ( columns ) VALUES ( values )"],
    "update_query": ["UPDATE table SET assignments where_clause"],
    "delete_query": ["DELETE FROM table where_clause"],
    "columns": ["column", "column , columns"],
    "column": ["name", "* "],
    "table": ["name"],
    "where_clause": ["", "WHERE condition"],
    "condition": ["column = value", "column > value", "column < value"],
    "assignments": ["column = value", "column = value , assignments"],
    "values": ["value", "value , values"],
    "value": ["string", "number", "NULL"],
    "string": ["' characters '"],
    "characters": ["", "character characters"],
    "character": ["a", "b", "c", "1", "2", "3", " "],
    "number": ["digit", "digit number"],
    "digit": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "name": ["a", "b", "c", "table1", "table2", "column1", "column2"],
}
