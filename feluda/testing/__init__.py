"""
Testing Package

This package provides advanced testing tools for Feluda.
"""

from feluda.testing.chaos import (
    ChaosConfig,
    ChaosTester,
    FailureMode,
    chaos_context,
    chaos_monkey,
    get_chaos_config,
    set_chaos_config,
)
from feluda.testing.fuzzing import (
    FuzzingConfig,
    FuzzingStrategy,
    Fuzzer,
    GrammarFuzzer,
    JSON_GRAMMAR,
    MutationFuzzer,
    RandomFuzzer,
    SQL_GRAMMAR,
    XML_GRAMMAR,
    create_fuzzer,
    fuzz_function,
)
from feluda.testing.metamorphic import (
    AdditionRelation,
    EqualityRelation,
    InverseRelation,
    MetamorphicRelation,
    MetamorphicTest,
    MultiplicationRelation,
    run_metamorphic_test,
    run_metamorphic_tests,
)

__all__ = [
    # Chaos testing
    "ChaosConfig",
    "ChaosTester",
    "FailureMode",
    "chaos_context",
    "chaos_monkey",
    "get_chaos_config",
    "set_chaos_config",
    
    # Fuzzing
    "FuzzingConfig",
    "FuzzingStrategy",
    "Fuzzer",
    "RandomFuzzer",
    "GrammarFuzzer",
    "MutationFuzzer",
    "create_fuzzer",
    "fuzz_function",
    "JSON_GRAMMAR",
    "XML_GRAMMAR",
    "SQL_GRAMMAR",
    
    # Metamorphic testing
    "MetamorphicRelation",
    "MetamorphicTest",
    "EqualityRelation",
    "AdditionRelation",
    "MultiplicationRelation",
    "InverseRelation",
    "run_metamorphic_test",
    "run_metamorphic_tests",
]
