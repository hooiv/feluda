"""
Unit tests for the testing module.
"""

import unittest
from unittest import mock
import time

import numpy as np

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


class TestChaosModule(unittest.TestCase):
    """Test cases for the chaos testing module."""
    
    def test_chaos_config(self):
        """Test the ChaosConfig class."""
        # Create a config
        config = ChaosConfig(
            enabled=True,
            failure_probability=0.1,
            enabled_failure_modes=[FailureMode.EXCEPTION, FailureMode.DELAY],
            exception_types=[ValueError, RuntimeError],
            max_delay_ms=500,
            seed=42,
        )
        
        # Check the attributes
        self.assertTrue(config.enabled)
        self.assertEqual(config.failure_probability, 0.1)
        self.assertEqual(config.enabled_failure_modes, [FailureMode.EXCEPTION, FailureMode.DELAY])
        self.assertEqual(config.exception_types, [ValueError, RuntimeError])
        self.assertEqual(config.max_delay_ms, 500)
        
        # Test should_fail
        # With seed=42, the first call should return False
        self.assertFalse(config.should_fail())
        
        # Test get_failure_mode
        failure_mode = config.get_failure_mode()
        self.assertIn(failure_mode, [FailureMode.EXCEPTION, FailureMode.DELAY])
        
        # Test get_exception_type
        exception_type = config.get_exception_type()
        self.assertIn(exception_type, [ValueError, RuntimeError])
        
        # Test get_delay_ms
        delay_ms = config.get_delay_ms()
        self.assertGreaterEqual(delay_ms, 0)
        self.assertLessEqual(delay_ms, 500)
    
    def test_get_set_chaos_config(self):
        """Test the get_chaos_config and set_chaos_config functions."""
        # Get the default config
        default_config = get_chaos_config()
        self.assertIsInstance(default_config, ChaosConfig)
        self.assertFalse(default_config.enabled)
        
        # Create a new config
        new_config = ChaosConfig(
            enabled=True,
            failure_probability=0.2,
        )
        
        # Set the new config
        set_chaos_config(new_config)
        
        # Get the config again
        config = get_chaos_config()
        self.assertIsInstance(config, ChaosConfig)
        self.assertTrue(config.enabled)
        self.assertEqual(config.failure_probability, 0.2)
        
        # Reset to the default config
        set_chaos_config(default_config)
    
    def test_chaos_monkey_decorator(self):
        """Test the chaos_monkey decorator."""
        # Create a function with the decorator
        @chaos_monkey(failure_probability=1.0, enabled_failure_modes=[FailureMode.EXCEPTION])
        def test_function():
            return "success"
        
        # Enable chaos testing
        original_config = get_chaos_config()
        config = ChaosConfig(
            enabled=True,
            failure_probability=1.0,
            enabled_failure_modes=[FailureMode.EXCEPTION],
            exception_types=[ValueError],
            seed=42,
        )
        set_chaos_config(config)
        
        # Call the function
        with self.assertRaises(ValueError):
            test_function()
        
        # Reset the config
        set_chaos_config(original_config)
    
    def test_chaos_context(self):
        """Test the chaos_context context manager."""
        # Create a function
        def test_function():
            return "success"
        
        # Use the context manager
        with chaos_context(
            enabled=True,
            failure_probability=1.0,
            enabled_failure_modes=[FailureMode.EXCEPTION],
            exception_types=[ValueError],
            seed=42,
        ):
            # Call the function with the chaos_monkey decorator
            @chaos_monkey()
            def decorated_function():
                return test_function()
            
            with self.assertRaises(ValueError):
                decorated_function()
        
        # Outside the context, the function should work normally
        @chaos_monkey()
        def decorated_function():
            return test_function()
        
        self.assertEqual(decorated_function(), "success")
    
    def test_chaos_tester(self):
        """Test the ChaosTester class."""
        # Create a tester
        tester = ChaosTester(
            failure_probability=1.0,
            enabled_failure_modes=[FailureMode.EXCEPTION],
            exception_types=[ValueError],
            seed=42,
        )
        
        # Create a function
        def test_function():
            return "success"
        
        # Test the function
        result = tester.test_function(test_function)
        self.assertIsNone(result)  # Should fail with an exception
        
        # Test multiple times
        results = tester.test_function_multiple(test_function, iterations=10)
        self.assertEqual(results["iterations"], 10)
        self.assertEqual(results["successes"], 0)
        self.assertEqual(results["failures"], 10)


class TestFuzzingModule(unittest.TestCase):
    """Test cases for the fuzzing module."""
    
    def test_fuzzing_config(self):
        """Test the FuzzingConfig class."""
        # Create a config
        config = FuzzingConfig(
            strategy=FuzzingStrategy.RANDOM,
            min_length=10,
            max_length=100,
            seed=42,
        )
        
        # Check the attributes
        self.assertEqual(config.strategy, FuzzingStrategy.RANDOM)
        self.assertEqual(config.min_length, 10)
        self.assertEqual(config.max_length, 100)
        self.assertIsNotNone(config.random)
        
        # Create a grammar-based config
        config = FuzzingConfig(
            strategy=FuzzingStrategy.GRAMMAR_BASED,
            grammar=JSON_GRAMMAR,
            min_length=10,
            max_length=100,
            seed=42,
        )
        
        # Check the attributes
        self.assertEqual(config.strategy, FuzzingStrategy.GRAMMAR_BASED)
        self.assertEqual(config.grammar, JSON_GRAMMAR)
        
        # Create a mutation-based config
        config = FuzzingConfig(
            strategy=FuzzingStrategy.MUTATION_BASED,
            corpus=["test1", "test2"],
            min_length=10,
            max_length=100,
            seed=42,
        )
        
        # Check the attributes
        self.assertEqual(config.strategy, FuzzingStrategy.MUTATION_BASED)
        self.assertEqual(config.corpus, ["test1", "test2"])
    
    def test_random_fuzzer(self):
        """Test the RandomFuzzer class."""
        # Create a config
        config = FuzzingConfig(
            strategy=FuzzingStrategy.RANDOM,
            min_length=10,
            max_length=20,
            seed=42,
        )
        
        # Create a fuzzer
        fuzzer = RandomFuzzer(config)
        
        # Generate a fuzzed input
        input_str = fuzzer.generate()
        
        # Check the input
        self.assertIsInstance(input_str, str)
        self.assertGreaterEqual(len(input_str), 10)
        self.assertLessEqual(len(input_str), 20)
        
        # Generate a batch
        batch = fuzzer.generate_batch(5)
        
        # Check the batch
        self.assertEqual(len(batch), 5)
        for input_str in batch:
            self.assertIsInstance(input_str, str)
            self.assertGreaterEqual(len(input_str), 10)
            self.assertLessEqual(len(input_str), 20)
    
    def test_grammar_fuzzer(self):
        """Test the GrammarFuzzer class."""
        # Create a simple grammar
        grammar = {
            "start": ["sentence"],
            "sentence": ["subject verb object"],
            "subject": ["I", "You", "He", "She", "They"],
            "verb": ["eat", "sleep", "run", "jump", "swim"],
            "object": ["food", "bed", "race", "hurdle", "pool"],
        }
        
        # Create a config
        config = FuzzingConfig(
            strategy=FuzzingStrategy.GRAMMAR_BASED,
            grammar=grammar,
            min_length=1,
            max_length=100,
            seed=42,
        )
        
        # Create a fuzzer
        fuzzer = GrammarFuzzer(config)
        
        # Generate a fuzzed input
        input_str = fuzzer.generate()
        
        # Check the input
        self.assertIsInstance(input_str, str)
        
        # Generate a batch
        batch = fuzzer.generate_batch(5)
        
        # Check the batch
        self.assertEqual(len(batch), 5)
        for input_str in batch:
            self.assertIsInstance(input_str, str)
    
    def test_mutation_fuzzer(self):
        """Test the MutationFuzzer class."""
        # Create a config
        config = FuzzingConfig(
            strategy=FuzzingStrategy.MUTATION_BASED,
            corpus=["Hello, world!", "Testing, 1, 2, 3"],
            min_length=1,
            max_length=100,
            seed=42,
        )
        
        # Create a fuzzer
        fuzzer = MutationFuzzer(config)
        
        # Generate a fuzzed input
        input_str = fuzzer.generate()
        
        # Check the input
        self.assertIsInstance(input_str, str)
        
        # Generate a batch
        batch = fuzzer.generate_batch(5)
        
        # Check the batch
        self.assertEqual(len(batch), 5)
        for input_str in batch:
            self.assertIsInstance(input_str, str)
    
    def test_create_fuzzer(self):
        """Test the create_fuzzer function."""
        # Test creating a random fuzzer
        config = FuzzingConfig(
            strategy=FuzzingStrategy.RANDOM,
            min_length=10,
            max_length=20,
            seed=42,
        )
        
        fuzzer = create_fuzzer(config)
        self.assertIsInstance(fuzzer, RandomFuzzer)
        
        # Test creating a grammar-based fuzzer
        config = FuzzingConfig(
            strategy=FuzzingStrategy.GRAMMAR_BASED,
            grammar=JSON_GRAMMAR,
            min_length=10,
            max_length=20,
            seed=42,
        )
        
        fuzzer = create_fuzzer(config)
        self.assertIsInstance(fuzzer, GrammarFuzzer)
        
        # Test creating a mutation-based fuzzer
        config = FuzzingConfig(
            strategy=FuzzingStrategy.MUTATION_BASED,
            corpus=["Hello, world!", "Testing, 1, 2, 3"],
            min_length=10,
            max_length=20,
            seed=42,
        )
        
        fuzzer = create_fuzzer(config)
        self.assertIsInstance(fuzzer, MutationFuzzer)
        
        # Test unsupported strategy
        config = FuzzingConfig(
            strategy="unsupported",
            min_length=10,
            max_length=20,
            seed=42,
        )
        
        with self.assertRaises(ValueError):
            create_fuzzer(config)
    
    def test_fuzz_function(self):
        """Test the fuzz_function function."""
        # Create a function to fuzz
        def parse_json(json_str):
            try:
                import json
                return json.loads(json_str)
            except Exception:
                raise ValueError("Invalid JSON")
        
        # Create a config
        config = FuzzingConfig(
            strategy=FuzzingStrategy.RANDOM,
            min_length=1,
            max_length=10,
            seed=42,
        )
        
        # Fuzz the function
        with mock.patch("feluda.testing.fuzzing.create_fuzzer") as mock_create:
            mock_fuzzer = mock.MagicMock()
            mock_fuzzer.generate.side_effect = ["invalid", "invalid", "{}", "invalid", "[]"]
            mock_create.return_value = mock_fuzzer
            
            results = fuzz_function(parse_json, config, iterations=5)
            
            mock_create.assert_called_once_with(config)
            self.assertEqual(mock_fuzzer.generate.call_count, 5)
            
            self.assertEqual(results["iterations"], 5)
            self.assertEqual(results["successes"], 2)  # "{}" and "[]" are valid JSON
            self.assertEqual(results["failures"], 3)  # "invalid" is not valid JSON
            self.assertIn("ValueError", results["exceptions"])
            self.assertEqual(results["exceptions"]["ValueError"], 3)


class TestMetamorphicModule(unittest.TestCase):
    """Test cases for the metamorphic testing module."""
    
    def test_equality_relation(self):
        """Test the EqualityRelation class."""
        # Create a relation
        relation = EqualityRelation(
            transformation=lambda x: x[::-1][::-1],  # Reverse twice = identity
            tolerance=1e-6,
        )
        
        # Check the attributes
        self.assertEqual(relation.relation, MetamorphicRelation.EQUALITY)
        self.assertEqual(relation.tolerance, 1e-6)
        
        # Test transform_input
        input_data = "Hello, world!"
        transformed = relation.transform_input(input_data)
        self.assertEqual(transformed, input_data)
        
        # Test verify_relation
        self.assertTrue(relation.verify_relation(input_data, transformed))
        self.assertFalse(relation.verify_relation(input_data, "Different"))
        
        # Test with numeric data
        relation = EqualityRelation(
            transformation=lambda x: x + 1e-7,  # Small difference
            tolerance=1e-6,
        )
        
        self.assertTrue(relation.verify_relation(1.0, 1.0 + 1e-7))
        self.assertFalse(relation.verify_relation(1.0, 1.0 + 1e-5))
        
        # Test with numpy arrays
        relation = EqualityRelation(
            transformation=lambda x: x + 1e-7,  # Small difference
            tolerance=1e-6,
        )
        
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7])
        
        self.assertTrue(relation.verify_relation(arr1, arr2))
    
    def test_addition_relation(self):
        """Test the AdditionRelation class."""
        # Create a relation
        relation = AdditionRelation(
            input_delta=1,
            output_delta=2,
            tolerance=1e-6,
        )
        
        # Check the attributes
        self.assertEqual(relation.relation, MetamorphicRelation.ADDITION)
        self.assertEqual(relation.input_delta, 1)
        self.assertEqual(relation.output_delta, 2)
        self.assertEqual(relation.tolerance, 1e-6)
        
        # Test transform_input
        input_data = 5
        transformed = relation.transform_input(input_data)
        self.assertEqual(transformed, 6)
        
        # Test verify_relation
        self.assertTrue(relation.verify_relation(10, 12))
        self.assertFalse(relation.verify_relation(10, 13))
        
        # Test with numpy arrays
        relation = AdditionRelation(
            input_delta=np.array([1, 1, 1]),
            output_delta=np.array([2, 2, 2]),
            tolerance=1e-6,
        )
        
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([2.0, 3.0, 4.0])  # arr1 + input_delta
        
        transformed = relation.transform_input(arr1)
        self.assertTrue(np.array_equal(transformed, arr2))
        
        output1 = np.array([10.0, 20.0, 30.0])
        output2 = np.array([12.0, 22.0, 32.0])  # output1 + output_delta
        
        self.assertTrue(relation.verify_relation(output1, output2))
    
    def test_multiplication_relation(self):
        """Test the MultiplicationRelation class."""
        # Create a relation
        relation = MultiplicationRelation(
            input_factor=2,
            output_factor=4,
            tolerance=1e-6,
        )
        
        # Check the attributes
        self.assertEqual(relation.relation, MetamorphicRelation.MULTIPLICATION)
        self.assertEqual(relation.input_factor, 2)
        self.assertEqual(relation.output_factor, 4)
        self.assertEqual(relation.tolerance, 1e-6)
        
        # Test transform_input
        input_data = 5
        transformed = relation.transform_input(input_data)
        self.assertEqual(transformed, 10)
        
        # Test verify_relation
        self.assertTrue(relation.verify_relation(10, 40))
        self.assertFalse(relation.verify_relation(10, 41))
        
        # Test with numpy arrays
        relation = MultiplicationRelation(
            input_factor=2,
            output_factor=4,
            tolerance=1e-6,
        )
        
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([2.0, 4.0, 6.0])  # arr1 * input_factor
        
        transformed = relation.transform_input(arr1)
        self.assertTrue(np.array_equal(transformed, arr2))
        
        output1 = np.array([10.0, 20.0, 30.0])
        output2 = np.array([40.0, 80.0, 120.0])  # output1 * output_factor
        
        self.assertTrue(relation.verify_relation(output1, output2))
    
    def test_run_metamorphic_test(self):
        """Test the run_metamorphic_test function."""
        # Create a function
        def square(x):
            return x * x
        
        # Create a relation
        relation = MultiplicationRelation(
            input_factor=2,
            output_factor=4,  # square(2x) = 4 * square(x)
            tolerance=1e-6,
        )
        
        # Run the test
        result = run_metamorphic_test(square, 5, relation)
        
        # Check the result
        self.assertEqual(result["relation"], MetamorphicRelation.MULTIPLICATION)
        self.assertEqual(result["original_input"], 5)
        self.assertEqual(result["transformed_input"], 10)
        self.assertEqual(result["original_output"], 25)
        self.assertEqual(result["transformed_output"], 100)
        self.assertTrue(result["relation_holds"])
    
    def test_run_metamorphic_tests(self):
        """Test the run_metamorphic_tests function."""
        # Create a function
        def square(x):
            return x * x
        
        # Create relations
        relations = [
            MultiplicationRelation(
                input_factor=2,
                output_factor=4,  # square(2x) = 4 * square(x)
                tolerance=1e-6,
            ),
            AdditionRelation(
                input_delta=1,
                output_delta=2 * 5 + 1,  # square(x+1) = square(x) + 2x + 1
                tolerance=1e-6,
            ),
        ]
        
        # Run the tests
        results = run_metamorphic_tests(square, 5, relations)
        
        # Check the results
        self.assertEqual(results["function"], "square")
        self.assertEqual(results["input_data"], 5)
        self.assertEqual(len(results["tests"]), 2)
        self.assertEqual(results["passed"], 2)
        self.assertEqual(results["failed"], 0)
        
        # Check the first test
        test1 = results["tests"][0]
        self.assertEqual(test1["relation"], MetamorphicRelation.MULTIPLICATION)
        self.assertTrue(test1["relation_holds"])
        
        # Check the second test
        test2 = results["tests"][1]
        self.assertEqual(test2["relation"], MetamorphicRelation.ADDITION)
        self.assertTrue(test2["relation_holds"])


if __name__ == "__main__":
    unittest.main()
