"""
Base Operator Module

This module defines the BaseFeludaOperator abstract base class that all Feluda operators
must inherit from. It provides a standardized interface and contract enforcement for operators.
"""

import abc
import inspect
import logging
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union, cast

import deal
from pydantic import BaseModel, Field, ValidationError, create_model

from feluda.exceptions import (
    OperatorContractError,
    OperatorExecutionError,
    OperatorInitializationError,
    OperatorValidationError,
)

log = logging.getLogger(__name__)

T = TypeVar("T")
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
ParametersType = TypeVar("ParametersType", bound=BaseModel)


class BaseFeludaOperator(abc.ABC, Generic[InputType, OutputType, ParametersType]):
    """
    Abstract base class for all Feluda operators.

    All operators must inherit from this class and implement the required methods.
    This class provides contract enforcement and standardized interfaces.

    Type Parameters:
        InputType: The type of input data the operator accepts.
        OutputType: The type of output data the operator produces.
        ParametersType: The type of parameters the operator accepts (must be a Pydantic model).
    """

    # Class variables
    name: str
    description: str
    version: str
    parameters_model: Type[ParametersType]

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the operator with the given parameters.

        Args:
            parameters: Optional dictionary of parameters to configure the operator.
                        If not provided, default values from the parameters_model will be used.

        Raises:
            OperatorInitializationError: If initialization fails.
            OperatorValidationError: If parameter validation fails.
        """
        try:
            # Validate parameters using the Pydantic model
            if parameters is None:
                parameters = {}

            try:
                self.parameters = self.parameters_model(**parameters)
            except ValidationError as e:
                validation_errors = e.errors()
                error_msg = f"Parameter validation failed for operator {self.__class__.__name__}"
                log.error(f"{error_msg}: {validation_errors}")
                raise OperatorValidationError(
                    message=error_msg,
                    operator_type=self.__class__.__name__,
                    details={"validation_errors": validation_errors}
                ) from e

            # Initialize the operator
            self._initialize()

        except OperatorValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Wrap any other exceptions in OperatorInitializationError
            error_msg = f"Failed to initialize operator {self.__class__.__name__}"
            log.exception(error_msg)
            raise OperatorInitializationError(
                message=error_msg,
                operator_type=self.__class__.__name__,
                details={"cause": str(e)}
            ) from e

    @abc.abstractmethod
    def _initialize(self) -> None:
        """
        Initialize the operator after parameter validation.

        This method should be implemented by subclasses to perform any initialization
        that depends on the validated parameters.

        Raises:
            OperatorInitializationError: If initialization fails.
        """
        pass

    @deal.pre(lambda self, input_data: self._validate_input(input_data))
    @deal.post(lambda result: result is not None)
    @deal.raises(OperatorExecutionError)
    def run(self, input_data: InputType) -> OutputType:
        """
        Run the operator on the given input data.

        This method enforces contracts using the deal library:
        - Pre-condition: Input data must be valid.
        - Post-condition: Output must not be None.
        - Raises: Only OperatorExecutionError is allowed to be raised.

        Args:
            input_data: The input data to process.

        Returns:
            The processed output data.

        Raises:
            OperatorExecutionError: If execution fails.
            OperatorContractError: If a contract is violated.
        """
        try:
            # Execute the operator
            result = self._execute(input_data)

            # Validate the output
            if not self._validate_output(result):
                raise OperatorContractError(
                    message=f"Output validation failed for operator {self.__class__.__name__}",
                    operator_type=self.__class__.__name__,
                    contract_type="post"
                )

            return result

        except deal.PreContractError as e:
            # Convert deal pre-condition errors to OperatorContractError
            error_msg = f"Pre-condition contract violated for operator {self.__class__.__name__}"
            log.error(f"{error_msg}: {str(e)}")
            raise OperatorContractError(
                message=error_msg,
                operator_type=self.__class__.__name__,
                contract_type="pre",
                details={"cause": str(e)}
            ) from e

        except deal.PostContractError as e:
            # Convert deal post-condition errors to OperatorContractError
            error_msg = f"Post-condition contract violated for operator {self.__class__.__name__}"
            log.error(f"{error_msg}: {str(e)}")
            raise OperatorContractError(
                message=error_msg,
                operator_type=self.__class__.__name__,
                contract_type="post",
                details={"cause": str(e)}
            ) from e

        except OperatorContractError:
            # Re-raise contract errors
            raise

        except Exception as e:
            # Wrap any other exceptions in OperatorExecutionError
            error_msg = f"Execution failed for operator {self.__class__.__name__}"
            log.exception(error_msg)
            raise OperatorExecutionError(
                message=error_msg,
                operator_type=self.__class__.__name__,
                input_data=input_data,
                cause=e
            ) from e

    @abc.abstractmethod
    def _execute(self, input_data: InputType) -> OutputType:
        """
        Execute the operator on the given input data.

        This method should be implemented by subclasses to perform the actual
        processing of the input data.

        Args:
            input_data: The input data to process.

        Returns:
            The processed output data.

        Raises:
            Exception: Any exception that occurs during execution.
        """
        pass

    def _validate_input(self, input_data: InputType) -> bool:
        """
        Validate the input data.

        This method can be overridden by subclasses to provide custom input validation.
        The default implementation returns True, accepting any input.

        Args:
            input_data: The input data to validate.

        Returns:
            True if the input is valid, False otherwise.
        """
        return True

    def _validate_output(self, output_data: OutputType) -> bool:
        """
        Validate the output data.

        This method can be overridden by subclasses to provide custom output validation.
        The default implementation returns True, accepting any output.

        Args:
            output_data: The output data to validate.

        Returns:
            True if the output is valid, False otherwise.
        """
        return output_data is not None

    @classmethod
    def create_parameters_model(
        cls,
        model_name: str,
        **field_definitions: Any
    ) -> Type[BaseModel]:
        """
        Create a Pydantic model for operator parameters.

        This is a helper method for creating parameter models with proper typing.

        Args:
            model_name: The name of the model.
            **field_definitions: Field definitions as keyword arguments.
                                Each field should be a tuple of (type, field_info).

        Returns:
            A new Pydantic model class.
        """
        return create_model(model_name, **field_definitions)

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the operator.

        Returns:
            A dictionary containing information about the operator.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "parameters": self.parameters.model_dump() if hasattr(self, "parameters") else {},
            "input_type": str(inspect.signature(self._execute).parameters["input_data"].annotation),
            "output_type": str(inspect.signature(self._execute).return_annotation),
        }
