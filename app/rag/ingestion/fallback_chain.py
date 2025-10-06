"""
Fallback chain pattern for robust document processing.

This module provides a fallback mechanism that tries multiple processors
in sequence until one succeeds, aggregating errors from all attempts.

Example:
    PDF → Try text layer extraction → Try OCR → Try raw strings
    Image → Try Tesseract → Try EasyOCR → Try PaddleOCR
"""

from typing import List, Callable, Any, Optional, Awaitable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime

import structlog

from .models_result import ProcessResult, ProcessorError, ErrorCode, ProcessingStage

logger = structlog.get_logger(__name__)

T = TypeVar('T')


@dataclass
class FallbackAttempt:
    """Record of a fallback attempt."""
    processor_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[ProcessorError] = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FallbackResult:
    """Result from fallback chain execution."""
    success: bool
    result: Optional[Any] = None
    attempts: List[FallbackAttempt] = field(default_factory=list)
    total_duration_ms: float = 0.0
    
    def get_error_summary(self) -> str:
        """Get summary of all errors encountered."""
        if self.success:
            return "Success"
        
        errors = []
        for attempt in self.attempts:
            if attempt.error:
                errors.append(f"{attempt.processor_name}: {attempt.error.message}")
        
        return "; ".join(errors) if errors else "All processors failed"
    
    def get_successful_processor(self) -> Optional[str]:
        """Get name of processor that succeeded."""
        for attempt in self.attempts:
            if attempt.success:
                return attempt.processor_name
        return None


class FallbackChain:
    """
    Fallback chain for trying multiple processors in sequence.
    
    Tries processors in order until one succeeds. Aggregates errors
    from all attempts for debugging and monitoring.
    """
    
    def __init__(self, name: str = "fallback_chain"):
        """
        Initialize fallback chain.
        
        Args:
            name: Chain name for logging
        """
        self.name = name
        logger.info("FallbackChain initialized", name=name)
    
    async def execute(self,
                     processors: List[tuple[str, Callable[..., Awaitable[T]]]],
                     *args,
                     **kwargs) -> FallbackResult:
        """
        Execute fallback chain.
        
        Tries each processor in sequence until one succeeds.
        
        Args:
            processors: List of (name, processor_function) tuples
            *args: Arguments to pass to processors
            **kwargs: Keyword arguments to pass to processors
            
        Returns:
            FallbackResult with success status and result
        """
        if not processors:
            logger.error("No processors provided to fallback chain", chain=self.name)
            return FallbackResult(
                success=False,
                result=None,
                attempts=[],
                total_duration_ms=0.0
            )
        
        attempts = []
        start_time = datetime.utcnow()
        
        logger.info(
            "Starting fallback chain",
            chain=self.name,
            processor_count=len(processors)
        )
        
        for processor_name, processor_func in processors:
            attempt_start = datetime.utcnow()
            
            try:
                logger.debug(
                    "Trying processor",
                    chain=self.name,
                    processor=processor_name
                )
                
                # Execute processor
                result = await processor_func(*args, **kwargs)
                
                # Calculate duration
                duration_ms = (datetime.utcnow() - attempt_start).total_seconds() * 1000
                
                # Check if result indicates success
                is_success = self._is_successful_result(result)
                
                if is_success:
                    # Success!
                    attempt = FallbackAttempt(
                        processor_name=processor_name,
                        success=True,
                        result=result,
                        error=None,
                        duration_ms=duration_ms
                    )
                    attempts.append(attempt)
                    
                    total_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    logger.info(
                        "Fallback chain succeeded",
                        chain=self.name,
                        processor=processor_name,
                        attempts=len(attempts),
                        duration_ms=total_duration_ms
                    )
                    
                    return FallbackResult(
                        success=True,
                        result=result,
                        attempts=attempts,
                        total_duration_ms=total_duration_ms
                    )
                else:
                    # Result indicates failure
                    error = self._extract_error_from_result(result, processor_name)
                    
                    attempt = FallbackAttempt(
                        processor_name=processor_name,
                        success=False,
                        result=result,
                        error=error,
                        duration_ms=duration_ms
                    )
                    attempts.append(attempt)
                    
                    logger.warning(
                        "Processor failed, trying next",
                        chain=self.name,
                        processor=processor_name,
                        error=error.message if error else "Unknown error"
                    )
                
            except Exception as e:
                # Exception during processing
                duration_ms = (datetime.utcnow() - attempt_start).total_seconds() * 1000
                
                error = ProcessorError(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=str(e),
                    stage=ProcessingStage.EXTRACTION,
                    retriable=True,
                    details={"exception_type": type(e).__name__}
                )
                
                attempt = FallbackAttempt(
                    processor_name=processor_name,
                    success=False,
                    result=None,
                    error=error,
                    duration_ms=duration_ms
                )
                attempts.append(attempt)
                
                logger.warning(
                    "Processor raised exception, trying next",
                    chain=self.name,
                    processor=processor_name,
                    error=str(e),
                    exception_type=type(e).__name__
                )
        
        # All processors failed
        total_duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        logger.error(
            "All processors in fallback chain failed",
            chain=self.name,
            attempts=len(attempts),
            duration_ms=total_duration_ms
        )
        
        return FallbackResult(
            success=False,
            result=None,
            attempts=attempts,
            total_duration_ms=total_duration_ms
        )
    
    def _is_successful_result(self, result: Any) -> bool:
        """
        Check if result indicates success.
        
        Args:
            result: Result from processor
            
        Returns:
            True if successful
        """
        # Check ProcessResult
        if isinstance(result, ProcessResult):
            # Success if text is not empty and no fatal errors
            return bool(result.text and result.text.strip()) and not result.has_fatal_errors()
        
        # Check dict result (legacy format)
        if isinstance(result, dict):
            text = result.get('text', '')
            return bool(text and text.strip())
        
        # Check string result
        if isinstance(result, str):
            return bool(result.strip())
        
        # Other types: consider success if not None
        return result is not None
    
    def _extract_error_from_result(self, result: Any, processor_name: str) -> ProcessorError:
        """
        Extract error information from result.
        
        Args:
            result: Result from processor
            processor_name: Name of processor
            
        Returns:
            ProcessorError
        """
        # Check ProcessResult
        if isinstance(result, ProcessResult):
            if result.errors:
                # Return first error
                return result.errors[0]
            else:
                # No explicit error, but result was unsuccessful
                return ProcessorError(
                    code=ErrorCode.EMPTY_CONTENT,
                    message=f"{processor_name} produced empty content",
                    stage=ProcessingStage.EXTRACTION,
                    retriable=True
                )
        
        # Check dict result
        if isinstance(result, dict):
            error_msg = result.get('error', 'Unknown error')
            return ProcessorError(
                code=ErrorCode.EXTRACTION_FAILED,
                message=f"{processor_name}: {error_msg}",
                stage=ProcessingStage.EXTRACTION,
                retriable=True
            )
        
        # Default error
        return ProcessorError(
            code=ErrorCode.EXTRACTION_FAILED,
            message=f"{processor_name} failed",
            stage=ProcessingStage.EXTRACTION,
            retriable=True
        )


async def process_with_fallbacks(
    processors: List[tuple[str, Callable[..., Awaitable[T]]]],
    *args,
    chain_name: str = "fallback",
    **kwargs
) -> FallbackResult:
    """
    Convenience function to process with fallback chain.
    
    Args:
        processors: List of (name, processor_function) tuples
        *args: Arguments to pass to processors
        chain_name: Name for the fallback chain
        **kwargs: Keyword arguments to pass to processors
        
    Returns:
        FallbackResult
    """
    chain = FallbackChain(name=chain_name)
    return await chain.execute(processors, *args, **kwargs)

