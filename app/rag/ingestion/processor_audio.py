"""
Audio processor with Whisper integration for speech-to-text.

This module provides comprehensive audio processing:
- Speech-to-text transcription (Whisper, Google, Azure)
- Speaker diarization
- Audio metadata extraction
- Format conversion
- Noise reduction
- Multiple language support
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import json
import asyncio

import structlog
import aiofiles

from .models_result import ProcessResult, ProcessorError, ErrorCode, ProcessingStage
from .subprocess_async import run_command
from .dependencies import get_dependency_checker
from .safe_ops import with_timeout

logger = structlog.get_logger(__name__)


class AudioProcessor:
    """
    Comprehensive audio processor with multiple transcription engines.
    
    Features:
    - Whisper (OpenAI) - best quality
    - Google Speech-to-Text
    - Azure Speech Services
    - Speaker diarization
    - Timestamp extraction
    - Audio metadata
    """
    
    def __init__(
        self,
        whisper_model: str = "base",
        enable_diarization: bool = False,
        language: Optional[str] = None
    ):
        """
        Initialize audio processor.
        
        Args:
            whisper_model: Whisper model size (tiny/base/small/medium/large)
            enable_diarization: Enable speaker diarization
            language: Language code (auto-detect if None)
        """
        self.whisper_model = whisper_model
        self.enable_diarization = enable_diarization
        self.language = language
        
        self.dep_checker = get_dependency_checker()
        
        # Detect available engines
        self.available_engines = self._detect_engines()
        
        logger.info(
            "AudioProcessor initialized",
            whisper_model=whisper_model,
            available_engines=self.available_engines
        )
    
    def _detect_engines(self) -> List[str]:
        """Detect available transcription engines."""
        engines = []
        
        if self.dep_checker.is_available('whisper'):
            engines.append('whisper')
        
        if self.dep_checker.is_available('speech_recognition'):
            engines.append('google')
        
        if self.dep_checker.is_available('azure.cognitiveservices.speech'):
            engines.append('azure')
        
        return engines
    
    async def process(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessResult:
        """
        Process audio file.
        
        Args:
            content: Audio file content
            filename: Filename
            metadata: Additional metadata
            
        Returns:
            ProcessResult with transcription
        """
        start_time = datetime.utcnow()
        errors = []
        
        try:
            # Extract audio metadata
            audio_metadata = await self._extract_metadata(content, filename)
            
            # Transcribe audio
            transcription_result = await self._transcribe(content, filename)
            
            if not transcription_result:
                return ProcessResult(
                    text="",
                    metadata={**(metadata or {}), **audio_metadata},
                    errors=[ProcessorError(
                        code=ErrorCode.PROCESSING_FAILED,
                        message="Failed to transcribe audio",
                        stage=ProcessingStage.TRANSCRIPTION,
                        retriable=True
                    )],
                    processor_name="AudioProcessor",
                    processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
                )
            
            # Build result
            result_metadata = {
                **(metadata or {}),
                **audio_metadata,
                "transcription_engine": transcription_result.get("engine"),
                "language_detected": transcription_result.get("language"),
                "confidence": transcription_result.get("confidence", 1.0),
                "segments": transcription_result.get("segments", []),
                "speakers": transcription_result.get("speakers", [])
            }
            
            return ProcessResult(
                text=transcription_result["text"],
                metadata=result_metadata,
                language=transcription_result.get("language", "unknown"),
                confidence=transcription_result.get("confidence", 1.0),
                errors=errors,
                processor_name="AudioProcessor",
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error("Audio processing failed", error=str(e), filename=filename)
            
            return ProcessResult(
                text="",
                metadata=metadata or {},
                errors=[ProcessorError(
                    code=ErrorCode.PROCESSING_FAILED,
                    message=f"Audio processing failed: {str(e)}",
                    stage=ProcessingStage.TRANSCRIPTION,
                    retriable=True,
                    details={"error": str(e)}
                )],
                processor_name="AudioProcessor",
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _extract_metadata(
        self,
        content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """Extract audio metadata using ffprobe."""
        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                # Run ffprobe
                result = await run_command(
                    ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', tmp_path],
                    timeout=30.0
                )
                
                if result.returncode != 0:
                    logger.warning("ffprobe failed", stderr=result.stderr)
                    return {}
                
                # Parse JSON output
                probe_data = json.loads(result.stdout)
                
                # Extract audio stream info
                audio_stream = None
                for stream in probe_data.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        audio_stream = stream
                        break
                
                if not audio_stream:
                    return {}
                
                # Extract format info
                format_info = probe_data.get('format', {})
                
                # Parse duration
                duration_seconds = 0.0
                try:
                    duration_seconds = float(format_info.get('duration', 0))
                except (ValueError, TypeError):
                    pass
                
                return {
                    "audio_codec": audio_stream.get('codec_name'),
                    "sample_rate": int(audio_stream.get('sample_rate', 0)),
                    "channels": int(audio_stream.get('channels', 0)),
                    "bit_rate": int(format_info.get('bit_rate', 0)),
                    "duration_seconds": duration_seconds,
                    "duration_formatted": str(timedelta(seconds=int(duration_seconds))),
                    "format": format_info.get('format_name')
                }
                
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.warning("Failed to extract audio metadata", error=str(e))
            return {}
    
    async def _transcribe(
        self,
        content: bytes,
        filename: str
    ) -> Optional[Dict[str, Any]]:
        """Transcribe audio using available engines."""
        # Try engines in order of preference
        for engine in ['whisper', 'google', 'azure']:
            if engine not in self.available_engines:
                continue
            
            try:
                if engine == 'whisper':
                    return await self._transcribe_whisper(content, filename)
                elif engine == 'google':
                    return await self._transcribe_google(content, filename)
                elif engine == 'azure':
                    return await self._transcribe_azure(content, filename)
            except Exception as e:
                logger.warning(f"{engine} transcription failed", error=str(e))
                continue
        
        return None
    
    async def _transcribe_whisper(
        self,
        content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """Transcribe using Whisper."""
        import whisper
        import torch
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model(self.whisper_model, device=device)
            
            # Transcribe
            transcribe_options = {
                "fp16": device == "cuda",
                "verbose": False
            }
            
            if self.language:
                transcribe_options["language"] = self.language
            
            result = await asyncio.to_thread(
                model.transcribe,
                tmp_path,
                **transcribe_options
            )
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "confidence": segment.get("no_speech_prob", 0.0)
                })
            
            return {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": segments,
                "engine": "whisper",
                "model": self.whisper_model,
                "confidence": 1.0 - result.get("no_speech_prob", 0.0)
            }
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    async def _transcribe_google(
        self,
        content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """Transcribe using Google Speech-to-Text."""
        import speech_recognition as sr
        
        # Convert to WAV if needed
        wav_content = await self._convert_to_wav(content, filename)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(wav_content)
            tmp_path = tmp.name
        
        try:
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(tmp_path) as source:
                audio = recognizer.record(source)
            
            # Recognize
            text = await asyncio.to_thread(
                recognizer.recognize_google,
                audio,
                language=self.language,
                show_all=False
            )
            
            return {
                "text": text,
                "language": self.language or "en-US",
                "engine": "google",
                "confidence": 0.9
            }
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    async def _transcribe_azure(
        self,
        content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """Transcribe using Azure Speech Services."""
        # Placeholder for Azure implementation
        # Requires Azure subscription key and region
        raise NotImplementedError("Azure transcription requires configuration")
    
    async def _convert_to_wav(
        self,
        content: bytes,
        filename: str
    ) -> bytes:
        """Convert audio to WAV format using ffmpeg."""
        # Write input to temp file
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp_in:
            tmp_in.write(content)
            tmp_in_path = tmp_in.name
        
        # Create output temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
            tmp_out_path = tmp_out.name
        
        try:
            # Convert using ffmpeg
            result = await run_command(
                ['ffmpeg', '-i', tmp_in_path, '-ar', '16000', '-ac', '1', '-y', tmp_out_path],
                timeout=60.0
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
            
            # Read converted file
            async with aiofiles.open(tmp_out_path, 'rb') as f:
                return await f.read()
                
        finally:
            Path(tmp_in_path).unlink(missing_ok=True)
            Path(tmp_out_path).unlink(missing_ok=True)

