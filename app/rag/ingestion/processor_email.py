"""
Email processor for EML, MSG files.

This module provides comprehensive email processing:
- EML (RFC 822) support
- MSG (Outlook) support
- Header extraction
- Body extraction (text and HTML)
- Attachment handling
- Thread detection
- Metadata extraction
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import email
from email import policy
from email.parser import BytesParser
import base64

import structlog

from .models_result import ProcessResult, ProcessorError, ErrorCode, ProcessingStage
from .dependencies import get_dependency_checker

logger = structlog.get_logger(__name__)


class EmailProcessor:
    """
    Comprehensive email processor.
    
    Features:
    - EML and MSG support
    - Header extraction
    - Body extraction (text/HTML)
    - Attachment handling
    - Thread detection
    - Metadata extraction
    """
    
    def __init__(
        self,
        extract_attachments: bool = True,
        max_attachments: int = 10,
        prefer_html: bool = False
    ):
        """
        Initialize email processor.
        
        Args:
            extract_attachments: Extract attachment metadata
            max_attachments: Maximum attachments to process
            prefer_html: Prefer HTML body over plain text
        """
        self.extract_attachments = extract_attachments
        self.max_attachments = max_attachments
        self.prefer_html = prefer_html
        
        self.dep_checker = get_dependency_checker()
        
        logger.info(
            "EmailProcessor initialized",
            extract_attachments=extract_attachments,
            max_attachments=max_attachments
        )
    
    async def process(
        self,
        content: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessResult:
        """
        Process email file.
        
        Args:
            content: Email file content
            filename: Filename
            metadata: Additional metadata
            
        Returns:
            ProcessResult with extracted content
        """
        start_time = datetime.utcnow()
        errors = []
        
        try:
            # Detect format
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.eml':
                result_data = await self._process_eml(content, filename)
            elif file_ext == '.msg':
                result_data = await self._process_msg(content, filename)
            else:
                return ProcessResult(
                    text="",
                    metadata=metadata or {},
                    errors=[ProcessorError(
                        code=ErrorCode.UNSUPPORTED_FORMAT,
                        message=f"Unsupported email format: {file_ext}",
                        stage=ProcessingStage.EXTRACTION,
                        retriable=False
                    )],
                    processor_name="EmailProcessor",
                    processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
                )
            
            # Build text representation
            text_parts = []
            
            # Headers
            text_parts.append(f"From: {result_data.get('from', 'Unknown')}")
            text_parts.append(f"To: {result_data.get('to', 'Unknown')}")
            
            if result_data.get('cc'):
                text_parts.append(f"CC: {result_data['cc']}")
            
            text_parts.append(f"Subject: {result_data.get('subject', 'No Subject')}")
            text_parts.append(f"Date: {result_data.get('date', 'Unknown')}")
            text_parts.append("")
            
            # Body
            text_parts.append(result_data.get('body', ''))
            
            # Attachments
            if result_data.get('attachments'):
                text_parts.append("")
                text_parts.append("Attachments:")
                for att in result_data['attachments']:
                    text_parts.append(f"  - {att['filename']} ({att['size']} bytes)")
            
            result_metadata = {
                **(metadata or {}),
                "email_from": result_data.get('from'),
                "email_to": result_data.get('to'),
                "email_cc": result_data.get('cc'),
                "email_subject": result_data.get('subject'),
                "email_date": result_data.get('date'),
                "email_message_id": result_data.get('message_id'),
                "email_in_reply_to": result_data.get('in_reply_to'),
                "email_references": result_data.get('references'),
                "attachments": result_data.get('attachments', [])
            }
            
            return ProcessResult(
                text="\n".join(text_parts),
                metadata=result_metadata,
                errors=errors,
                processor_name="EmailProcessor",
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error("Email processing failed", error=str(e), filename=filename)
            
            return ProcessResult(
                text="",
                metadata=metadata or {},
                errors=[ProcessorError(
                    code=ErrorCode.PROCESSING_FAILED,
                    message=f"Email processing failed: {str(e)}",
                    stage=ProcessingStage.EXTRACTION,
                    retriable=True,
                    details={"error": str(e)}
                )],
                processor_name="EmailProcessor",
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _process_eml(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Process EML file."""
        # Parse email
        msg = BytesParser(policy=policy.default).parsebytes(content)
        
        # Extract headers
        from_addr = msg.get('From', '')
        to_addr = msg.get('To', '')
        cc_addr = msg.get('CC', '')
        subject = msg.get('Subject', '')
        date = msg.get('Date', '')
        message_id = msg.get('Message-ID', '')
        in_reply_to = msg.get('In-Reply-To', '')
        references = msg.get('References', '')
        
        # Extract body
        body = self._extract_body(msg)
        
        # Extract attachments
        attachments = []
        if self.extract_attachments:
            attachments = self._extract_attachments(msg)
        
        return {
            "from": from_addr,
            "to": to_addr,
            "cc": cc_addr,
            "subject": subject,
            "date": date,
            "message_id": message_id,
            "in_reply_to": in_reply_to,
            "references": references,
            "body": body,
            "attachments": attachments
        }
    
    async def _process_msg(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Process MSG file using extract-msg."""
        try:
            import extract_msg
            
            msg = extract_msg.Message(content)
            
            # Extract headers
            from_addr = msg.sender or ''
            to_addr = msg.to or ''
            cc_addr = msg.cc or ''
            subject = msg.subject or ''
            date = str(msg.date) if msg.date else ''
            
            # Extract body
            body = msg.body or ''
            if self.prefer_html and msg.htmlBody:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(msg.htmlBody, 'html.parser')
                body = soup.get_text()
            
            # Extract attachments
            attachments = []
            if self.extract_attachments:
                for att in msg.attachments[:self.max_attachments]:
                    attachments.append({
                        "filename": att.longFilename or att.shortFilename or "unknown",
                        "size": len(att.data) if att.data else 0,
                        "content_type": att.mimeType or "application/octet-stream"
                    })
            
            return {
                "from": from_addr,
                "to": to_addr,
                "cc": cc_addr,
                "subject": subject,
                "date": date,
                "message_id": "",
                "in_reply_to": "",
                "references": "",
                "body": body,
                "attachments": attachments
            }
            
        except ImportError:
            logger.warning("extract-msg not available, MSG processing limited")
            raise NotImplementedError("MSG processing requires extract-msg library")
    
    def _extract_body(self, msg) -> str:
        """Extract email body."""
        body = ""
        
        if msg.is_multipart():
            # Get text parts
            text_parts = []
            html_parts = []
            
            for part in msg.walk():
                content_type = part.get_content_type()
                
                if content_type == 'text/plain':
                    try:
                        text_parts.append(part.get_content())
                    except Exception as e:
                        logger.warning("Failed to extract text part", error=str(e))
                
                elif content_type == 'text/html':
                    try:
                        html_parts.append(part.get_content())
                    except Exception as e:
                        logger.warning("Failed to extract HTML part", error=str(e))
            
            # Prefer HTML if configured
            if self.prefer_html and html_parts:
                from bs4 import BeautifulSoup
                html = "\n".join(html_parts)
                soup = BeautifulSoup(html, 'html.parser')
                body = soup.get_text()
            elif text_parts:
                body = "\n".join(text_parts)
            elif html_parts:
                from bs4 import BeautifulSoup
                html = "\n".join(html_parts)
                soup = BeautifulSoup(html, 'html.parser')
                body = soup.get_text()
        else:
            # Single part message
            content_type = msg.get_content_type()
            
            if content_type == 'text/plain':
                body = msg.get_content()
            elif content_type == 'text/html':
                from bs4 import BeautifulSoup
                html = msg.get_content()
                soup = BeautifulSoup(html, 'html.parser')
                body = soup.get_text()
        
        return body.strip()
    
    def _extract_attachments(self, msg) -> List[Dict[str, Any]]:
        """Extract attachment metadata."""
        attachments = []
        
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            
            if part.get('Content-Disposition') is None:
                continue
            
            filename = part.get_filename()
            if not filename:
                continue
            
            if len(attachments) >= self.max_attachments:
                break
            
            # Get attachment size
            payload = part.get_payload(decode=True)
            size = len(payload) if payload else 0
            
            attachments.append({
                "filename": filename,
                "size": size,
                "content_type": part.get_content_type()
            })
        
        return attachments

