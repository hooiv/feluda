"""
Documentation server for Feluda.

This module provides a server for serving documentation.
"""

import http.server
import logging
import os
import socketserver
import threading
import webbrowser
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from feluda.config import get_config
from feluda.docs.generator import DocGenerator, get_doc_generator
from feluda.observability import get_logger

log = get_logger(__name__)


class DocServer:
    """
    Documentation server.
    
    This class is responsible for serving documentation.
    """
    
    def __init__(
        self,
        doc_generator: Optional[DocGenerator] = None,
        host: str = "localhost",
        port: int = 8000,
        doc_dir: Optional[str] = None,
    ):
        """
        Initialize the documentation server.
        
        Args:
            doc_generator: The documentation generator.
            host: The server host.
            port: The server port.
            doc_dir: The documentation directory.
        """
        self.doc_generator = doc_generator or get_doc_generator()
        self.host = host
        self.port = port
        self.doc_dir = doc_dir or self.doc_generator.config.output_dir
        self.server = None
        self.thread = None
        self.lock = threading.RLock()
    
    def generate_docs(self, package_name: str) -> None:
        """
        Generate documentation for a package.
        
        Args:
            package_name: The package name.
        """
        with self.lock:
            self.doc_generator.generate_docs(package_name)
    
    def start(self, open_browser: bool = True) -> None:
        """
        Start the documentation server.
        
        Args:
            open_browser: Whether to open a browser.
        """
        with self.lock:
            if self.server:
                return
            
            # Create a request handler
            handler = http.server.SimpleHTTPRequestHandler
            
            # Create the server
            self.server = socketserver.TCPServer((self.host, self.port), handler)
            
            # Set the server directory
            os.chdir(self.doc_dir)
            
            # Start the server in a separate thread
            self.thread = threading.Thread(target=self.server.serve_forever)
            self.thread.daemon = True
            self.thread.start()
            
            log.info(f"Documentation server started at http://{self.host}:{self.port}")
            
            # Open a browser
            if open_browser:
                webbrowser.open(f"http://{self.host}:{self.port}")
    
    def stop(self) -> None:
        """
        Stop the documentation server.
        """
        with self.lock:
            if not self.server:
                return
            
            # Stop the server
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            
            # Wait for the thread to finish
            if self.thread:
                self.thread.join()
                self.thread = None
            
            log.info("Documentation server stopped")


# Global documentation server instance
_doc_server = None
_doc_server_lock = threading.RLock()


def get_doc_server() -> DocServer:
    """
    Get the global documentation server instance.
    
    Returns:
        The global documentation server instance.
    """
    global _doc_server
    
    with _doc_server_lock:
        if _doc_server is None:
            # Create the documentation server
            _doc_server = DocServer(
                host=get_config().doc_server_host or "localhost",
                port=int(get_config().doc_server_port or 8000),
            )
        
        return _doc_server
