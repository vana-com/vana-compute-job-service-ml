"""
Query Engine Client for Vana Query API

This module provides a client for executing queries against the Vana Query Engine.
It handles query submission, status polling, and results downloading in a robust manner.
"""

import os
import time
import logging
import requests
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Container for query execution results"""
    success: bool
    data: Dict[str, Any]
    file_path: Optional[Path] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


class QueryError(Exception):
    """Exception raised for query execution errors"""
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class QueryEngineClient:
    """Client for executing queries against the Vana Query Engine"""

    def __init__(
        self, 
        timeout_seconds: int = 150,
        poll_interval: int = 5,
        query_engine_url: Optional[str] = None
    ):
        """
        Initialize the QueryEngineClient
        
        Args:
            timeout_seconds: Maximum time to wait for results in seconds (default: 150)
            poll_interval: Seconds between status checks (default: 5)
            query_engine_url: Override the default Query Engine URL
        """
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.query_engine_url = query_engine_url or os.getenv("QUERY_ENGINE_URL", "https://query.vana.org")
        
        logger.info("QueryEngineClient initialized")

    def execute_query(
        self, 
        job_id: int, 
        refiner_id: int, 
        query: str,
        query_signature: str,
        results_dir: Path,
        params: Optional[List[Any]] = None,
        results_name: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> QueryResult:
        """
        Execute a query and wait for results
        
        Args:
            job_id: The compute job ID
            refiner_id: The data refiner ID
            query: The SQL query to execute
            query_signature: Signed token for authenticating the request
            results_dir: Directory where query results will be saved
            params: Optional parameters to include with the query
            results_name: Name of the results file (defaults to <query_id>.db)
            max_retries: Maximum number of submission retry attempts
            retry_delay: Seconds to wait between retries
            
        Returns:
            QueryResult object containing success status and result data
            
        Example:
            >>> client = QueryEngineClient()
            >>> result = client.execute_query(
            >>>     job_id=21, 
            >>>     refiner_id=12, 
            >>>     query="SELECT * FROM data",
            >>>     query_signature="abc123",
            >>>     results_dir=Path("/path/to/results"),
            >>>     params=[1, 2, 3],
            >>>     results_name="query_results.db"
            >>> )
            >>> if result.success:
            >>>     print(f"Query successful, results at: {result.file_path}")
        """
        if not isinstance(results_dir, Path):
            results_dir = Path(results_dir)
            
        # Ensure results directory exists
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate parameters
        validation_result = self._validate_query_params(query, query_signature)
        if validation_result:
            return validation_result
        
        # Execute with retry logic
        return self._execute_with_retry(
            job_id, refiner_id, query, query_signature, 
            results_dir, params, results_name, 
            max_retries, retry_delay
        )
        
    def _validate_query_params(self, query: str, query_signature: str) -> Optional[QueryResult]:
        """
        Validate query parameters
        
        Args:
            query: The SQL query to execute
            query_signature: Signed token for authenticating the request
            
        Returns:
            QueryResult with error if validation fails, None if validation passes
        """
        if not query:
            return QueryResult(
                success=False,
                data={},
                error="Query cannot be empty",
                status_code=400
            )
            
        if not query_signature:
            return QueryResult(
                success=False,
                data={},
                error="Query signature is required",
                status_code=400
            )
            
        return None
        
    def _execute_with_retry(
        self,
        job_id: int, 
        refiner_id: int, 
        query: str,
        query_signature: str,
        results_dir: Path,
        params: Optional[List[Any]],
        results_name: Optional[str],
        max_retries: int,
        retry_delay: int
    ) -> QueryResult:
        """
        Execute query with retry logic
        
        Args:
            job_id: The compute job ID
            refiner_id: The data refiner ID
            query: The SQL query to execute
            query_signature: Signed token for authenticating the request
            results_dir: Directory where query results will be saved
            params: Optional parameters to include with the query
            results_name: Name of the results file
            max_retries: Maximum number of submission retry attempts
            retry_delay: Seconds to wait between retries
            
        Returns:
            QueryResult object containing success status and result data
        """
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Submit the query
                query_id = self._submit_query(job_id, refiner_id, query, query_signature, params)
                
                # Poll for results
                return self._poll_until_complete(query_id, query_signature, results_dir, results_name)
                
            except QueryError as e:
                # Only retry connection/timeout errors
                if e.status_code in (500, 502, 503, 504, 429):
                    retry_count += 1
                    last_error = e
                    logger.warning(
                        f"Retrying query after error ({retry_count}/{max_retries}): {e.message}"
                    )
                    time.sleep(retry_delay)
                    continue
                
                # Don't retry other errors
                logger.error(f"Query error: {e.message}")
                return QueryResult(
                    success=False,
                    data={},
                    error=e.message,
                    status_code=e.status_code
                )
                
            except Exception as e:
                # Log unexpected errors
                error_msg = f"Unexpected error executing query: {str(e)}"
                logger.exception(error_msg)
                return QueryResult(
                    success=False,
                    data={},
                    error=error_msg,
                    status_code=500
                )
        
        # If we get here, we've exhausted retries
        if last_error:
            return QueryResult(
                success=False,
                data={},
                error=f"Query failed after {max_retries} retries: {last_error.message}",
                status_code=last_error.status_code
            )
        
        return QueryResult(
            success=False,
            data={},
            error=f"Query failed after {max_retries} retries",
            status_code=500
        )

    def _submit_query(
        self, 
        job_id: int, 
        refiner_id: int, 
        query: str,
        query_signature: str,
        params: Optional[List[Any]] = None
    ) -> str:
        """
        Submit a query to the Query Engine
        
        Args:
            job_id: The compute job ID
            refiner_id: The data refiner ID
            query: SQL query to execute
            query_signature: Authentication signature
            params: Query parameters
            
        Returns:
            Query ID for tracking
            
        Raises:
            QueryError: If the query submission fails
        """
        url = f"{self.query_engine_url}/query"
        headers = self._get_headers(query_signature)
        data = {
            "query": query,
            "params": params or [],
            "refiner_id": refiner_id,
            "job_id": str(job_id)
        }
        
        logger.info(f"Submitting query: {query}")
        
        try:
            with requests.post(url, headers=headers, json=data, timeout=30) as response:
                response.raise_for_status()
                
                response_data = response.json()
                query_id = response_data.get("query_id", "")
                
                if not query_id:
                    raise QueryError("No query ID returned from server", 500)
                    
                logger.info(f"Query submitted successfully with ID: {query_id}")
                return query_id
            
        except requests.HTTPError as e:
            error_detail = self._extract_error_details(e.response)
            raise QueryError(
                f"HTTP error submitting query: {error_detail}",
                e.response.status_code
            )
        except requests.ConnectionError as e:
            raise QueryError(f"Connection error: {str(e)}", 502)
        except requests.Timeout as e:
            raise QueryError("Request timed out after 30 seconds", 504)
        except ValueError as e:
            # JSON parsing error
            raise QueryError(f"Invalid response format: {str(e)}", 500)
        except Exception as e:
            raise QueryError(f"Error submitting query: {str(e)}", 500)

    def _poll_until_complete(
        self, 
        query_id: str,
        query_signature: str,
        results_dir: Path,
        results_name: Optional[str] = None
    ) -> QueryResult:
        """
        Poll for query completion and download results when ready
        
        Args:
            query_id: The ID of the query to poll
            query_signature: Authentication signature
            results_dir: Directory where results will be saved
            results_name: Filename for results (defaults to <query_id>.db)
            
        Returns:
            QueryResult object with outcome
            
        Raises:
            QueryError: If polling fails or times out
        """
        url = f"{self.query_engine_url}/query/{query_id}"
        headers = self._get_headers(query_signature)
        
        start_time = time.time()
        logger.info(f"Polling for results of query {query_id}")
        
        while (time.time() - start_time) < self.timeout_seconds:
            try:
                response_data = self._fetch_query_status(url, headers, query_id)
                query_status = response_data.get("query_status", "")
                
                # Handle the query status
                result = self._handle_query_status(
                    query_id, query_status, response_data, 
                    query_signature, results_dir, results_name
                )
                
                if result:
                    return result
                
                # Wait before polling again
                time.sleep(self.poll_interval)
                
            except QueryError:
                # Pass through QueryError exceptions
                raise
            except requests.HTTPError as e:
                error_detail = self._extract_error_details(e.response)
                raise QueryError(
                    f"HTTP error polling query: {error_detail}",
                    e.response.status_code
                )
            except requests.ConnectionError as e:
                raise QueryError(f"Connection error polling query: {str(e)}", 502)
            except requests.Timeout as e:
                raise QueryError(f"Request timed out polling for results after {self.timeout_seconds} seconds", 504)
            except ValueError as e:
                # JSON parsing error
                raise QueryError(f"Invalid response format: {str(e)}", 500)
            except Exception as e:
                raise QueryError(f"Error polling query: {str(e)}", 500)
        
        # If we get here, we've exceeded the timeout
        raise QueryError(
            f"Timeout exceeded ({self.timeout_seconds}s) waiting for query results",
            408
        )
        
    def _fetch_query_status(
        self, 
        url: str, 
        headers: Dict[str, str], 
        query_id: str
    ) -> Dict[str, Any]:
        """
        Fetch query status from the server
        
        Args:
            url: URL to fetch status from
            headers: Request headers
            query_id: The query ID to check
            
        Returns:
            Response data from the server
            
        Raises:
            QueryError: If the status fetch fails
        """
        with requests.get(url, headers=headers, timeout=30) as response:
            if response.status_code == 404:
                raise QueryError(f"Query {query_id} not found", 404)
            
            response.raise_for_status()
            response_data = response.json()
            
            # Check for query ID mismatches
            if "query_id" in response_data and response_data["query_id"] != query_id:
                raise QueryError(f"Query ID mismatch: {response_data['query_id']} != {query_id}")
            
            if "query_id" not in response_data:
                response_data["query_id"] = query_id
                
            return response_data
    
    def _handle_query_status(
        self,
        query_id: str,
        query_status: str,
        response_data: Dict[str, Any],
        query_signature: str,
        results_dir: Path,
        results_name: Optional[str]
    ) -> Optional[QueryResult]:
        """
        Handle different query status values
        
        Args:
            query_id: The ID of the query
            query_status: Status of the query
            response_data: Response data from the server
            query_signature: Authentication signature
            results_dir: Directory where results will be saved
            results_name: Filename for results
            
        Returns:
            QueryResult if the query has completed (success or failure),
            None if still processing
        """
        if query_status == "success":
            logger.info(f"Query {query_id} completed successfully")
            
            # Download results if URL is provided
            results_url = response_data.get("query_results")
            if results_url:
                # Generate results path
                if not results_name:
                    results_name = f"{query_id}.db"
                    
                # Download the results
                downloaded_path = self._download_results(
                    url=results_url, 
                    query_signature=query_signature,
                    results_path=results_dir / results_name
                )
                response_data["downloaded_path"] = str(downloaded_path)
            
                return QueryResult(
                    success=True,
                    data=response_data,
                    file_path=downloaded_path
                )
            
            # Query succeeded but no results URL
            logger.warning(f"Query {query_id} succeeded but no results URL provided")
            return QueryResult(
                success=True,
                data=response_data,
                file_path=None
            )
        
        if query_status == "failed":
            error_msg = response_data.get("error", f"Query {query_id} failed")
            logger.error(error_msg)
            return QueryResult(
                success=False,
                data=response_data,
                error=error_msg
            )
        
        logger.info(f"Query {query_id} is still '{query_status}', waiting {self.poll_interval} seconds before polling again")
        return None

    def _download_results(
        self, 
        url: str, 
        query_signature: str,
        results_path: Path
    ) -> Path:
        """
        Download query results to the specified path
        
        Args:
            url: URL to download results from
            query_signature: Authentication signature
            results_path: Path where results will be saved
            
        Returns:
            Path where the results were saved
            
        Raises:
            QueryError: If download fails
        """
        logger.info(f"Downloading query results from {url} to {results_path}")
        
        try:
            headers = self._get_headers(query_signature)
            
            # Create parent directory if it doesn't exist
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            self._stream_download(url, headers, results_path)
            
            logger.info(f"Successfully downloaded query results to {results_path}")
            return results_path
                
        except requests.HTTPError as e:
            error_detail = self._extract_error_details(e.response)
            raise QueryError(
                f"HTTP error downloading results: {error_detail}",
                e.response.status_code
            )
        except requests.ConnectionError as e:
            raise QueryError(f"Connection error downloading results: {str(e)}", 502)
        except requests.Timeout as e:
            raise QueryError("Download timed out after 60 seconds", 504)
        except PermissionError as e:
            raise QueryError(f"Permission denied writing to {results_path}: {str(e)}", 500)
        except OSError as e:
            raise QueryError(f"OS error writing to {results_path}: {str(e)}", 500)
        except Exception as e:
            raise QueryError(f"Error downloading results: {str(e)}", 500)
            
    def _stream_download(self, url: str, headers: Dict[str, str], file_path: Path) -> None:
        """
        Stream download from URL to file
        
        Args:
            url: URL to download from
            headers: Request headers
            file_path: Path to save file to
            
        Raises:
            Various exceptions for download failures
        """
        with requests.get(url, headers=headers, timeout=60, stream=True) as response:
            response.raise_for_status()
            
            # Save the file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    def _get_headers(self, query_signature: str) -> Dict[str, str]:
        """
        Return standard headers for API requests
        
        Args:
            query_signature: Authentication signature
            
        Returns:
            Dictionary of HTTP headers
        """
        return {
            "Content-Type": "application/json",
            "X-Query-Signature": query_signature
        }
        
    def _extract_error_details(self, response) -> str:
        """
        Extract detailed error information from a response
        
        Args:
            response: Response object from requests
            
        Returns:
            String containing error details
        """
        status_code = response.status_code
        error_detail = f"Status code: {status_code}"
        
        try:
            error_json = response.json()
            if "detail" in error_json:
                error_detail += f", Detail: {error_json['detail']}"
            elif "message" in error_json:
                error_detail += f", Message: {error_json['message']}"
            elif "error" in error_json:
                error_detail += f", Error: {error_json['error']}"
        except (ValueError, KeyError):
            # Can't parse JSON or expected keys not found
            error_detail += f", Response: {response.text[:100]}"
            
        return error_detail
