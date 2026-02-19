"""
SQLite database operations for storing chat history and document metadata.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import pandas as pd

class DatabaseManager:
    """
    Manages SQLite database operations for the application.
    Stores chat history, document metadata, and conversation context.
    """
    
    def __init__(self, db_path: str = "chat_history.db"):
        """
        Initialize database connection and create tables.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Create necessary tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    file_hash TEXT UNIQUE,
                    file_size INTEGER,
                    upload_time TIMESTAMP,
                    chunk_count INTEGER,
                    metadata TEXT
                )
            """)
            
            # Chat history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    question TEXT,
                    answer TEXT,
                    timestamp TIMESTAMP,
                    document_ids TEXT,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON chat_history(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_history(timestamp)")
    
    def add_document_record(self, file_name: str, file_hash: str, file_size: int, 
                           chunk_count: int, metadata: Dict = None) -> int:
        """
        Record document metadata in database.
        
        Returns:
            document_id: ID of inserted record
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (file_name, file_hash, file_size, upload_time, chunk_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                file_name, file_hash, file_size, 
                datetime.now(), chunk_count,
                json.dumps(metadata) if metadata else None
            ))
            return cursor.lastrowid
    
    def add_chat_record(self, session_id: str, question: str, answer: str, 
                       document_ids: List[int], metadata: Dict = None):
        """Save chat interaction to database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history 
                (session_id, question, answer, timestamp, document_ids, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id, question, answer, datetime.now(),
                json.dumps(document_ids), json.dumps(metadata) if metadata else None
            ))
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Retrieve chat history for a session."""
        with self.get_connection() as conn:
            query = """
                SELECT question, answer, timestamp 
                FROM chat_history 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(session_id, limit))
            return df.to_dict('records')
    
    def get_document_stats(self) -> Dict:
        """Get statistics about uploaded documents."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_docs,
                    SUM(file_size) as total_size,
                    AVG(chunk_count) as avg_chunks
                FROM documents
            """)
            row = cursor.fetchone()
            return dict(row) if row else {}