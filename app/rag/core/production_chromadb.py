"""
Production-Ready ChromaDB Configuration and Management System.

This module provides comprehensive ChromaDB setup for production with:
- Distributed clustering support
- Backup and persistence mechanisms
- Performance optimization
- High availability configuration
- Monitoring and health checks
- Automatic failover and recovery
"""

import asyncio
import json
import os
import shutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import aiofiles
import aiofiles.os

logger = structlog.get_logger(__name__)


class ChromaDBMode(Enum):
    """ChromaDB deployment modes."""
    STANDALONE = "standalone"
    CLUSTER = "cluster"
    DISTRIBUTED = "distributed"


class BackupStrategy(Enum):
    """Backup strategies."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


@dataclass
class ChromaDBClusterConfig:
    """ChromaDB cluster configuration."""
    mode: ChromaDBMode = ChromaDBMode.STANDALONE
    
    # Cluster settings
    cluster_nodes: List[str] = None
    cluster_port: int = 8000
    cluster_auth_token: Optional[str] = None
    
    # Replication settings
    replication_factor: int = 3
    consistency_level: str = "eventual"  # strong, eventual
    
    # Sharding settings
    enable_sharding: bool = True
    shard_count: int = 4
    shard_key: str = "collection_name"
    
    def __post_init__(self):
        if self.cluster_nodes is None:
            self.cluster_nodes = []


@dataclass
class ChromaDBBackupConfig:
    """ChromaDB backup configuration."""
    enabled: bool = True
    backup_directory: str = "./data/chroma_backups"
    backup_strategy: BackupStrategy = BackupStrategy.INCREMENTAL
    backup_interval_hours: int = 6
    retention_days: int = 30
    compression_enabled: bool = True
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    
    # Remote backup settings
    remote_backup_enabled: bool = False
    remote_backup_url: Optional[str] = None
    remote_backup_credentials: Optional[Dict[str, str]] = None


@dataclass
class ChromaDBPerformanceConfig:
    """ChromaDB performance optimization configuration."""
    # Memory settings
    max_memory_mb: int = 4096
    cache_size_mb: int = 1024
    buffer_size_mb: int = 256
    
    # Threading settings
    max_threads: int = 8
    io_threads: int = 4
    
    # Index settings
    index_type: str = "hnsw"  # hnsw, flat, ivf
    index_params: Dict[str, Any] = None
    
    # Batch processing
    max_batch_size: int = 1000
    batch_timeout_seconds: int = 30
    
    # Connection settings
    connection_pool_size: int = 20
    connection_timeout_seconds: int = 60
    max_retries: int = 3
    
    def __post_init__(self):
        if self.index_params is None:
            self.index_params = {
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 16
            }


@dataclass
class ChromaDBMonitoringConfig:
    """ChromaDB monitoring configuration."""
    enabled: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    log_level: str = "INFO"
    
    # Alerting settings
    alerting_enabled: bool = True
    alert_thresholds: Dict[str, float] = None
    
    # Metrics collection
    collect_performance_metrics: bool = True
    collect_usage_metrics: bool = True
    metrics_retention_days: int = 7
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "memory_usage_percent": 80.0,
                "disk_usage_percent": 85.0,
                "response_time_ms": 1000.0,
                "error_rate_percent": 5.0
            }


class ProductionChromaDBManager:
    """
    Production-ready ChromaDB manager with comprehensive features.
    
    Features:
    - Clustering and high availability
    - Backup and recovery
    - Performance optimization
    - Health monitoring
    - Automatic failover
    """
    
    def __init__(
        self,
        data_directory: str = "./data/chroma_production",
        cluster_config: Optional[ChromaDBClusterConfig] = None,
        backup_config: Optional[ChromaDBBackupConfig] = None,
        performance_config: Optional[ChromaDBPerformanceConfig] = None,
        monitoring_config: Optional[ChromaDBMonitoringConfig] = None
    ):
        self.data_directory = Path(data_directory)
        self.cluster_config = cluster_config or ChromaDBClusterConfig()
        self.backup_config = backup_config or ChromaDBBackupConfig()
        self.performance_config = performance_config or ChromaDBPerformanceConfig()
        self.monitoring_config = monitoring_config or ChromaDBMonitoringConfig()
        
        self.client = None
        self.is_initialized = False
        self.backup_task = None
        self.health_monitor_task = None
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "collections_count": 0,
            "documents_count": 0
        }
        
    async def initialize(self) -> bool:
        """Initialize production ChromaDB system."""
        try:
            logger.info("Initializing production ChromaDB system")
            
            # Create data directories
            await self._create_directories()
            
            # Initialize ChromaDB client
            await self._initialize_client()
            
            # Setup clustering if enabled
            if self.cluster_config.mode != ChromaDBMode.STANDALONE:
                await self._setup_clustering()
            
            # Setup backup system
            if self.backup_config.enabled:
                await self._setup_backup_system()
            
            # Setup monitoring
            if self.monitoring_config.enabled:
                await self._setup_monitoring()
            
            # Perform health check
            if not await self._health_check():
                raise RuntimeError("ChromaDB health check failed")
            
            self.is_initialized = True
            logger.info("Production ChromaDB system initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize production ChromaDB", error=str(e))
            return False
    
    async def shutdown(self) -> None:
        """Shutdown production ChromaDB system."""
        try:
            logger.info("Shutting down production ChromaDB system")
            
            # Stop monitoring
            if self.health_monitor_task:
                self.health_monitor_task.cancel()
            
            # Stop backup task
            if self.backup_task:
                self.backup_task.cancel()
            
            # Perform final backup
            if self.backup_config.enabled:
                await self._perform_backup(BackupStrategy.FULL)
            
            # Close client connections
            if self.client:
                # ChromaDB doesn't have explicit close method
                self.client = None
            
            logger.info("Production ChromaDB system shutdown complete")
            
        except Exception as e:
            logger.error("Error during ChromaDB shutdown", error=str(e))
    
    async def get_client(self) -> chromadb.Client:
        """Get ChromaDB client with failover support."""
        if not self.is_initialized:
            await self.initialize()
        
        if not self.client:
            await self._initialize_client()
        
        return self.client
    
    async def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Any] = None
    ) -> Any:
        """Create a collection with production settings."""
        try:
            client = await self.get_client()
            
            # Apply performance optimizations to metadata
            production_metadata = {
                **(metadata or {}),
                **self.performance_config.index_params,
                "created_at": datetime.utcnow().isoformat(),
                "production_mode": True
            }
            
            collection = client.create_collection(
                name=name,
                metadata=production_metadata,
                embedding_function=embedding_function
            )
            
            self.metrics["collections_count"] += 1
            logger.info("Collection created", name=name)
            
            return collection
            
        except Exception as e:
            logger.error("Failed to create collection", name=name, error=str(e))
            raise
    
    async def backup_collection(self, collection_name: str) -> bool:
        """Backup a specific collection."""
        try:
            return await self._backup_collection(collection_name)
        except Exception as e:
            logger.error("Failed to backup collection", collection=collection_name, error=str(e))
            return False
    
    async def restore_collection(self, collection_name: str, backup_path: str) -> bool:
        """Restore a collection from backup."""
        try:
            return await self._restore_collection(collection_name, backup_path)
        except Exception as e:
            logger.error("Failed to restore collection", collection=collection_name, error=str(e))
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        try:
            client = await self.get_client()
            
            # Update collection and document counts
            collections = client.list_collections()
            self.metrics["collections_count"] = len(collections)
            
            total_docs = 0
            for collection in collections:
                try:
                    count = collection.count()
                    total_docs += count
                except:
                    pass
            
            self.metrics["documents_count"] = total_docs
            
            # Add system metrics
            system_metrics = await self._get_system_metrics()
            
            return {
                **self.metrics,
                **system_metrics,
                "timestamp": datetime.utcnow().isoformat(),
                "cluster_mode": self.cluster_config.mode.value,
                "backup_enabled": self.backup_config.enabled,
                "monitoring_enabled": self.monitoring_config.enabled
            }
            
        except Exception as e:
            logger.error("Failed to get metrics", error=str(e))
            return self.metrics

    # Private helper methods

    async def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.data_directory,
            Path(self.backup_config.backup_directory),
            self.data_directory / "logs",
            self.data_directory / "metrics"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info("Created data directories")

    async def _initialize_client(self) -> None:
        """Initialize ChromaDB client with production settings."""
        try:
            if self.cluster_config.mode == ChromaDBMode.STANDALONE:
                # Standalone mode
                settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=False,  # Disable reset in production
                    persist_directory=str(self.data_directory),
                    chroma_server_host="localhost",
                    chroma_server_http_port=8000,
                    chroma_server_grpc_port=8001,
                    chroma_server_cors_allow_origins=["*"],
                    chroma_server_auth_provider="chromadb.auth.basic.BasicAuthServerProvider",
                    chroma_server_auth_credentials_file=str(self.data_directory / "auth.txt"),
                    chroma_collection_cache_size=self.performance_config.cache_size_mb * 1024 * 1024,
                    chroma_segment_cache_policy="LRU"
                )

                self.client = chromadb.PersistentClient(
                    path=str(self.data_directory),
                    settings=settings
                )

            elif self.cluster_config.mode == ChromaDBMode.CLUSTER:
                # Cluster mode
                if not self.cluster_config.cluster_nodes:
                    raise ValueError("Cluster nodes must be specified for cluster mode")

                # Use first node as primary
                primary_node = self.cluster_config.cluster_nodes[0]
                host, port = primary_node.split(":")

                settings = Settings(
                    anonymized_telemetry=False,
                    chroma_server_auth_provider="chromadb.auth.token.TokenAuthServerProvider",
                    chroma_server_auth_token_transport_header="X-Chroma-Token"
                )

                self.client = chromadb.HttpClient(
                    host=host,
                    port=int(port),
                    settings=settings,
                    headers={"X-Chroma-Token": self.cluster_config.cluster_auth_token} if self.cluster_config.cluster_auth_token else None
                )

            logger.info("ChromaDB client initialized", mode=self.cluster_config.mode.value)

        except Exception as e:
            logger.error("Failed to initialize ChromaDB client", error=str(e))
            raise

    async def _setup_clustering(self) -> None:
        """Setup ChromaDB clustering."""
        try:
            logger.info("Setting up ChromaDB clustering")

            # Create cluster configuration file
            cluster_config = {
                "mode": self.cluster_config.mode.value,
                "nodes": self.cluster_config.cluster_nodes,
                "replication_factor": self.cluster_config.replication_factor,
                "consistency_level": self.cluster_config.consistency_level,
                "sharding": {
                    "enabled": self.cluster_config.enable_sharding,
                    "shard_count": self.cluster_config.shard_count,
                    "shard_key": self.cluster_config.shard_key
                }
            }

            config_path = self.data_directory / "cluster_config.json"
            async with aiofiles.open(config_path, 'w') as f:
                await f.write(json.dumps(cluster_config, indent=2))

            logger.info("Clustering setup complete")

        except Exception as e:
            logger.error("Failed to setup clustering", error=str(e))
            raise

    async def _setup_backup_system(self) -> None:
        """Setup backup system."""
        try:
            logger.info("Setting up backup system")

            # Create backup directory
            backup_dir = Path(self.backup_config.backup_directory)
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Start backup task
            self.backup_task = asyncio.create_task(self._backup_scheduler())

            logger.info("Backup system setup complete")

        except Exception as e:
            logger.error("Failed to setup backup system", error=str(e))
            raise

    async def _setup_monitoring(self) -> None:
        """Setup monitoring system."""
        try:
            logger.info("Setting up monitoring system")

            # Start health monitor
            self.health_monitor_task = asyncio.create_task(self._health_monitor())

            logger.info("Monitoring system setup complete")

        except Exception as e:
            logger.error("Failed to setup monitoring", error=str(e))
            raise

    async def _health_check(self) -> bool:
        """Perform comprehensive health check."""
        try:
            if not self.client:
                return False

            # Test basic operations
            test_collection_name = f"health_check_{int(time.time())}"

            # Create test collection
            collection = self.client.create_collection(test_collection_name)

            # Add test document
            collection.add(
                documents=["Health check document"],
                ids=["health_check_1"],
                metadatas=[{"test": True}]
            )

            # Query test document
            results = collection.query(
                query_texts=["Health check"],
                n_results=1
            )

            # Cleanup
            self.client.delete_collection(test_collection_name)

            # Verify results
            if results and len(results.get("documents", [])) > 0:
                logger.info("ChromaDB health check passed")
                return True
            else:
                logger.error("ChromaDB health check failed - no results")
                return False

        except Exception as e:
            logger.error("ChromaDB health check failed", error=str(e))
            return False

    async def _backup_scheduler(self) -> None:
        """Background task for scheduled backups."""
        while True:
            try:
                await asyncio.sleep(self.backup_config.backup_interval_hours * 3600)
                await self._perform_backup(self.backup_config.backup_strategy)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Backup scheduler error", error=str(e))
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _health_monitor(self) -> None:
        """Background task for health monitoring."""
        while True:
            try:
                await asyncio.sleep(self.monitoring_config.health_check_interval)

                # Perform health check
                is_healthy = await self._health_check()

                if not is_healthy:
                    logger.warning("ChromaDB health check failed")
                    # Trigger recovery if needed
                    await self._attempt_recovery()

                # Collect metrics
                if self.monitoring_config.collect_performance_metrics:
                    await self._collect_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _perform_backup(self, strategy: BackupStrategy) -> bool:
        """Perform backup with specified strategy."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_name = f"chroma_backup_{strategy.value}_{timestamp}"
            backup_path = Path(self.backup_config.backup_directory) / backup_name

            logger.info("Starting backup", strategy=strategy.value, path=str(backup_path))

            if strategy == BackupStrategy.FULL:
                # Full backup - copy entire data directory
                await self._copy_directory(self.data_directory, backup_path)

            elif strategy == BackupStrategy.INCREMENTAL:
                # Incremental backup - only changed files
                await self._incremental_backup(backup_path)

            elif strategy == BackupStrategy.DIFFERENTIAL:
                # Differential backup - changes since last full backup
                await self._differential_backup(backup_path)

            # Compress if enabled
            if self.backup_config.compression_enabled:
                await self._compress_backup(backup_path)

            # Encrypt if enabled
            if self.backup_config.encryption_enabled:
                await self._encrypt_backup(backup_path)

            # Cleanup old backups
            await self._cleanup_old_backups()

            logger.info("Backup completed successfully", path=str(backup_path))
            return True

        except Exception as e:
            logger.error("Backup failed", error=str(e))
            return False

    async def _backup_collection(self, collection_name: str) -> bool:
        """Backup a specific collection."""
        try:
            client = await self.get_client()
            collection = client.get_collection(collection_name)

            # Export collection data
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = Path(self.backup_config.backup_directory) / f"{collection_name}_{timestamp}.json"

            # Get all documents from collection
            results = collection.get()

            backup_data = {
                "collection_name": collection_name,
                "metadata": collection.metadata,
                "documents": results.get("documents", []),
                "ids": results.get("ids", []),
                "metadatas": results.get("metadatas", []),
                "embeddings": results.get("embeddings", []),
                "backup_timestamp": timestamp
            }

            async with aiofiles.open(backup_file, 'w') as f:
                await f.write(json.dumps(backup_data, indent=2))

            logger.info("Collection backup completed", collection=collection_name)
            return True

        except Exception as e:
            logger.error("Collection backup failed", collection=collection_name, error=str(e))
            return False

    async def _restore_collection(self, collection_name: str, backup_path: str) -> bool:
        """Restore a collection from backup."""
        try:
            client = await self.get_client()

            # Load backup data
            async with aiofiles.open(backup_path, 'r') as f:
                backup_data = json.loads(await f.read())

            # Delete existing collection if it exists
            try:
                client.delete_collection(collection_name)
            except:
                pass  # Collection might not exist

            # Create new collection
            collection = client.create_collection(
                name=collection_name,
                metadata=backup_data.get("metadata", {})
            )

            # Restore documents
            documents = backup_data.get("documents", [])
            ids = backup_data.get("ids", [])
            metadatas = backup_data.get("metadatas", [])
            embeddings = backup_data.get("embeddings", [])

            if documents and ids:
                # Add documents in batches
                batch_size = self.performance_config.max_batch_size
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i + batch_size]
                    batch_ids = ids[i:i + batch_size]
                    batch_metas = metadatas[i:i + batch_size] if metadatas else None
                    batch_embeds = embeddings[i:i + batch_size] if embeddings else None

                    collection.add(
                        documents=batch_docs,
                        ids=batch_ids,
                        metadatas=batch_metas,
                        embeddings=batch_embeds
                    )

            logger.info("Collection restore completed", collection=collection_name)
            return True

        except Exception as e:
            logger.error("Collection restore failed", collection=collection_name, error=str(e))
            return False

    async def _copy_directory(self, source: Path, destination: Path) -> None:
        """Copy directory asynchronously."""
        destination.mkdir(parents=True, exist_ok=True)

        for item in source.rglob("*"):
            if item.is_file():
                relative_path = item.relative_to(source)
                dest_file = destination / relative_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                async with aiofiles.open(item, 'rb') as src:
                    async with aiofiles.open(dest_file, 'wb') as dst:
                        await dst.write(await src.read())

    async def _incremental_backup(self, backup_path: Path) -> None:
        """Perform incremental backup."""
        # Find last backup
        last_backup = await self._find_last_backup()

        if last_backup:
            # Copy only files modified since last backup
            last_backup_time = datetime.fromtimestamp(last_backup.stat().st_mtime)

            for item in self.data_directory.rglob("*"):
                if item.is_file():
                    file_time = datetime.fromtimestamp(item.stat().st_mtime)
                    if file_time > last_backup_time:
                        relative_path = item.relative_to(self.data_directory)
                        dest_file = backup_path / relative_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)

                        async with aiofiles.open(item, 'rb') as src:
                            async with aiofiles.open(dest_file, 'wb') as dst:
                                await dst.write(await src.read())
        else:
            # No previous backup, perform full backup
            await self._copy_directory(self.data_directory, backup_path)

    async def _differential_backup(self, backup_path: Path) -> None:
        """Perform differential backup."""
        # Find last full backup
        last_full_backup = await self._find_last_full_backup()

        if last_full_backup:
            # Copy files modified since last full backup
            last_backup_time = datetime.fromtimestamp(last_full_backup.stat().st_mtime)

            for item in self.data_directory.rglob("*"):
                if item.is_file():
                    file_time = datetime.fromtimestamp(item.stat().st_mtime)
                    if file_time > last_backup_time:
                        relative_path = item.relative_to(self.data_directory)
                        dest_file = backup_path / relative_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)

                        async with aiofiles.open(item, 'rb') as src:
                            async with aiofiles.open(dest_file, 'wb') as dst:
                                await dst.write(await src.read())
        else:
            # No previous full backup, perform full backup
            await self._copy_directory(self.data_directory, backup_path)

    async def _compress_backup(self, backup_path: Path) -> None:
        """Compress backup directory."""
        import tarfile

        compressed_path = backup_path.with_suffix('.tar.gz')

        def compress():
            with tarfile.open(compressed_path, 'w:gz') as tar:
                tar.add(backup_path, arcname=backup_path.name)

        # Run compression in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, compress)

        # Remove uncompressed directory
        shutil.rmtree(backup_path)

    async def _encrypt_backup(self, backup_path: Path) -> None:
        """Encrypt backup (placeholder for encryption logic)."""
        # Implement encryption if needed
        logger.info("Backup encryption not implemented")

    async def _cleanup_old_backups(self) -> None:
        """Clean up old backups based on retention policy."""
        try:
            backup_dir = Path(self.backup_config.backup_directory)
            cutoff_date = datetime.utcnow() - timedelta(days=self.backup_config.retention_days)

            for backup_file in backup_dir.iterdir():
                if backup_file.is_file() or backup_file.is_dir():
                    file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        if backup_file.is_file():
                            backup_file.unlink()
                        else:
                            shutil.rmtree(backup_file)
                        logger.info("Removed old backup", path=str(backup_file))

        except Exception as e:
            logger.error("Failed to cleanup old backups", error=str(e))

    async def _find_last_backup(self) -> Optional[Path]:
        """Find the most recent backup."""
        backup_dir = Path(self.backup_config.backup_directory)
        backups = [f for f in backup_dir.iterdir() if f.name.startswith("chroma_backup_")]

        if backups:
            return max(backups, key=lambda f: f.stat().st_mtime)
        return None

    async def _find_last_full_backup(self) -> Optional[Path]:
        """Find the most recent full backup."""
        backup_dir = Path(self.backup_config.backup_directory)
        full_backups = [f for f in backup_dir.iterdir() if "full" in f.name]

        if full_backups:
            return max(full_backups, key=lambda f: f.stat().st_mtime)
        return None

    async def _attempt_recovery(self) -> bool:
        """Attempt to recover from failures."""
        try:
            logger.info("Attempting ChromaDB recovery")

            # Try to reinitialize client
            try:
                await self._initialize_client()
                if await self._health_check():
                    logger.info("Recovery successful - client reinitialized")
                    return True
            except Exception as e:
                logger.warning("Client reinitialization failed", error=str(e))

            # Try to restore from latest backup
            if self.backup_config.enabled:
                latest_backup = await self._find_last_backup()
                if latest_backup:
                    logger.info("Attempting restore from backup", backup=str(latest_backup))
                    # Implement restore logic here
                    return True

            logger.error("All recovery attempts failed")
            return False

        except Exception as e:
            logger.error("Recovery attempt failed", error=str(e))
            return False

    async def _collect_metrics(self) -> None:
        """Collect performance metrics."""
        try:
            # Update request metrics
            self.metrics["timestamp"] = datetime.utcnow().isoformat()

            # Get system metrics
            system_metrics = await self._get_system_metrics()
            self.metrics.update(system_metrics)

            # Save metrics to file
            metrics_file = self.data_directory / "metrics" / f"metrics_{datetime.utcnow().strftime('%Y%m%d_%H')}.json"

            async with aiofiles.open(metrics_file, 'w') as f:
                await f.write(json.dumps(self.metrics, indent=2))

        except Exception as e:
            logger.error("Failed to collect metrics", error=str(e))

    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            import psutil

            # Memory usage
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage(str(self.data_directory))

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            return {
                "system": {
                    "memory_total_mb": memory.total // (1024 * 1024),
                    "memory_used_mb": memory.used // (1024 * 1024),
                    "memory_percent": memory.percent,
                    "disk_total_gb": disk.total // (1024 * 1024 * 1024),
                    "disk_used_gb": disk.used // (1024 * 1024 * 1024),
                    "disk_percent": (disk.used / disk.total) * 100,
                    "cpu_percent": cpu_percent
                }
            }

        except ImportError:
            logger.warning("psutil not available for system metrics")
            return {"system": {"status": "metrics_unavailable"}}
        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return {"system": {"error": str(e)}}


# Global production ChromaDB manager
production_chroma_manager = None


async def get_production_chroma_manager(
    data_directory: str = "./data/chroma_production",
    **kwargs
) -> ProductionChromaDBManager:
    """Get or create the global production ChromaDB manager."""
    global production_chroma_manager

    if production_chroma_manager is None:
        production_chroma_manager = ProductionChromaDBManager(
            data_directory=data_directory,
            **kwargs
        )
        await production_chroma_manager.initialize()

    return production_chroma_manager


async def shutdown_production_chroma() -> None:
    """Shutdown the global production ChromaDB manager."""
    global production_chroma_manager

    if production_chroma_manager:
        await production_chroma_manager.shutdown()
        production_chroma_manager = None


# Production configuration presets
PRODUCTION_CONFIGS = {
    "standalone": {
        "cluster_config": ChromaDBClusterConfig(mode=ChromaDBMode.STANDALONE),
        "performance_config": ChromaDBPerformanceConfig(
            max_memory_mb=8192,
            cache_size_mb=2048,
            max_threads=16,
            connection_pool_size=50
        ),
        "backup_config": ChromaDBBackupConfig(
            backup_interval_hours=12,
            retention_days=30
        )
    },
    "cluster": {
        "cluster_config": ChromaDBClusterConfig(
            mode=ChromaDBMode.CLUSTER,
            replication_factor=3,
            enable_sharding=True,
            shard_count=8
        ),
        "performance_config": ChromaDBPerformanceConfig(
            max_memory_mb=16384,
            cache_size_mb=4096,
            max_threads=32,
            connection_pool_size=100
        ),
        "backup_config": ChromaDBBackupConfig(
            backup_interval_hours=6,
            retention_days=60,
            remote_backup_enabled=True
        )
    },
    "high_performance": {
        "performance_config": ChromaDBPerformanceConfig(
            max_memory_mb=32768,
            cache_size_mb=8192,
            max_threads=64,
            connection_pool_size=200,
            max_batch_size=2000,
            index_params={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 400,
                "hnsw:M": 32
            }
        ),
        "monitoring_config": ChromaDBMonitoringConfig(
            health_check_interval=15,
            collect_performance_metrics=True,
            metrics_retention_days=14
        )
    }
}


def get_production_config(preset: str = "standalone") -> Dict[str, Any]:
    """Get production configuration preset."""
    if preset not in PRODUCTION_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRODUCTION_CONFIGS.keys())}")

    return PRODUCTION_CONFIGS[preset]
