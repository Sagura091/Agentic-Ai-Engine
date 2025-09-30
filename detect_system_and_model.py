#!/usr/bin/env python3
"""
System Detection and Ollama Model Recommendation Script

This script:
1. Detects system specifications (CPU, RAM, GPU)
2. Checks if Ollama is installed
3. Recommends the best multimodal model for the system
4. Optionally pulls the recommended model

Usage:
    python detect_system_and_model.py
"""

import asyncio
import platform
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


class SystemSpecs:
    """Detect system specifications."""
    
    def __init__(self):
        self.os = platform.system()
        self.cpu_count = os.cpu_count() or 1
        self.ram_gb = self._get_ram_gb()
        self.gpu_info = self._get_gpu_info()
        self.has_nvidia = "nvidia" in self.gpu_info.lower()
        self.has_amd = "amd" in self.gpu_info.lower()
        self.has_apple_silicon = self._is_apple_silicon()
    
    def _get_ram_gb(self) -> float:
        """Get total RAM in GB."""
        try:
            if self.os == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulong = ctypes.c_ulong
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", c_ulong),
                        ("dwMemoryLoad", c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return stat.ullTotalPhys / (1024 ** 3)
            
            elif self.os == "Linux":
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            return int(line.split()[1]) / (1024 ** 2)
            
            elif self.os == "Darwin":  # macOS
                result = subprocess.run(['sysctl', 'hw.memsize'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return int(result.stdout.split()[1]) / (1024 ** 3)
        except:
            pass
        
        return 8.0  # Default fallback
    
    def _get_gpu_info(self) -> str:
        """Get GPU information."""
        try:
            if self.os == "Windows":
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 
                                       'get', 'name'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout
            
            elif self.os == "Linux":
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'VGA' in line or 'Display' in line:
                            return line
            
            elif self.os == "Darwin":
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout
        except:
            pass
        
        return "Unknown GPU"
    
    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        if self.os != "Darwin":
            return False
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            return 'Apple' in result.stdout
        except:
            return False


class OllamaManager:
    """Manage Ollama installation and models."""
    
    MULTIMODAL_MODELS = {
        # Model: (min_ram_gb, min_vram_gb, description)
        "llama3.2-vision:90b": (64, 48, "Highest quality, requires powerful hardware"),
        "llama3.2-vision:11b": (16, 12, "High quality, good balance"),
        "llama3.2-vision": (8, 6, "Standard quality multimodal model"),
        "llava:34b": (32, 20, "High quality vision model"),
        "llava:13b": (16, 10, "Good quality vision model"),
        "llava:7b": (8, 6, "Efficient vision model"),
        "llava": (6, 4, "Efficient multimodal model (default)"),
    }
    
    @staticmethod
    async def is_installed() -> bool:
        """Check if Ollama is installed."""
        try:
            process = await asyncio.create_subprocess_exec(
                'ollama', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except FileNotFoundError:
            return False
    
    @staticmethod
    async def get_installed_models() -> list:
        """Get list of installed Ollama models."""
        try:
            process = await asyncio.create_subprocess_exec(
                'ollama', 'list',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            
            if process.returncode == 0:
                lines = stdout.decode().strip().split('\n')[1:]  # Skip header
                return [line.split()[0] for line in lines if line.strip()]
        except:
            pass
        return []
    
    @staticmethod
    def recommend_model(specs: SystemSpecs) -> Tuple[str, str]:
        """Recommend the best multimodal model for the system."""
        ram_gb = specs.ram_gb
        
        # Adjust for GPU acceleration
        effective_ram = ram_gb
        if specs.has_nvidia or specs.has_apple_silicon:
            effective_ram *= 1.5  # GPU can handle more
        
        # Find the best model that fits
        for model, (min_ram, min_vram, description) in OllamaManager.MULTIMODAL_MODELS.items():
            if effective_ram >= min_ram:
                return model, description
        
        # Fallback to smallest model
        return "llava:7b", "Efficient vision model (minimum requirements)"
    
    @staticmethod
    async def pull_model(model_name: str) -> bool:
        """Pull an Ollama model."""
        try:
            print(f"      Pulling model: {model_name}")
            print(f"      This may take several minutes...")

            process = await asyncio.create_subprocess_exec(
                'ollama', 'pull', model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT  # Combine stderr with stdout
            )

            # Stream output
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                output = line.decode().strip()
                if output:  # Only print non-empty lines
                    print(f"      {output}")

            await process.wait()

            if process.returncode != 0:
                print(f"      Model pull failed with exit code: {process.returncode}")
                print(f"      The model '{model_name}' may not exist or Ollama may have issues")
                print(f"      Try: ollama pull {model_name}")
                return False

            return True
        except Exception as e:
            print(f"      Error pulling model: {e}")
            return False


async def detect_and_recommend():
    """Main detection and recommendation function."""
    print("Detecting system specifications...")
    print()
    
    # Detect system specs
    specs = SystemSpecs()
    
    print(f"  OS: {specs.os}")
    print(f"  CPU Cores: {specs.cpu_count}")
    print(f"  RAM: {specs.ram_gb:.1f} GB")
    print(f"  GPU: {specs.gpu_info.strip()[:60]}")
    
    if specs.has_nvidia:
        print(f"  GPU Acceleration: NVIDIA CUDA")
    elif specs.has_apple_silicon:
        print(f"  GPU Acceleration: Apple Metal")
    elif specs.has_amd:
        print(f"  GPU Acceleration: AMD ROCm (limited support)")
    else:
        print(f"  GPU Acceleration: CPU only")
    
    print()
    
    # Check Ollama
    print("Checking Ollama installation...")
    ollama_installed = await OllamaManager.is_installed()
    
    if not ollama_installed:
        print("  Ollama is NOT installed")
        print()
        print("  To install Ollama:")
        print("    Windows/Mac: https://ollama.ai/download")
        print("    Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        return None, None
    
    print("  Ollama is installed")
    
    # Get installed models
    installed_models = await OllamaManager.get_installed_models()
    if installed_models:
        print(f"  Installed models: {', '.join(installed_models[:3])}")
        if len(installed_models) > 3:
            print(f"                    ... and {len(installed_models) - 3} more")
    
    print()
    
    # Recommend model
    recommended_model, description = OllamaManager.recommend_model(specs)
    
    print("Recommended multimodal model:")
    print(f"  Model: {recommended_model}")
    print(f"  Description: {description}")
    print()
    
    return recommended_model, installed_models


async def main():
    """Main function."""
    recommended_model, installed_models = await detect_and_recommend()
    
    if recommended_model is None:
        return {
            'ollama_installed': False,
            'recommended_model': None,
            'model_pulled': False
        }
    
    # Check if model is already installed
    if installed_models and recommended_model in installed_models:
        print(f"Model {recommended_model} is already installed!")
        return {
            'ollama_installed': True,
            'recommended_model': recommended_model,
            'model_pulled': True,
            'already_installed': True
        }
    
    # Ask to pull model
    print(f"Would you like to pull {recommended_model}? (y/N): ", end='', flush=True)
    
    # For automated scripts, check environment variable
    auto_pull = os.environ.get('AUTO_PULL_MODEL', '').lower() == 'true'
    
    if auto_pull:
        response = 'y'
        print('y (auto)')
    else:
        try:
            response = input().strip().lower()
        except:
            response = 'n'
    
    if response == 'y':
        print()
        success = await OllamaManager.pull_model(recommended_model)
        
        if success:
            print()
            print(f"Successfully pulled {recommended_model}!")
            return {
                'ollama_installed': True,
                'recommended_model': recommended_model,
                'model_pulled': True
            }
        else:
            print()
            print(f"Failed to pull {recommended_model}")
            return {
                'ollama_installed': True,
                'recommended_model': recommended_model,
                'model_pulled': False
            }
    else:
        print()
        print("Skipping model download. You can pull it later with:")
        print(f"  ollama pull {recommended_model}")
        return {
            'ollama_installed': True,
            'recommended_model': recommended_model,
            'model_pulled': False
        }


if __name__ == "__main__":
    result = asyncio.run(main())
    
    # Print result for scripting
    if result:
        print()
        print("=" * 60)
        print("RESULT:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print("=" * 60)

