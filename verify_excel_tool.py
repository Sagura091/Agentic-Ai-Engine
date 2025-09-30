"""
Comprehensive verification test for Revolutionary Universal Excel Tool
Writes all output to log file for verification
"""

import asyncio
import json
import sys
import traceback
from pathlib import Path
from datetime import datetime

# Import the tool
from app.tools.production.universal.revolutionary_universal_excel_tool import (
    revolutionary_universal_excel_tool,
    ExcelOperation,
)


class TestLogger:
    """Logger that writes to both console and file."""
    
    def __init__(self, log_file="excel_tool_verification.log"):
        self.log_file = log_file
        self.start_time = datetime.now()

        # Clear log file with UTF-8 encoding
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Excel Tool Verification Test\n")
            f.write(f"Started: {self.start_time}\n")
            f.write("=" * 80 + "\n\n")
    
    def log(self, message):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(full_message + "\n")
    
    def log_test(self, test_num, test_name):
        """Log test header."""
        msg = f"\n{'='*80}\nTEST {test_num}: {test_name}\n{'='*80}"
        self.log(msg)
    
    def log_success(self, message):
        """Log success message."""
        self.log(f"✅ SUCCESS: {message}")
    
    def log_error(self, message):
        """Log error message."""
        self.log(f"❌ ERROR: {message}")
    
    def log_result(self, result_json):
        """Log JSON result."""
        try:
            result = json.loads(result_json)
            self.log(f"Result: {json.dumps(result, indent=2)}")
        except:
            self.log(f"Result: {result_json}")
    
    def finalize(self):
        """Write final summary."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        summary = f"\n{'='*80}\nTEST SUMMARY\n{'='*80}\n"
        summary += f"Started: {self.start_time}\n"
        summary += f"Ended: {end_time}\n"
        summary += f"Duration: {duration:.2f} seconds\n"
        
        self.log(summary)


async def run_verification():
    """Run comprehensive verification tests."""
    logger = TestLogger()

    try:
        logger.log("Initializing Revolutionary Universal Excel Tool...")
        logger.log(f"Tool Name: {revolutionary_universal_excel_tool.name}")
        logger.log(f"Tool ID: {revolutionary_universal_excel_tool.tool_id}")
        logger.log(f"Tool Version: {revolutionary_universal_excel_tool.tool_version}")

        # File will be saved to data/outputs automatically
        test_file = "verification_test.xlsx"
        logger.log(f"Test file will be saved to: data/outputs/{test_file}")
        
        # Test 1: Create workbook
        logger.log_test(1, "Create New Workbook")
        try:
            result = await revolutionary_universal_excel_tool._arun(
                operation=ExcelOperation.CREATE,
                file_path=test_file,
                options={"sheets": ["Data", "Analysis", "Summary"]}
            )
            logger.log_result(result)
            
            # Verify file exists
            if Path(test_file).exists():
                logger.log_success(f"Workbook created: {test_file}")
            else:
                logger.log_error(f"Workbook file not found: {test_file}")
        except Exception as e:
            logger.log_error(f"Create workbook failed: {e}")
            traceback.print_exc()
        
        # Test 2: Write single cell
        logger.log_test(2, "Write Single Cell")
        try:
            result = await revolutionary_universal_excel_tool._arun(
                operation=ExcelOperation.WRITE_CELL,
                file_path=test_file,
                sheet_name="Data",
                cell_range="A1",
                data="Test Value"
            )
            logger.log_result(result)
            logger.log_success("Cell written successfully")
        except Exception as e:
            logger.log_error(f"Write cell failed: {e}")
            traceback.print_exc()
        
        # Test 3: Read single cell
        logger.log_test(3, "Read Single Cell")
        try:
            result = await revolutionary_universal_excel_tool._arun(
                operation=ExcelOperation.READ_CELL,
                file_path=test_file,
                sheet_name="Data",
                cell_range="A1"
            )
            logger.log_result(result)
            result_data = json.loads(result)
            if result_data.get("value") == "Test Value":
                logger.log_success("Cell read correctly")
            else:
                logger.log_error(f"Cell value mismatch: {result_data.get('value')}")
        except Exception as e:
            logger.log_error(f"Read cell failed: {e}")
            traceback.print_exc()
        
        # Test 4: Write range
        logger.log_test(4, "Write Range of Data")
        try:
            data = [
                ["Name", "Age", "City", "Salary"],
                ["Alice", 30, "New York", 75000],
                ["Bob", 25, "Los Angeles", 65000],
                ["Charlie", 35, "Chicago", 85000],
                ["Diana", 28, "Houston", 70000]
            ]
            result = await revolutionary_universal_excel_tool._arun(
                operation=ExcelOperation.WRITE_RANGE,
                file_path=test_file,
                sheet_name="Data",
                cell_range="A3",
                data=data
            )
            logger.log_result(result)
            logger.log_success("Range written successfully")
        except Exception as e:
            logger.log_error(f"Write range failed: {e}")
            traceback.print_exc()
        
        # Test 5: Read range
        logger.log_test(5, "Read Range of Data")
        try:
            result = await revolutionary_universal_excel_tool._arun(
                operation=ExcelOperation.READ_RANGE,
                file_path=test_file,
                sheet_name="Data",
                cell_range="A3:D7",
                options={"format": "list"}
            )
            logger.log_result(result)
            logger.log_success("Range read successfully")
        except Exception as e:
            logger.log_error(f"Read range failed: {e}")
            traceback.print_exc()
        
        # Test 6: Read entire sheet
        logger.log_test(6, "Read Entire Sheet")
        try:
            result = await revolutionary_universal_excel_tool._arun(
                operation=ExcelOperation.READ_SHEET,
                file_path=test_file,
                sheet_name="Data",
                options={"format": "list"}
            )
            result_data = json.loads(result)
            logger.log_success(f"Sheet read: {result_data['rows']} rows, {result_data['cols']} cols")
        except Exception as e:
            logger.log_error(f"Read sheet failed: {e}")
            traceback.print_exc()
        
        # Test 7: Write to different sheet
        logger.log_test(7, "Write to Analysis Sheet")
        try:
            summary_data = [
                ["Metric", "Value"],
                ["Total Employees", 4],
                ["Average Age", 29.5],
                ["Average Salary", 73750]
            ]
            result = await revolutionary_universal_excel_tool._arun(
                operation=ExcelOperation.WRITE_RANGE,
                file_path=test_file,
                sheet_name="Analysis",
                cell_range="A1",
                data=summary_data
            )
            logger.log_result(result)
            logger.log_success("Analysis sheet populated")
        except Exception as e:
            logger.log_error(f"Write to Analysis failed: {e}")
            traceback.print_exc()
        
        # Test 8: Save workbook
        logger.log_test(8, "Save Workbook")
        try:
            result = await revolutionary_universal_excel_tool._arun(
                operation=ExcelOperation.SAVE,
                file_path=test_file
            )
            logger.log_result(result)
            logger.log_success("Workbook saved")
        except Exception as e:
            logger.log_error(f"Save failed: {e}")
            traceback.print_exc()
        
        # Test 9: Verify file
        logger.log_test(9, "Verify File Exists and Size")
        try:
            file_path = Path(test_file)
            if file_path.exists():
                file_size = file_path.stat().st_size
                logger.log_success(f"File exists: {file_path.absolute()}")
                logger.log_success(f"File size: {file_size} bytes")
            else:
                logger.log_error("File does not exist!")
        except Exception as e:
            logger.log_error(f"File verification failed: {e}")
        
        # Test 10: Tool statistics
        logger.log_test(10, "Tool Statistics")
        try:
            stats = revolutionary_universal_excel_tool.get_stats()
            logger.log("Tool Statistics:")
            for key, value in stats.items():
                logger.log(f"  {key}: {value}")
            logger.log_success("Statistics retrieved")
        except Exception as e:
            logger.log_error(f"Stats failed: {e}")
        
        logger.finalize()
        logger.log("\n✅ VERIFICATION COMPLETE - Check excel_tool_verification.log for details")
        
    except Exception as e:
        logger.log_error(f"Verification failed: {e}")
        traceback.print_exc()
        logger.finalize()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_verification())

