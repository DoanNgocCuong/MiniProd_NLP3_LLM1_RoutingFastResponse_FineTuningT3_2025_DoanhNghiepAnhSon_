import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def csv_to_excel(csv_path):
    """
    Convert specific CSV file to Excel file with proper formatting
    
    Args:
        csv_path (str): Path to input CSV file
    """
    try:
        logger.info(f"Starting conversion of {csv_path}")
        
        # Check if file exists
        if not os.path.exists(csv_path):
            logger.error(f"File not found: {csv_path}")
            return
            
        logger.info("Reading CSV file...")
        # Read CSV file with proper encoding and quoting
        df = pd.read_csv(csv_path, encoding='utf-8', quoting=1)
        logger.info(f"Successfully read CSV with {len(df)} rows")
        
        # Generate excel path
        excel_path = os.path.splitext(csv_path)[0] + '.xlsx'
        logger.info(f"Will save to: {excel_path}")
            
        logger.info("Writing to Excel...")
        # Create Excel writer object with xlsxwriter engine
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            # Write dataframe to excel
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            
            logger.info("Formatting Excel columns...")
            # Set column widths
            worksheet.set_column('A:A', 40)  # Id column
            worksheet.set_column('B:B', 50)  # Name column
            worksheet.set_column('C:C', 15)  # Description column
            worksheet.set_column('D:D', 100) # Content column
            
            # Add text wrapping for content column
            wrap_format = workbook.add_format({'text_wrap': True})
            worksheet.set_column('D:D', 100, wrap_format)
        
        logger.info(f"Successfully converted {csv_path} to {excel_path}")
        logger.info(f"Output file size: {os.path.getsize(excel_path) / 1024:.2f} KB")
        
    except Exception as e:
        logger.error(f"Error converting file: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting CSV to Excel conversion script")
    
    # Get current directory
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")
    
    # List all CSV files in directory
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
    logger.info(f"Found CSV files: {csv_files}")
    
    # Convert the specific file
    csv_file = "data-1741773423502.csv"
    if csv_file in csv_files:
        csv_to_excel(csv_file)
    else:
        logger.error(f"Target file {csv_file} not found in current directory")
