import pandas as pd
import re
import argparse
import sys
from typing import Dict, List

def load_excel(file_path: str) -> pd.DataFrame:
    """
    Load Excel file using pandas
    """
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        raise Exception(f"Error loading Excel file: {str(e)}")

def convert_urls(df: pd.DataFrame, domain: str = "https://www.novulo.com") -> pd.DataFrame:
    """
    Convert URLs based on content type and create full URLs including domain
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    domain (str): Domain name to prepend to URLs
    """
    def create_slug(title: str) -> str:
        # Convert to lowercase and replace spaces with hyphens
        slug = title.lower()
        # Remove special characters and replace spaces with hyphens
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'[\s-]+', '-', slug)
        # Remove any double hyphens that might have been created
        slug = re.sub(r'-+', '-', slug)
        return slug.strip('-')
    
    def convert_url(row: pd.Series) -> str:
        title = row['Title']
        content_type = row['Type']
        
        # Create base slug from title
        slug = create_slug(title)
        
        # Convert based on content type
        if content_type.lower() == 'blog':
            path = f'/blog/{slug}'
        elif content_type.lower() == 'nieuws':
            path = f'/nieuws/{slug}'
        else:
            # For other content types, keep as is but with proper slug
            path = f'/{content_type.lower()}/{slug}'
            
        # Create full URL with domain
        full_url = f"{domain}{path}"
        
        return full_url
    
    # Create new column with converted URLs while keeping original title
    df_result = df.copy()
    df_result['New_URL'] = df.apply(convert_url, axis=1)
    
    return df_result

def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(description='Convert URLs in Excel file based on content type.')
    parser.add_argument('input_file', help='Path to input Excel file')
    parser.add_argument('--output_file', help='Path to output Excel file (optional)')
    parser.add_argument('--domain', default='https://www.novulo.com', help='Domain name (default: https://www.novulo.com)')
    
    args = parser.parse_args()
    
    try:
        print(f"Loading Excel file: {args.input_file}")
        df = load_excel(args.input_file)
        
        print("Converting URLs...")
        df_with_new_urls = convert_urls(df, domain=args.domain)
        
        # Display sample of conversions
        print("\nSample URL Conversions:")
        print(df_with_new_urls[['Title', 'Type', 'New_URL']].head().to_string())
        
        # Save to Excel if output file is specified
        if args.output_file:
            print(f"\nSaving results to: {args.output_file}")
            df_with_new_urls.to_excel(args.output_file, index=False)
            print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()