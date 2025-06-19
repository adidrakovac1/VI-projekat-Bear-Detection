import os
import sys

def filter_yolo_annotations():
    """
    This script filters YOLO annotation files (.txt) to keep only specified classes.
    It processes all .txt files in a user-specified subfolder.
    """
    # --- 1. Get user input for the folder name ---
    folder_name = input("Enter the name of the folder containing .txt and .jpg files: ")

    # --- 2. Validate that the folder exists ---
    if not os.path.isdir(folder_name):
        print(f"\nError: Folder '{folder_name}' not found.")
        print("Please make sure the script is in the same directory as this folder.")
        sys.exit(1) # Exit the script if the folder doesn't exist

    # --- 3. Get user input for the classes to keep ---
    classes_input = input("Enter the class numbers you want to KEEP (separated by spaces, e.g., '0 2 5'): ")

    try:
        # Convert the input string of numbers into a set of integers for fast lookups
        # Example: "0 2 5" becomes {0, 2, 5}
        desired_classes = set(int(c) for c in classes_input.split())
    except ValueError:
        print("\nError: Invalid input. Please enter only numbers separated by spaces.")
        sys.exit(1)

    print(f"\nFiltering started. Will only keep classes: {sorted(list(desired_classes))}")
    print(f"Processing folder: '{folder_name}'...\n")

    processed_files_count = 0
    # --- 4. Iterate over all files in the specified folder ---
    for filename in os.listdir(folder_name):
        # We only care about the .txt annotation files
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_name, filename)
            
            lines_to_keep = []

            try:
                # Open the file for reading
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # Check each line in the file
                for line in lines:
                    # Skip empty lines
                    if line.strip() == "":
                        continue
                    
                    # The first number on the line is the class ID
                    parts = line.strip().split()
                    class_id = int(parts[0])

                    # --- 5. The core logic: check if the class is desired ---
                    if class_id in desired_classes:
                        lines_to_keep.append(line)

                # --- 6. Overwrite the original file with the filtered lines ---
                with open(file_path, 'w') as f:
                    f.writelines(lines_to_keep)
                
                print(f"  - Processed: {filename}")
                processed_files_count += 1

            except Exception as e:
                print(f"  - Error processing file {filename}: {e}")

    print(f"\nFiltering complete. Processed {processed_files_count} .txt files in '{folder_name}'.")


# This makes the script runnable
if __name__ == "__main__":
    filter_yolo_annotations()