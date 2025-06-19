import os

def get_class_mapping(prompt_num):
    print(f"\nEnter class mapping #{prompt_num}")
    class_from = int(input("Classes FROM: "))
    class_to = int(input("Classes TO: "))
    return class_from, class_to

def modify_txt_files(folder, mappings):
    folder_path = os.path.join(os.getcwd(), folder)

    if not os.path.isdir(folder_path):
        print(f"Folder '{folder}' not found!")
        return

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    class_id = int(parts[0])
                    for old_class, new_class in mappings:
                        if class_id == old_class:
                            parts[0] = str(new_class)
                            break
                new_lines.append(" ".join(parts) + "\n")

            with open(file_path, 'w') as f:
                f.writelines(new_lines)

    print(f"All .txt files in '{folder}' updated successfully.")

if __name__ == "__main__":
    folder_name = input("Enter the folder name (e.g. AB): ")

    mapping1 = get_class_mapping(1)
    mapping2 = get_class_mapping(2)

    modify_txt_files(folder_name, [mapping1, mapping2])
