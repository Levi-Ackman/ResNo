import os
import argparse

def execute_sh_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".sh"):
                sh_file_path = os.path.join(root, file)
                print(f"Executing {sh_file_path}")
                os.system(f"chmod +x {sh_file_path}")  # Ensure the script is executable
                os.system(sh_file_path)
def main():
    parser = argparse.ArgumentParser(description="Execute .sh files in a specified directory")
    parser.add_argument("--opt", choices=[1,2,3,4,5],type=int, required=True, help="Specify the option")
    args = parser.parse_args()
    if args.opt == 1:
        scripts_directory = "MTSF/visual"
    elif args.opt == 2:
        scripts_directory = "MTSF/2"
    elif args.opt == 3:
        scripts_directory = "scripts/kmeans100"
    elif args.opt == 4:
        scripts_directory = "scripts/schaefer100"
    elif args.opt == 5:
        scripts_directory = "scripts/ward100"
    else:
        print("Invalid option specified.")
        return
    execute_sh_files_in_directory(scripts_directory)

if __name__ == "__main__":
    main()
