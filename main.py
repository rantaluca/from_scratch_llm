from datasets import load_dataset 

def main():

    print("RobGPT - Main ")
    prompt = "What is your prompt?: \n"
    user_input = input(prompt)
    print(f"Asking: {user_input} to RobGPT")
    

if __name__ == "__main__":
    main()