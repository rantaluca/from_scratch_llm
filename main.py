from datasets import load_dataset 

def main():
    print("="*50)
    print("RobGPT - Chat Mode ")
    prompt = "What is your prompt?: \n"
    user_input = input(prompt)
    print("="*50)
    print(f"Asking: {user_input} to RobGPT")
    

if __name__ == "__main__":
    main()