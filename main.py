from app.routers.chat import generate_prompt_parts
import asyncio

while True:
    input_text = input("Enter your text: ")

    generate_prompt_parts(input_text)

    if input_text == "exit":
        break


print("Goodbye!")


# create virtual environment
# python3 -m venv .venv

# activate virtual environment
# source .venv/bin/activate

# install requirements
# pip install -r requirements.txt
