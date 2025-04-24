from openai import AsyncOpenAI, RateLimitError
from dotenv import load_dotenv
import os
from halo import Halo
import asyncio
import backoff
import time
import base64

load_dotenv()

# setup the OpenAI Client
client = AsyncOpenAI()

# Azure OpenAI variables from .env file
OPENAI_MODEL = os.environ.get("OPENAI_MODEL")

def open_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as infile:
        return infile.read()
    
###  OpenAI chat completions call with backoff for rate limits
@backoff.on_exception(backoff.expo, RateLimitError)
async def chat(**kwargs):
    try:
        spinner = Halo(text="Calling gpt-image-1 API...", spinner="dots")
        spinner.start()
        # print(kwargs)

        start_time = time.time()  # Record the start time
        
        img = await client.images.generate(**kwargs)

        end_time = time.time()  # Record the end time

        elapsed_time = end_time - start_time  # Calculate the elapsed time in seconds
        minutes, seconds = divmod(
            elapsed_time, 60
        )  # Convert seconds to minutes and seconds
        formatted_time = (
            f"{int(minutes)} minutes and {seconds:.2f} seconds"  # Format the time
        )
        
        tokens = img.usage

        spinner.stop()

        return img, tokens, formatted_time
    except Exception as yikes:
        print(f'\n\nError communicating with OpenAI: "{yikes}"')
        exit(0)


async def main():
    response_id = ""
    while True:
        # Get user query
        query = input(f"\nGenerate an Image with gpt-image-1 (using model: {OPENAI_MODEL}): ")
        if query.lower() == "exit":
            exit(0)

        #prompt = open_file("./prompts/prompt.xml")
        
        # Build the kwargs for the chat call
        chat_kwargs = {
            "model": OPENAI_MODEL,
            "prompt": query,
            "n": 1,
            "size": "1024x1024",
            #"response_format": "b64_json",
        }

        img, tokens, formatted_time = await chat(**chat_kwargs)

        print(f"Time elapsed: {formatted_time}")
        print(f"Your question took a total of: {tokens.total_tokens} tokens")
        print(f"Your question prompt used: {tokens.input_tokens}")
        print(f"Your image output used: {tokens.output_tokens}")
        
        image_bytes = base64.b64decode(img.data[0].b64_json)
        with open("output.png", "wb") as f:
            f.write(image_bytes)
        
if __name__ == "__main__":
    asyncio.run(main())
