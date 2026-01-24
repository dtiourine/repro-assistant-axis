import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

def main():
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    page = client.models.list()
    
    for i in range(len(page.data)):
        page_i = page.data[i]
        print(page_i.id)
    
if __name__ == "__main__":
    main()