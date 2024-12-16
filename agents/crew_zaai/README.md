# AI Agents

**Steps to run the code:**
1. Create a virtual environment with python 3.10.15
`conda create --name myenv python=3.10.15`
3. Activate in your new virtual environment
`conda activate myenv`
4. Install the required requirements
    - `pip install -r requirements.txt`
5. Create a folder called `assets/` to store the results
6. Add an .env file under `ai-agents/crew_zaai/src/crew_zaai/.env` with:
   ```
    YOUTUBE_API_KEY=<Your key>
    OPENAI_API_KEY=<Your key>
    SEARXNG_BASE_URL=https://search.zaai.ai
   ```
7. Run the main.py file

## Folder Structure:
------------

    ├── crew_zaai
    │
    ├──── src/crew_zaai
    ├────── assets                      <- results
    ├────── config                      <- agent and taks definition files
    ├────── tools                       <- tools to be used by the agents
    │
    │────── .env                        <- file with enviroment variables
    │
    │────── requirements.txt            <- package version for installing
    │
    │────── crew.py                     <- crew definition
    └────── main.py                     <- file to generate the blog post
--------
