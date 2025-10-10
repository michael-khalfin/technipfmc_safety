# set up
1. first make sure you are in (.venv) and python 3.11 
 - make venv
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
 - install dependency
    python -m pip install -U pip
    pip install graphrag pyarrow spacy
 - download spacy
    python -m spacy download en_core_web_md

2. deploy ollama
 - ollama serve
 - ollama pull nomic-embed-text
 - ollama pull mistral

## check whether ollama is work
Invoke-WebRequest http://localhost:11434/api/tags | Select-Object -ExpandProperty StatusCode
here should return 200, then kill this test progress
netstat -ano | findstr 11434
taskkill /PID <PID we seen before> /F


# !!! how to run (in terminal)
graphrag index --root .\christmas --config .\christmas\settings.yaml

graphrag query --root .\christmas --config .\christmas\settings.yaml --method local --query "Who is Scrooge and what are his main relationships?"

## another way to run (but i m not sure it really works)
python -m graphrag.cli.main index --root .\christmas --config .\christmas\settings.yaml --method fast --verbose

## query output see: logs/query_example.log