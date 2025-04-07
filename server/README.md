# Local development server

## To Start Local Dev Server ##

- Make sure you are cd'd in to server, if not
```bash
cd server
```
- Run the local server
``` bash
 uvicorn main:app --host 127.0.0.1 --port 8000 --reload
 ```

- Watch the console for any HTTP errors