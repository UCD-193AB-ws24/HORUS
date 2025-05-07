# Local development server

## To Start Local Dev Server ##

- Make sure you are cd'd in to server, if not
```bash
cd server
```
- Run the local server on http
``` bash
 uvicorn main:app --host 0.0.0.0 --port 8001
 ```


- Run the local server on https (doesnt work yet)
``` bash
 uvicorn main:app --host 0.0.0.0 --port 8001 --ssl_keyfile ./certs/key.pem --ssl_certfile ./certs/cert.pem
 ```

- Run the ngrok tunnel
 ```bash
 ngrok http 8001
 ```

- Watch the console for any HTTP errors