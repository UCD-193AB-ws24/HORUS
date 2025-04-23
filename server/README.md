# Local development server

## To Start Local Dev Server ##

- Make sure you are cd'd in to server, if not
```bash
cd server
```
- Run the local server
``` bash
 uvicorn main:app --host 0.0.0.0 --port 8001 --ssl_keyfile ./certs/key.pem --ssl_certfile ./certs/cert.pem
 ```

- Watch the console for any HTTP errors