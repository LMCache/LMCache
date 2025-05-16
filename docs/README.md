## Install Prerequisitesgiot
```
pip install -r requirements-docs.txt
```
## To build docs from source

First run 

``` 
make clean
```

To build HTML

```
make html
```

Serve documentation page locally

```
python -m http.server -d build/html/
```

#### Launch your browser and open localhost:8000.