# Work of Tom Engelmann and Rishikesh Bharti

Our work is part of the course: M. Grum: Advanced AI-based ApplicaGon Systems

We created a sarcasm detection model, based on the iSarcasmEval dataset. Our repo constists of a CNN-Like neural network and an OLS model to predict sarcasm.
We provided docker-compose files to run our code an making inferences.

Simply start the containers by:

 ```docker-compose -f ./images/docker-compose.ai.yml up -d ```

 Access to codebas_ai container:

```docker exec -it codebase_ai /bin/sh``` 

Execute the ols and ann model

```python apply_annSolution.py```

or

```python apply_olsSolution.py```

This will start the inference scripts inside the container, using our trained models
