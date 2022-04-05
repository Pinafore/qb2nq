# qb2nq
![GitHub repo size](https://img.shields.io/github/repo-size/Pinafore/qb2nq?logo=Files&logoColor=white&style=for-the-badge)
![Lines of code](https://img.shields.io/tokei/lines/github/Pinafore/qb2nq?color=royalblue&logo=Visual%20Studio%20Code&style=for-the-badge)
![GitHub language count](https://img.shields.io/github/languages/count/Pinafore/qb2nq?color=lightgreen&style=for-the-badge)
![GitHub contributors](https://img.shields.io/github/contributors/Pinafore/qb2nq?color=lightgreen&style=for-the-badge)

qb2nq (*/ˈkæməl/*, **Q**uiz**B**owl **2** **N**atural **Q**uestions transformation) is a project to transform complicated trivia questions in the quizbowl dataset to simpler Natural Questions (NQ) dataset for better Question-Answering (QA) performance.

### Execution steps

Please run our code by git cloning the repository, then changing directory to our repository followed by the following commands. 

With Script: 
```
make clean
make
python compute_lat_frequency.py
python transform_question.py
python test_transform.py
```

You can run compute_lat_frequency.py to generate your lat_frequency.json code that is used by transform_question.py. We have provided a default lat_frequency.json to test our code faster.

The default --limit argument in transform_question.py is 20 which means our code will convert 20 QB questions to NQ-like questions. Please set --limit argument to -1 for converting the entire QB dataset to NQ-like dataset. 

You can also run the quality_classifier.py to score the generated NQ-like questions based on their quality.
