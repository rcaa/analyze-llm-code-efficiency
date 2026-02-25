# Correctness, Execution Time, and Memory Usage of LLM-Generated Code: A Large-Scale Empirical Study on LeetCode

## Authors

- Adenilson Ramos [adenilson.ramos@ufape.edu.br](mailto:adenilson.ramos@ufape.edu.br)
- Rodrigo Andrade [rodrigo.andrade@ufape.edu.br](mailto:rodrigo.andrade@ufape.edu.br)

## Abstract

The integration of Large Language Models (LLMs) into software development has shifted automated code generation from a theoretical possibility to a practical necessity for modern developers. However, as these models are increasingly tasked with solving complex algorithmic problems, the focus must expand beyond simple functional correctness to include non-functional requirements such as computational efficiency and resource constraints. To address this, we conduct a large-scale empirical evaluation of 11 freely available LLMs hosted on the Groq Cloud platform. Using LeetCode as a standardized evaluation environment, we analyzed 10,666 judged submissions to assess how these models perform across a stratified sample of 336 algorithmic problems varying in difficulty (Easy, Medium, and Hard). Our investigation covers three critical dimensions: acceptance rate, execution time, and memory consumption across C++, Java, and Python3. The results reveal an overall acceptance rate of 42.7\%, with performance disparities between models. We found that while model choice is the primary determinant of whether a solution is correct, the choice of programming language is the main driver of execution efficiency. Compiled languages (C++ and Java) achieved higher acceptance and faster execution, whereas Python3 proved to be the most memory-efficient. Furthermore, we find a correlation between acceptance and memory usage, which suggests that models producing more correct solutions often generate more resource-intensive code. These findings provide a benchmark for selecting LLMs based on specific requirements, whether the priority is reliability, execution time, or a low memory consumption.


## Repository Structure

- `/datasets`: Databases and questions used for the experiments.
- `/data`: Raw experiment results, logs, and code execution outputs.
    - `/{modelo-llm}`: 
        - `/{language}`:  
- `/src`: Source code organized by function:
    - `/scraping`: Scripts for collecting questions and data.
    - `/selection`: Algorithms for problem and challenge selection.
    - `/generation`: Scripts for code generation using LLMs.
    - `/evaluation`: Scripts for evaluating energy consumption, memory, and performance.
- `/results`: Final analyses, tables, graphs, and reports interpreting the results.
- `/docs`: Usage instructions and produced papers/articles.
- `/notebooks`: Jupyter Notebooks for interactive data exploration.


---

