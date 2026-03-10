# acuhit-predictive-engine

this is our hackathon project. we built this clinical ai pipeline in just 48 hours. 
it's a fast-paced work in progress, but we designed it to be scalable and highly reliant on speedy data processing frameworks.

## what it does
the engine processes continuous flows of tabular and textual data, structuring it for machine learning applications. we abstract away domain-specific labels into generic temporal and categorical outcome variables, predicting general sequence outcomes over a given time horizon. 

we used duckdb and polars because we needed extreme performance to handle multi-million row datasets reliably under severe time constraints.

## architecture
- `src/`: contains all the core generic processor modules, from raw data formatting to automated feature engineering.
- `config/`: settings and metadata configurations.
- `data/`: local storage for processing buffers (not tracked).

## how to run
install the dependencies and execute the phased data processor scripts to construct the outcome prediction matrix.

we had lots of fun (and coffee) building this.