



# Status
Functional code


# structure
```bash
.
├── data                   # datasets
├── src                    # source code 
│   ├── configs            # configs files, to edit for new experiments
│   ├── features           # folder to process the features for the classification pipeline ***
│   ├── ml                 # ML pipeline *** 
│   ├── ml                 # Pattern mining ***
├── notebooks              # notebooks to plot the results
├── results                # automatically generated results folder
└── README.md              # Hello World
```


# Run
## Classification
- add your dataset in the data folder
- add your own sequencer in src/features/sequencers, such that you can use it in ```pipeline_maker.py```
- edit ```pipeline_maker.py``` to add your dataset as an option when using the config file
- add your own models (if needed) in ml/models/ based on the model superclass such that it can be used in ```xval_maker.py```

run ```python script_classification.py --seeds```

## Pattern mining
- add your dataset in the data folder
- add your own sequencer in src/pattern_mining/features/sequencers, such that you can use it in ```pm_pipeline.py ```
- edit ```data_pipeline``` to add your dataset as an option when using the config file
- add your own models (if needed) in ml/models/ based on the model superclass such that it can be used in ```pm_pipeline```

- all files with the word "config" in also need to be edited to make the pipeline run !

run ```pyton script_patternmining.py --mining --sequences```

#  Files information
```exampledataset_config.yaml```
This file is used to decide what across which demographic groups the differential pattern mining algorithm need to be, and gives information about where to find the dataset, and what demographic attributes are available.

```pipeline_maker``` 
Used to load the sequences
