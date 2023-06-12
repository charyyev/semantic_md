# Semantic-MD: Infusing Monocular Depth with Semantic Signals
This is the official GitHub of the "Semantic-MD: Infusing Monocular Depth with Semantic Signals" project.

Contributors: [Sapar Charyyev](https://github.com/charyyev), [Ankita Ghosh](https://github.com/ankitaghosh9), [Oliver Lemke](https://github.com/oliver-lemke)

---

# Usage
### Setting up dependencies
Setup virtual environment:
```shell
python3 -m venv venvs
source venvs/bin/activate
cd path/to/repo
pip install -r requirements.txt
```
Download pretrained weights:
```shell
python source/scripts/download_model_weights.py
```
This will create a new folder ```pretrained_weights```.

### Downloading and formatting dataset
```shell
python HyperSim_Data/download_data.py
```
This will download data the required xxx.hdf5 files in the ```HyperSim_Data``` folder.

**NOTE**: The data is approximately 94GB and will take upto several hours to download

```shell
python source/script/extract_data.py
```
This will convert the xxx.hdf5 files into xxx.npy files and store them in a new directory ```HyperSim_Data_extracted``` and
create new path files inside ```source/datasets/paths```

**NOTE**: Before executing this file please reformat the *root path part* of the paths in files ```source/datasets/paths/xxx.txt``` <br>
(*root path part*: /cluster/project/infk/courses/252-0579-00L/group22_semanticMD/)

To download a small sample of the dataset for testing purposes: ``` ./source/scripts/local_hypersim.sh```

If you only want to test visualization, please copy the test_data/ folder into your project directory and specify the relative path with subpaths/vis_data in the config.

### Setting up the config file
Please set up a ```configs/user.yaml``` config file to specify your local paths. Please refer to the ```configs/template.yaml``` for instructions.
This project cannot run without it.

### Running the project
Please make sure that your root path is set correctly. All import paths are relative to source/.

#### Visualizing Models
Download model weights from [here](https://drive.google.com/file/d/1d0y0q_3sTj6Ba6uAT5quSMMQaEa-X59u/view?usp=sharing). <br>
Specify the path of the downloaded model weights in `visualize` parameter of the config files.

We provide our best performing models in the above link. In need of testing other methods, please contact us for the required models.

Please run `main_vis.py` in scripts/vis/maim/.

#### Training Models
Please run `python3 main.py`

`--config <entry_config.yaml>` specifies the config yaml to be used as an entry point. Default is user.yaml.
If you follow the config setup as specified in the template, this option needs not be set.

---

# Project Setup

For more specific details on each file and the associated code, please check the files for comments.

<pre>
project_root_dir/ 				<--- root directory of the project
├── source/ 					<--- all code stored here
│   ├── datasets/
│   │   ├── hypersim_dataset.py 		<--- contains the dataset implmentation for the HyperSim dataset
│   │   └── ...
│   ├── models/
│   │   ├── __init__.py 			<--- contains the model_factory which is responsible for building a model
│   │   ├── template_model.py 			<--- template for how a model should look like
│   │   ├── specialized_networks/ 		<--- use this folder for special changes to the network
│   │   │   ├── special_example.py 		<--- example for such a network change
│   │   │   └── ...
│   ├── scripts/ 				<--- contains scripts to automate certain tasks, mostly not relevant to the final execution of the project
│   │   ├── vis/ 				<--- scripts for running visualization
│   │   │   ├── main_vis.py 			<--- run this script to run visualization
│   │   │   └── x_visualizer.py 		<--- various implementations of the visualizer class; for more info check comments in file
│   │   │   └── ...
│   ├── trainer.py 				<--- contains the trainer class implementations
│   │   ├── base_trainer.py 			<--- base class implementation of the trainer class, can be extended 
│   │   ├── multi_loss_trainer.py 		<--- one such extension of the base trainer; takes care of training multi_loss (2 heads) model
│   │   └── ...
│   ├── utils/
│   │   ├── configs.py 				<--- ease of use class for accessing config
│   │   ├── conversions.py 			<--- implements methods of converting semantic map as seen in the paper
│   │   ├── eval_metrics.py 			<--- additional metrics to keep track of
│   │   ├── logs.py 				<--- project-specific logging configuration
│   │   ├── loss_functions.py 			<--- implementation of additional loss functions
│   │   ├── transforms.py 			<--- various transformations of image data
│   │   └── ...
│   ├── main.py 				<--- contains the main method
│   └── ...
│
├── configs/
│   ├── base.yaml 				<--- base config file used for changing the actual project
│   ├── template.yaml 				<--- template config for setting up user.yaml
│   └── user.yaml 				<--- personal config file to set up config for this specific workspace
│
├── logs/ 					<--- contains logs
│   └── ...
│
├── data/ 					<--- contains any used datasets
│   └── ...
│
├── pretrained_weights/ 			<--- contains model_weights
│   ├── template_weights/ 			<--- template configuration
│   │   ├── weights.pth 			<--- actual weights for the model
│   │   └── weights_object.pickle 		<--- metadata (config used for pretraining)
│
├── output/ 					<--- any model output
│   ├── template_output/
│   │   ├── best_checkpoints/
│   │   │   └── ... 				<--- explanation of checkpoint structure under checkpoints/
│   │   ├── checkpoints/
│   │   │   ├── epoch_x/ 			<--- model weights at checkpoint
│   │   │   │   ├── optimizer.pth 		<--- optimizer state at checkpoint
│   │   │   │   └── weights.pth 		<--- model weights at checkpoint
│   │   └── tensorboard/  			<--- tensorboard directory
│   │   └── wandb/ 				<--- wandb directory
│
├── .github/                                        
│   ├── workflows/ 				<--- github actions 
│   │   ├── black.yml
│   │   ├── isort.yml
│   │   ├── pylint.yml
│   │   └── ...
│
├── .gitignore 					<--- global .gitignore
├── requirements.txt
└── README.md
</pre>

---

# GitHub Actions
This project uses [black](https://pypi.org/project/black/) and
[isort](https://pypi.org/project/isort/) for formatting, and
[pylint](https://pypi.org/project/pylint/) for linting.

## PyCharm setup
1. Download the [File Watchers](https://www.jetbrains.com/help/pycharm/using-file-watchers.html)
   Plugin
2. Under Settings > Tools > File Watcher > + > \<custom>: setup a new watcher for each
	1. black
		- Name: Black Watcher
		- File type: Python
		- Scope: Project Files
		- Program: $PyInterpreterDirectory$/black
		- Arguments: $FilePath$
		- Output paths to refresh: $FilePath$
		- Working directory: $ProjectFileDir$
		- Additional: as wished
	2. isort
		- Name: iSort Watcher
		- Program: $PyInterpreterDirectory$/isort
		- Arguments: $FilePath$ --sp $ContentRoot$/.style/.isort.cfg --settings-path $ProjectFileDir$/pyproject.toml
	3. pylint
		- Name: PyLint Watcher
		- Program: $PyInterpreterDirectory$/pylint
		- Arguments: --msg-template="$FileDir$/{path}:{line}:{column}:{C}:({symbol}){msg}" $FilePath$ --rcfile $ProjectFileDir$/pyproject.toml

---

# Contribution
All works is ours except if otherwise indicated in the respective file.
(
- HyperSim_Data/download.py
- source/scripts/extract_data.py
)
