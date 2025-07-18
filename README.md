# Software and data

## Warning

In most of our code we used pickle.
We later noticed we could have used torch.save() and torch.load() instead,
which is more trustworthy and avoids trouble with different devices.
We still submit the code with pickle for consistency with the data we saved.
We plan to make it compatible with both in a later version.

We do not submit any of the actual pickle data.
So feel free to change your local version of the code.

## Structure

* ```TransformerLens``` (submodule): a fork of TransformerLens that supports OLMo.
* ```neuroscope``` (submodule): Code and results for the activation-based neuron analyses in Section 6. See the separate repo for more information.
* ```wcos_casestudies```: Code for most experiments
  * ```utils.py``` and ```plotting.py``` contain helper functions
  * ```main.py```: Run this to reproduce Section 5. There are options for each sub-experiment (see the argparse part of the code).
  * ```casestudies.py```: If you open this in VS Code, you will be able to run it cell by cell like a Jupyter notebook. It first generates the data for Appendix E (IO classes vs. functional roles), then selects the neurons for the case studies of Section 6 and does the weight-based part of the case studies.
  * ```selected_plot.py```: reproduce Figure 5 (selected layers of Llama).
  * ```defplot.py```: reproduce Figure 2 (definition plot).
* ```interactive.ipynb```: interactive vector visualisations of IO classes.

## Steps to reproduce

### Environment

First, create your environment and install requirements:

```[bash]
conda create -n wcos python==3.12.9
conda activate wcos
pip install -r requirements.txt
git submodule init --recursive
cd TransformerLens
pip install -e .
cd ..
```

### Section 5 (IO functionalities by layer)

```[bash]
cd wcos_casestudies
python main.py --categories --category_stats --quartiles --plot_fine --plot_coarse --plot_quartiles --plot_all_medians --model all
```

or any subset of these options / specific model names.

### Section 6 (Case studies)

#### Appendix E: IO classes vs. functional roles

For our case studies we limit the search space to prediction neurons,
so we first need to find out which these are.

Open ```casestudies.py``` in VS Code.
You will be able to run it cell by cell like a Jupyter Notebook.

#### Choosing neurons

Continue running the same script.

#### Weight-based part of the case studies

Final cell of the same script.

#### Activation-based part

See submodule ```neuroscope```.
