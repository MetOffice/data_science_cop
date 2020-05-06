# Data Science Community of Practice
This is the repository for  example and tutorial matieral created through the Met Office Data Science Community of Practice.


## Running the code
There are several ways to set up an environment to run the code in this repository:

### Clone this repository with git

* In a terminal, clone the repository using git<br>
``git clone https://github.com/MetOffice/data_science_cop.git``


### Conda environment -> (click [here](https://docs.conda.io/en/latest/miniconda.html) for miniconda installation instructions)

* In a terminal, navigate to the repository<br>
``cd data_science_cop/``

* In a terminal, create an environment from the `requirements.yml` supplied with this repository <br>
``conda env create --file requirements.yml``

* Activate the environment <br>
``source activate data-science-cop``

* Run Jupyter Lab <br>
``jupyter lab``

* Navigate to the relevant notebook and run it.

### Met Office IT - Scitools 

* In a terminal, navigate to the repository<br>
``cd data-science-cop/``

* Load the `experimental-current` scitools environment <br>
``module load scitools/experimental-current``

* Run Jupyter Lab <br>
``jupyter lab``

* Navigate to the relevant notebook and run it.

### Pangeo
* If you do not already have access to an instance of Pangeo (such as [Panzure](https://panzure.informaticslab.co.uk/)) then contact the administrators to obtain access.

* For [Panzure](https://panzure.informaticslab.co.uk/), you can get access by emailing your GitHub username to kevin.donkers@informaticslab.co.uk.

* Once you have access and have logged onto a Pangeo, you can navigate to a terminal window.<br>
<img src="images/pangeo_terminal.png" width="300"/>

* Clone the repository using git<br>
``git clone https://github.com/MetOffice/data_science_cop.git``

* Navigate to the repo<br>
``cd data_science_cop/``

* Create a conda environment<br>
``conda env create --file requirements.yml``

* Activate the conda environment in the terminal<br>
``source activate data-science-cop``

* Install the conda environment as a Jupyter kernel<br>
``python -m ipykernel install --name "data-science-cop" --display-name "Python (data-science-cop)" --user``

* Now if you navigate to the Jupyter notebook you want to run, **Python (data-science-cop)** will be available from the list of kernels in the top right hand corner of the Notebook<br>
<img src="images/kernel_select.png" width="300"/>

### Google Colab

Instructions:

* Go to https://colab.research.google.com/. You may need to login with a Google account.

* Open a new notebook from GitHub, using the URL for the repository:<br>
``https://github.com/MetOffice/data_science_cop`` <br>
<img src="images/google_colab_open.png" width="800"/>

* When you run the notebook, the required libraries will be installed when required.



