# SENSDeep
 Stacking (Stacked Generalization) Ensemble Deep Learning Method (SENSDeep) <br /> 
 https://doi.org/10.1007/s12539-022-00543-x <br />
 <br />
 <img src="https://user-images.githubusercontent.com/26387255/189489623-d10a2c44-4966-4303-bc94-2998ecdbe226.png" width=60% height=60%>
- Python 3.8 has been used when training and testing. The package requirements are listed in the requirements.txt file. The gpu versions of the relevant packages may need to be installed in order to run on the GPU.
- To download datasets and pre-trained models, use the links in the txt files in models and datasets folders. <br />
- Training and testing:
  - For window size = 7, fold = 1 and model = RNN with training dataset non_SS_PSI and testing datasets Dset 72, 164, 186, 355 and 448; <br />

  ```
  python stacked_generalization.py -p ./datasets/non_SS_PSI/datasets_7/ -us False -ad True -nvd False -pdb True -cw False -cwc True -sq False -sw False -sc False -st Normalizer -dv CPU -fn 1 -md 1
  ```

  - For window size = 7, fold = 1 and model = RNN with training dataset SS_PSI and testing datasets Dset 355 and 448; <br />

  ```
  python stacked_generalization.py -p ./datasets/SS_PSI/datasets_7/ -us False -ad True -nvd False -pdb False -cw False -cwc True -sq False -sw False -sc False -st Normalizer -dv CPU -fn 1 -md 1
  ```


- To see the prediction performance of SENSDeep on testing datasets using pre-trained models, stacking/predict_demo.ipynb can be used. <br />
