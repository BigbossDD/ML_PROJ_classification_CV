import pandas as pd
import numpy as np

def main():
    #getting the data from this path --> C:\Users\USER\OneDrive\Desktop\PSUT\ML\ML_PROJ_classification_CV\defungi
    #my data type is JPG
    data_path = r"defungi"
    with open(data_path, 'r') as f:
        data = f.read()
    print(data[0])


if __name__ == "__main__":
    main()
