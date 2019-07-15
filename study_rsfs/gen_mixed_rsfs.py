import pandas as pd
from math import *
import numpy as np
from sklearn import preprocessing
import pdb

# algo overview
# pick percentage of molecs to take from each distribution
#      (maybe project mean rsf onto line btw mean rsfs of
#       distributions and choose % based on position of projection on line)
# sample from respective distributions
# calculate estimated mean rsf
# based on how far away from actual rsf mean, choose different percentage

# components
#   need to know how rsfs are distributed for the two different types
#   compute mean rsf from rdf

