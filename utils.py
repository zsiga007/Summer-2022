import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt



# prime generating functions
# Python program to print prime factors
 
# A function to print all prime factors of
# a given number n
def primeFactors(n):
    fact_array = []
    # Print the number of two's that divide n
    while n % 2 == 0:
        fact_array.append(2)
        n = n / 2
         
    # n must be odd at this point
    # so a skip of 2 ( i = i + 2) can be used
    for i in range(3,int(math.sqrt(n))+1,2):
         
        # while i divides n , print i and divide n
        while n % i== 0:
            fact_array.append(i)
            n = n / i
             
    # Condition if n is a prime
    # number greater than 2
    if n > 2:
        fact_array.append(n)
    return fact_array


def isPrime(n):
  fact_array = primeFactors(n)
  if len(fact_array) == 1:
    return True
  return False


#primes up to 
def primes_upto(limit):
    prime = [True] * limit
    for n in range(2, limit):
        if prime[n]:
            yield n # n is a prime
            for c in range(n*n, limit, n):
                prime[c] = False # mark composites

# encoding algorithm
def encode(batch, digits, base):
  s=batch.size(dim=0)
  t=base**torch.arange(digits-1,-1,-1).to(torch.int64)
  batch=torch.nn.functional.one_hot(torch.remainder(batch//t[None, :].to(torch.int64), base), num_classes=base)
  batch=torch.reshape(batch, (s, digits*base))
  return(batch)

def decode(batch, digits, syst):
  batch=(torch.argmax(batch, dim=2)@t)[:, None]
  return(batch)

