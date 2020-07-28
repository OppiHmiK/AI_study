from time import ctime
import random as rand

def randomName(num):
    rand.seed(ctime())
    alphas = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789'
    nums = [rep for rep in range(1, len(alphas))]

    randAlpha = rand.sample(nums, num)
    fname = ''

    for alpha in randAlpha:
        fname += alphas[alpha]
    
    return fname

if __name__ == '__main__':
    print(randomName(10))